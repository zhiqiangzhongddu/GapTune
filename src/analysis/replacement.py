"""Fixed-predictor source-context replacement (paper App. C.6, Tables 7-8, Fig. 6).

Each repetition restores a trained source-available GapTune+ predictor
(``src.analysis.predictor``) and keeps it fixed: encoder, queries, relevance,
gates, head, normalisation statistics, target graphs, unprompted observations,
descriptors and evaluation objects.  Proxy collections (inverted and random,
``B`` graphs; the source-free builder with ``finetune.gaptune.proxy``) are
sampled with the predictor's source caps and pooled with its frozen queries
into replacement contexts ``C^(B)``, which replace only the retained source
contexts ``C*_s``; the identity replacement reuses ``C*_s`` itself.

Per active type ``q`` (Eq. 71): ``eps_q = max_k ||c^_k - c*_k||_2`` and
``delta_P,q`` = max over all prompted objects of all test graphs of
``||p^ - p||_2``.  Per prediction object ``delta_f = ||f(P^) - f(P)||_inf`` on
the pre-softmax logits (mean and max within a repetition), agreement of class
decisions, and the accuracy change with the paired t interval of Eq. 74 over
repetitions.  Checks: identity gives exactly 0 / 100 %, ``delta_P,q <= eps_q``
(Prop. 3.3, Eq. 72) and ``|dAcc| <= 100 - Agree``.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from src.analysis.predictor import (
    finetuned_checkpoints,
    prompted_logits,
    reference_pass,
    restore_predictor,
    study_dir,
    typed_prompts,
    write_tsv,
)
from src.analysis.stats import paired_t
from src.finetune.methods.gaptune_proxy import build_proxy_source_graphs
from src.utils.checkpoint import save_json_atomic
from src.utils.parsing import to_bool
from src.utils.pool import get_batch_vector

IDENTITY = "original"
_TOLERANCE = 1e-5  # float slack of the delta_P,q <= eps_q check, in response units


def proxy_contexts(runner, proxy_cfg, pretrain_extra) -> tuple[dict, dict]:
    """Replacement contexts ``C^(B)`` of one proxy collection, and its metadata.

    *proxy_cfg* is a ``finetune.gaptune.proxy`` node carrying the collection's
    ``mode`` and ``num_graphs``; observations are sampled like the predictor's
    own source bank (same caps and sampler stream).
    """
    task, model = runner.task, runner.model
    build_cfg = runner.cfg.clone()
    build_cfg.finetune.gaptune.proxy = proxy_cfg
    graphs, metadata = build_proxy_source_graphs(
        cfg=build_cfg,
        model=model,
        driver=task.driver,
        pretrain_cfg=runner.pretrain_cfg,
        pretrain_extra=pretrain_extra,
        device=runner.device,
    )
    banks, sampling = task.sample_source_banks(model, graphs, runner.device)
    with torch.no_grad():
        contexts = {key: prompt.source_context(banks[key].to(runner.device)) for key, prompt in task.prompt.types.items()}
    return contexts, {"proxy": metadata, **sampling}


def _set_retained(prompt, contexts: dict) -> None:
    for key, type_prompt in prompt.types.items():
        type_prompt.retained_source_context.copy_(contexts[key])


@torch.no_grad()
def replacement_measures(task, model, loader, device, replacements: dict) -> dict:
    """Eq. 71 errors, output changes, agreement and accuracies of every
    ``replacements[name]`` (contexts keyed by type) against the retained ``C*_s``."""
    prompt = task.prompt
    original = {key: type_prompt.retained_source_context.clone() for key, type_prompt in prompt.types.items()}
    active = [key for key, type_prompt in prompt.types.items() if not bool(type_prompt.source_empty)]
    delta_p = {name: dict.fromkeys(active, 0.0) for name in replacements}
    delta_f = {name: [] for name in replacements}
    agree, correct = dict.fromkeys(replacements, 0), dict.fromkeys(replacements, 0)
    original_correct = total = 0
    for data in loader:
        data = data.to(device)
        obs = reference_pass(task, model, data).obs
        batch = get_batch_vector(data)
        _set_retained(prompt, original)
        prompts = prompt(obs, batch, use_retained=True)
        logits, labels = prompted_logits(task, model, data, *prompts)
        decisions = logits.argmax(dim=-1)
        original_correct += int((decisions == labels).sum())
        total += labels.numel()
        reference = typed_prompts(*prompts)
        for name, contexts in replacements.items():
            _set_retained(prompt, contexts)
            replaced = prompt(obs, batch, use_retained=True)
            replaced_logits, _ = prompted_logits(task, model, data, *replaced)
            for key, value in typed_prompts(*replaced).items():
                delta_p[name][key] = max(delta_p[name][key], float((value - reference[key]).norm(dim=-1).max()))
            delta_f[name].append((replaced_logits - logits).abs().amax(dim=-1).cpu())
            replaced_decisions = replaced_logits.argmax(dim=-1)
            agree[name] += int((replaced_decisions == decisions).sum())
            correct[name] += int((replaced_decisions == labels).sum())
    _set_retained(prompt, original)

    original_accuracy = 100.0 * original_correct / total
    results = {}
    for name, contexts in replacements.items():
        eps = {key: float((contexts[key] - original[key]).norm(dim=-1).max()) for key in active}
        changes = torch.cat(delta_f[name])
        agreement, accuracy = 100.0 * agree[name] / total, 100.0 * correct[name] / total
        results[name] = {
            "eps": eps,
            "delta_p": delta_p[name],
            "ef_mean": float(changes.mean()),
            "ef_max": float(changes.max()),
            "agreement": agreement,
            "accuracy": accuracy,
            "original_accuracy": original_accuracy,
            "delta_acc": accuracy - original_accuracy,
            "prompt_bound_ok": all(delta_p[name][key] <= eps[key] + _TOLERANCE for key in active),
            "accuracy_bound_ok": abs(accuracy - original_accuracy) <= 100.0 - agreement + 1e-9,
        }
    return results


def _rows(seed: int, measures: dict) -> list[dict]:
    rows = []
    for name, m in measures.items():
        mode, _, budget = name.partition("_B")
        row = {"seed": seed, "contexts": mode, "budget": budget or "-", "ec": max(m["eps"].values(), default=0.0)}
        row.update({key: m[key] for key in ("ef_mean", "ef_max", "agreement", "accuracy", "delta_acc")})
        for key in m["eps"]:
            row[f"eps_{key}"], row[f"delta_p_{key}"] = m["eps"][key], m["delta_p"][key]
        checks = m["prompt_bound_ok"] and m["accuracy_bound_ok"]
        if name == IDENTITY:
            checks = checks and row["ec"] == 0 and m["ef_max"] == 0 and m["agreement"] == 100.0
            checks = checks and all(value == 0.0 for value in m["delta_p"].values())
        row["checks_ok"] = bool(checks)
        rows.append(row)
    return rows


def _summary(rows: list[dict]) -> list[dict]:
    """Means over repetitions; the accuracy change gets the Eq. 74 interval."""
    summary = []
    for condition in dict.fromkeys((row["contexts"], row["budget"]) for row in rows):
        group = [row for row in rows if (row["contexts"], row["budget"]) == condition]
        interval = paired_t([row["delta_acc"] for row in group])
        entry = {"contexts": condition[0], "budget": condition[1], "repetitions": len(group)}
        for key in group[0]:
            if key not in ("seed", "contexts", "budget", "delta_acc", "checks_ok"):
                entry[key] = float(np.mean([row[key] for row in group]))
        entry.update(delta_acc=interval["mean"], delta_acc_low=interval["low"], delta_acc_high=interval["high"])
        entry["checks_ok"] = all(row["checks_ok"] for row in group)
        summary.append(entry)
    return summary


def run_replacement(cfg) -> int:
    """``analysis.study replacement``: write per-seed JSON/TSV and summary.tsv; 1 if a check fails."""
    # Deterministic scatter kernels: the identity replacement recomputes the original outputs exactly.
    torch.use_deterministic_algorithms(True, warn_only=True)
    repetitions = finetuned_checkpoints(cfg)
    out_dir = study_dir(cfg, "replacement", repetitions[0][1])
    rows = []
    for seed, checkpoint in repetitions:
        runner, payload = restore_predictor(cfg, checkpoint)
        value_mode = runner.task.method_cfg.value_mode
        if value_mode in ("target", "free"):
            raise ValueError(f"[Analysis] value_mode={value_mode} has no source contexts to replace.")
        if not to_bool(runner.task.method_cfg.plus):
            raise ValueError("[Analysis] App. C.6 replaces the source contexts of a GapTune+ (plus True) checkpoint.")
        pretrain_extra = torch.load(runner.pretrained_checkpoint, map_location="cpu").get("extra") or {}
        replacements = {IDENTITY: {key: p.retained_source_context.clone() for key, p in runner.task.prompt.types.items()}}
        proxies = {}
        for mode in cfg.analysis.replacement.proxy_modes:
            for budget in cfg.analysis.replacement.budgets:
                proxy_cfg = cfg.finetune.gaptune.proxy.clone()
                proxy_cfg.mode, proxy_cfg.num_graphs = str(mode), int(budget)
                name = f"{mode}_B{budget}"
                replacements[name], proxies[name] = proxy_contexts(runner, proxy_cfg, pretrain_extra)
        measures = replacement_measures(runner.task, runner.model, runner.test_loader, runner.device, replacements)
        seed_rows = _rows(seed, measures)
        save_json_atomic(
            os.path.join(out_dir, f"rep{seed}.json"),
            {
                "seed": seed,
                "checkpoint": checkpoint,
                "checkpoint_test_acc": 100.0 * float((payload.get("metrics") or {}).get("test_acc", float("nan"))),
                "measures": measures,
                "proxies": proxies,
            },
        )
        write_tsv(os.path.join(out_dir, f"rep{seed}.tsv"), seed_rows)
        rows.extend(seed_rows)
    summary = _summary(rows)
    write_tsv(os.path.join(out_dir, "summary.tsv"), summary)
    for entry in summary:
        print(
            f"[Analysis][replacement] {entry['contexts']:>9} B={entry['budget']:>2} EC={entry['ec']:.4f} "
            f"Ef_mean={entry['ef_mean']:.4f} Ef_max={entry['ef_max']:.4f} agree={entry['agreement']:.2f}% "
            f"acc={entry['accuracy']:.2f}% dAcc={entry['delta_acc']:+.2f} "
            f"[{entry['delta_acc_low']:+.2f}, {entry['delta_acc_high']:+.2f}] checks_ok={entry['checks_ok']}"
        )
    print(f"[Analysis][replacement] Results in {out_dir}")
    return 0 if all(entry["checks_ok"] for entry in summary) else 1
