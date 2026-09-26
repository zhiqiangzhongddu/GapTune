"""App. A: prompt transferability under controlled graph shifts (paper Fig. 1).

Photo, full-graph node classification, frozen Photo/GCN/DGI checkpoint; the
support of repetition ``s`` is the ``(5, 0, 1)`` few-shot split of seed ``s``
under ``analysis.split_root`` (fixed across views). The views, the context-gap
samples and the prompt and head initialisations draw from three separate
streams seeded by ``np.random.SeedSequence(s).generate_state(3)``, so they
share no draws with each other, with other repetitions or with the split
generated from seed ``s``.
Per repetition:

- views: :func:`controlled_views` (``CONTROL`` + family x strength);
- context gaps: :func:`view_context_gaps` on the unprompted vanilla encoder;
- EdgePrompt+ on the repo's prompt-aware encoder (``FinetuneEdgePrompt``
  prompt, head, objective and optimiser; encoder frozen in eval mode, GCN
  normalisation recomputed per adjacency), initialised after ``set_seed``:
  a donor prompt + temporary head fit on the unperturbed graph; a fresh target
  prompt + temporary head fit on every view, the control included; then two
  fresh linear heads with one shared initialisation fit on the view above the
  frozen target and donor prompts (frozen prompts recompute on the view).
  Every fit runs ``finetune.epochs`` full-graph updates and restores the
  checkpoint the finetune runner's monitor selects (training loss without a
  validation split).
- ``R = 100 (A_match - A_transfer)`` on test accuracy, ``E = R - R_0`` with
  ``R_0`` the control's ``R`` (Eq. 28-29).

Summary (App. A.5): curve = mean ``E`` per (family, alpha); mean
within-repetition Spearman of ``(delta_q, E)`` over the nonzero conditions
(Eq. 30); percentile bootstrap over complete repetitions for both; pooled
within-family correlations (descriptive, no interval).
Outputs in ``<analysis.output_dir>/transfer/<checkpoint stem without its seed>/``:
``rep<seed>.json``, ``conditions.tsv`` (one row per repetition and view),
``curve.tsv`` and ``correlations.tsv``.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
from torch_geometric.data import Data

from src.finetune.encoders.edgeprompt import build_prompt_encoder
from src.finetune.frozen_load import check_frozen_encoder_load
from src.finetune.methods.edgeprompt import FinetuneEdgePrompt
from src.finetune.monitoring import resolve_finetune_monitor_spec
from src.utils.checkpoint import save_json_atomic
from src.utils.dataset_helpers import make_workflow_loaders
from src.utils.monitoring import is_metric_improved, monitor_uses_train_split, resolve_monitor_value
from src.utils.random import set_seed

from .context_gap import view_context_gaps
from .perturbations import CONTROL, FAMILIES, controlled_views
from .predictor import full_graph_setup, pretrained_checkpoint, study_dir, write_tsv
from .stats import mean_spearman, repetition_bootstrap, spearman

DATASET = "photo"
SPLIT = (5, 0.0, 1.0)  # five support nodes per class, no validation, the rest is test
GAP_TYPES = ("H", "M")


def _fit(task, model, graph, monitor) -> None:
    """``finetune.epochs`` full-graph updates; restore the state the runner's monitor selects."""
    device = graph.x.device
    optimizers = task.build_optimizers(model)
    best, best_state = monitor.best_metric, None
    for _ in range(int(task.cfg.finetune.epochs)):
        loss, logs = task.train_epoch(model, [graph], device, optimizers)
        val = {} if monitor_uses_train_split(monitor.name) else task.evaluate_split(
            model, [graph], device, "val", "val_mask"
        )
        value = resolve_monitor_value(monitor.name, train_loss=loss, train_logs=logs, val_metrics=val, test_metrics={})
        if monitor.name is None or is_metric_improved(value, best, monitor.mode):
            best, best_state = value, copy.deepcopy(task.state_dict())
    if best_state is not None:
        task.load_state_dict(best_state)


def _fit_prompt(cfg, model, graph, monitor) -> FinetuneEdgePrompt:
    """A fresh EdgePrompt+ prompt and temporary head fit on ``graph``; the prompt is then frozen."""
    task = FinetuneEdgePrompt(cfg).to(graph.x.device)
    _fit(task, model, graph, monitor)
    task.prompt.requires_grad_(False)
    return task


def _fresh_head_accuracy(task, model, graph, head_state, monitor) -> float:
    """Test accuracy of a head initialised from ``head_state`` and fit above ``task``'s frozen prompt."""
    task.classifier.load_state_dict(head_state)
    _fit(task, model, graph, monitor)
    return task.evaluate_split(model, [graph], graph.x.device, "test", "test_mask")["test_acc"]


def run_repetition(cfg, model, encoder, data, seed: int) -> list[dict]:
    """One row per view (``CONTROL`` first): eta, context gaps, accuracies, ``R`` and ``E``.

    ``model`` is the frozen EdgePrompt-aware encoder and ``encoder`` the same
    frozen weights as the vanilla encoder; both in eval mode. ``data`` carries
    labels and the train / val / test masks.
    """
    view_seed, gap_seed, init_seed = (int(s) for s in np.random.SeedSequence(seed).generate_state(3))
    views = controlled_views(
        data.x, data.edge_index, list(cfg.analysis.transfer.strengths), generator=torch.Generator().manual_seed(view_seed)
    )
    gaps = view_context_gaps(
        encoder,
        views,
        node_budget=int(cfg.analysis.node_budget),
        message_budget=int(cfg.analysis.message_budget),
        generator=torch.Generator().manual_seed(gap_seed),
    )
    monitor = resolve_finetune_monitor_spec(
        cfg,
        task_level="node",
        label_dim=int(cfg.finetune.dataset.label_dim or 1),
        few_shot_without_validation=not bool(data.val_mask.any()),
        task_cls=FinetuneEdgePrompt,
    )
    labels_and_split = {name: data[name] for name in ("y", "train_mask", "val_mask", "test_mask")}
    graphs = {key: Data(x=view.x, edge_index=view.edge_index, **labels_and_split) for key, view in views.items()}

    set_seed(init_seed)
    donor = _fit_prompt(cfg, model, graphs[CONTROL], monitor)
    rows = []
    for key, view in views.items():
        target = _fit_prompt(cfg, model, graphs[key], monitor)
        target.classifier.reset_parameters()  # one fresh head initialisation for both fits
        head_state = copy.deepcopy(target.classifier.state_dict())
        acc_match = _fresh_head_accuracy(target, model, graphs[key], head_state, monitor)
        acc_transfer = _fresh_head_accuracy(donor, model, graphs[key], head_state, monitor)
        rows.append({
            "seed": seed,
            "family": key[0],
            "alpha": key[1],
            "swaps": view.swaps,
            "eta": view.eta,
            "delta_H": gaps[key]["H"],
            "delta_M": gaps[key]["M"],
            "acc_match": acc_match,
            "acc_transfer": acc_transfer,
            "R": 100 * (acc_match - acc_transfer),
        })
    r0 = next(row["R"] for row in rows if (row["family"], row["alpha"]) == CONTROL)
    for row in rows:
        row["E"] = row["R"] - r0
    return rows


def summarize(rows: list[dict], num_samples: int) -> tuple[list[dict], list[dict]]:
    """Curve rows (mean ``E`` per nonzero condition) and correlation rows, with bootstrap intervals."""
    seeds = list(dict.fromkeys(row["seed"] for row in rows))
    reps = [[r for r in rows if r["seed"] == s and r["family"] != CONTROL[0]] for s in seeds]
    conditions = [(r["family"], r["alpha"]) for r in reps[0]]

    def statistic(sample):
        excess = np.array([[r["E"] for r in rep] for rep in sample])
        rhos = [
            mean_spearman([[r[f"delta_{q}"] for r in rep] for rep in sample], [[r["E"] for r in rep] for rep in sample])
            for q in GAP_TYPES
        ]
        return np.concatenate([excess.mean(axis=0), rhos])

    point = statistic(reps)
    low, high = repetition_bootstrap(reps, statistic, num_samples=num_samples)
    curve = []
    for i, (family, alpha) in enumerate(conditions):
        cells = [rep[i] for rep in reps]
        curve.append({
            "family": family,
            "alpha": alpha,
            "n": len(cells),
            "E_mean": point[i],
            "E_low": low[i],
            "E_high": high[i],
            **{f"{k}_mean": float(np.mean([c[k] for c in cells])) for k in ("eta", "delta_H", "delta_M")},
        })
    correlations = []
    for j, q in enumerate(GAP_TYPES, start=len(conditions)):
        # n = repetitions whose correlation is defined (Eq. 30 excludes the others)
        n = sum(np.isfinite(spearman([r[f"delta_{q}"] for r in rep], [r["E"] for r in rep])) for rep in reps)
        correlations.append(
            {"scope": "within_repetition", "q": q, "n": int(n), "rho": point[j], "low": low[j], "high": high[j]}
        )
    for family in FAMILIES:
        pooled = [r for rep in reps for r in rep if r["family"] == family]
        for q in GAP_TYPES:
            rho = spearman([r[f"delta_{q}"] for r in pooled], [r["E"] for r in pooled])
            correlations.append(
                {"scope": f"pooled_{family}", "q": q, "n": len(pooled), "rho": rho, "low": np.nan, "high": np.nan}
            )
    return curve, correlations


def _prompt_encoder(cfg, encoder, device: torch.device):
    """The EdgePrompt-aware encoder holding the vanilla ``encoder``'s frozen weights, in eval mode."""
    model = build_prompt_encoder(cfg, cfg.model.in_dim)
    missing, unexpected = model.load_state_dict(encoder.state_dict(), strict=False)
    check_frozen_encoder_load(
        encoder=model,
        missing_keys=missing,
        unexpected_keys=unexpected,
        min_match_ratio=float(cfg.finetune.frozen_load_min_match_ratio),
        require_frozen=True,
    )
    return model.to(device).eval().requires_grad_(False)


def run_transfer(cfg) -> int:
    """``analysis.study transfer``: every repetition of ``analysis.repetitions``, then the summary."""
    cfg = cfg.clone()
    path, _ = pretrained_checkpoint(cfg)
    if path is None:
        print("[Analysis][transfer] Unable to resolve the pretrained checkpoint (analysis.pretrained_checkpoint).")
        return 1
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    dataset, encoder = full_graph_setup(cfg, DATASET, path, device)
    model = _prompt_encoder(cfg, encoder, device)
    out_dir = study_dir(cfg, "transfer", path)
    print(f"[Analysis][transfer] checkpoint={path} output={out_dir}")

    rows = []
    for seed in (int(s) for s in cfg.analysis.repetitions):
        loader, _, _ = make_workflow_loaders(
            dataset=dataset,
            dataset_name=DATASET,
            task_level_raw="node",
            effective_task_level="node",
            batch_size=1,
            num_workers=0,
            split=SPLIT,
            seed=seed,
            induced=False,
            split_root=cfg.analysis.split_root,
        )
        rep = run_repetition(cfg, model, encoder, next(iter(loader)).to(device), seed)
        save_json_atomic(
            os.path.join(out_dir, f"rep{seed}.json"), {"checkpoint": path, "seed": seed, "conditions": rep}
        )
        print(f"[Analysis][transfer] seed={seed} " + " ".join(
            f"{r['family']}@{r['alpha']:g}:E={r['E']:.2f}" for r in rep
        ))
        rows.extend(rep)

    curve, correlations = summarize(rows, int(cfg.analysis.transfer.bootstrap_samples))
    write_tsv(os.path.join(out_dir, "conditions.tsv"), rows)
    write_tsv(os.path.join(out_dir, "curve.tsv"), curve)
    write_tsv(os.path.join(out_dir, "correlations.tsv"), correlations)
    for r in correlations[: len(GAP_TYPES)]:
        print(f"[Analysis][transfer] mean Spearman rho_{r['q']}={r['rho']:.3f} [{r['low']:.3f}, {r['high']:.3f}]")
    return 0
