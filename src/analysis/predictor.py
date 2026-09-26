"""Restored GapTune predictors for the fixed-predictor studies (paper App. C.5, C.6).

A repetition restores one trained GapTune(+) finetune checkpoint: the finetune
runner rebuilds the target dataset, split, loaders, frozen encoder and task from
the checkpoint's own config, then the saved encoder and task states (queries,
relevance, gates, head and the retained source contexts ``C*_s``) are loaded.
Nothing is refitted; evaluation reads the retained contexts (Eq. 18).
"""

from __future__ import annotations

import csv
import os
import re

import torch

from src.finetune.finetuner import FinetuneRunner
from src.finetune.readouts import task_native_readout
from src.finetune.utils import resolve_pretrained_checkpoint
from src.utils.paths import ensure_dir
from src.utils.run_helpers import resolve_seeds

# Settings of the analysis invocation that the checkpoint config does not override.
_INVOCATION_KEYS = ("analysis", "device")


def _pretrained_checkpoint(cfg) -> tuple[str, str | None]:
    explicit = str(cfg.analysis.pretrained_checkpoint or cfg.finetune.pretrained_checkpoint).strip()
    if explicit:
        return explicit, None
    first = cfg.clone()
    first.seed = resolve_seeds(cfg, requested_count=1)[0]  # as the finetune runtime resolves it
    path, run_name = resolve_pretrained_checkpoint(first)
    if not path:
        raise FileNotFoundError("[Analysis] Unable to resolve the pretrained checkpoint from pretrain.* / model.*.")
    return path, run_name


def finetuned_checkpoints(cfg) -> list[tuple[int, str]]:
    """``(seed, checkpoint)`` per repetition over the seeds of ``finetune.num_runs``.

    ``analysis.finetuned_checkpoint`` names the files (``{seed}`` placeholder);
    "" resolves each seed's checkpoint like the finetune runner, from the
    ``pretrain.*`` / ``model.*`` / ``finetune.*`` keys of the training command.
    """
    seeds = resolve_seeds(cfg, requested_count=int(cfg.finetune.num_runs))
    template = str(cfg.analysis.finetuned_checkpoint).strip()
    if template and "{seed}" not in template and len(seeds) > 1:
        raise ValueError("[Analysis] analysis.finetuned_checkpoint needs a {seed} placeholder when finetune.num_runs > 1.")
    paths = []
    for seed in seeds:
        if template:
            paths.append(template.format(seed=seed))
            continue
        run_cfg = cfg.clone()
        run_cfg.seed = int(seed)
        paths.append(FinetuneRunner(run_cfg, *_pretrained_checkpoint(cfg)).get_checkpoint_path_for_metrics())
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"[Analysis] Finetuned checkpoint(s) not found: {missing}")
    return list(zip(seeds, paths))


def restore_predictor(cfg, checkpoint: str) -> tuple[FinetuneRunner, dict]:
    """Return the set-up runner of *checkpoint* with its trained state, and the payload."""
    payload = torch.load(checkpoint, map_location="cpu")
    run_cfg = cfg.clone()
    saved = {key: value for key, value in payload["cfg"].items() if key not in _INVOCATION_KEYS}
    FinetuneRunner._merge_dict_into_cfg(run_cfg, saved)
    run_cfg.finetune.skip_if_exists = False
    pretrained = str(cfg.analysis.pretrained_checkpoint).strip() or run_cfg.finetune.pretrained_checkpoint
    runner = FinetuneRunner(run_cfg, pretrained, run_cfg.finetune.pretrained_run_name or None)
    if runner.finetune_method != "gaptune":
        raise ValueError(f"[Analysis] {checkpoint} is a '{runner.finetune_method}' checkpoint, not GapTune.")
    runner._setup()  # the runner's own dataset / split / loader / encoder / task construction
    objective = runner.task.objective
    if objective.task_type != "classification" or objective.label_dim != 1 or objective.output_dim < 2:
        raise ValueError("[Analysis] Fixed-predictor studies compare class decisions of single-label multiclass targets.")
    runner.model.load_state_dict(payload["model_state"])
    runner.task.load_state_dict(payload["extra"]["task_state"])
    runner.model.eval()
    runner.task.eval()
    return runner, payload


def reference_pass(task, model, data):
    """Detached unprompted pass: observations and descriptors of the prediction graphs."""
    return task.driver.forward(model, data, collect=True, projections=task.projections)


def prompted_logits(task, model, data, node_prompt, message_prompts) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-softmax head outputs and labels of the prompted pass (Eq. 13-17)."""
    node_repr = task.driver.forward(
        model, data, node_prompt=node_prompt, message_prompts=message_prompts, projections=task.projections
    ).node_repr
    representations, labels = task_native_readout(node_repr, data, task.task_level_raw)
    return task.head(representations), labels.view(-1)


def typed_prompts(node_prompt, message_prompts) -> dict[str, torch.Tensor]:
    """Prompts keyed by observation type ``N`` / ``M1``..``ML`` (inactive types omitted)."""
    prompts = {} if node_prompt is None else {"N": node_prompt}
    for layer, prompt in enumerate(message_prompts or [], start=1):
        if prompt is not None:
            prompts[f"M{layer}"] = prompt
    return prompts


def study_dir(cfg, study: str, checkpoint: str) -> str:
    """``<analysis.output_dir>/<study>/<run_tag>``; the run tag is the checkpoint stem without its seed."""
    stem = os.path.splitext(os.path.basename(checkpoint))[0]
    return str(ensure_dir(os.path.join(cfg.analysis.output_dir, study, re.sub(r"_seed\d+$", "", stem))))


def write_tsv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
