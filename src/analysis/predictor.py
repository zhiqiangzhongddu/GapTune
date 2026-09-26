"""Restored GapTune predictors for the fixed-predictor studies (paper App. C.5, C.6).

A repetition restores one trained GapTune(+) finetune checkpoint: the finetune
runner rebuilds the target dataset, split, loaders, frozen encoder and task from
the checkpoint's own config, then the saved encoder and task states (queries,
relevance, gates, head and the retained source contexts ``C*_s``) are loaded.
Nothing is refitted; evaluation reads the retained contexts (Eq. 18).
The module also holds the helpers the studies share: the pretrained-checkpoint
lookup, the full-graph setup of App. A / C.7 and the output layout.
"""

from __future__ import annotations

import csv
import os
import re

import torch

from src.data_loader import create_dataset, dataset_info
from src.finetune.finetuner import FinetuneRunner
from src.finetune.readouts import task_native_readout
from src.finetune.utils import resolve_pretrained_checkpoint
from src.model import build_encoder_from_cfg
from src.utils.checkpoint import cfg_to_dict
from src.utils.dataset_helpers import populate_dataset_cfg_from_meta
from src.utils.paths import ensure_dir
from src.utils.run_helpers import resolve_seeds

# Settings of the analysis invocation that the checkpoint config does not override.
_INVOCATION_KEYS = ("analysis", "device")


def pretrained_checkpoint(cfg, dataset: str | None = None) -> tuple[str | None, str | None]:
    """``(path, run_name)`` of the frozen pretrained encoder; ``(None, None)`` unless ``path`` is a file.

    ``analysis.pretrained_checkpoint`` (run name ``None``), else the finetune
    runner's lookup from ``pretrain.*`` / ``model.*`` for the first seed, with
    ``pretrain.dataset.name`` set to ``dataset`` when given.
    """
    path, run_name = str(cfg.analysis.pretrained_checkpoint or "").strip(), None
    if not path:
        lookup = cfg.clone()
        lookup.seed = resolve_seeds(cfg, requested_count=1)[0]  # as the finetune runtime resolves it
        if dataset is not None:
            lookup.pretrain.dataset.name = dataset
        path, run_name = resolve_pretrained_checkpoint(lookup)
    return (path, run_name) if path and os.path.isfile(path) else (None, None)


def full_graph_setup(cfg, name: str, path: str, device: torch.device):
    """``(dataset, encoder)``: dataset ``name`` as one full graph and the frozen vanilla encoder of ``path``.

    Node level, not induced. Merges the checkpoint's ``model.*`` settings and
    the dataset's metadata into ``cfg`` in place; the encoder is in eval mode
    on ``device``.
    """
    payload = torch.load(path, map_location="cpu")
    FinetuneRunner._merge_dict_into_cfg(cfg.model, cfg_to_dict(payload.get("cfg") or {}).get("model") or {})
    ds = cfg.finetune.dataset
    ds.name, ds.task_level, ds.induced = name, "node", False
    dataset = create_dataset(
        name=name,
        root=ds.root,
        task_level="node",
        feat_reduction=ds.feat_reduction,
        feat_reduction_dim=ds.feat_reduction_svd_dim,
        feature_svd_dir=ds.feature_svd_dir,
        induced=False,
    )
    populate_dataset_cfg_from_meta(cfg.model, ds, dataset_info(dataset, "node", name))
    encoder = build_encoder_from_cfg(cfg, cfg.model.in_dim)
    encoder.load_state_dict(payload["model_state"])
    return dataset, encoder.to(device).eval().requires_grad_(False)


def finetuned_checkpoints(cfg) -> list[tuple[int, str]]:
    """``(seed, checkpoint)`` per repetition over the seeds of ``finetune.num_runs``.

    ``analysis.finetuned_checkpoint`` names the files (``{seed}`` placeholder;
    a single file without it is labelled with the seed it was trained with);
    "" resolves each seed's checkpoint like the finetune runner, from the
    ``pretrain.*`` / ``model.*`` / ``finetune.*`` keys of the training command.
    """
    seeds = resolve_seeds(cfg, requested_count=int(cfg.finetune.num_runs))
    template = str(cfg.analysis.finetuned_checkpoint).strip()
    if template and "{seed}" not in template and len(seeds) > 1:
        raise ValueError("[Analysis] analysis.finetuned_checkpoint needs a {seed} placeholder when finetune.num_runs > 1.")
    if not template:
        lookup = cfg.clone()  # the training command's explicit finetune.pretrained_checkpoint counts as well
        lookup.analysis.pretrained_checkpoint = cfg.analysis.pretrained_checkpoint or cfg.finetune.pretrained_checkpoint
        pretrained = pretrained_checkpoint(lookup)
        if pretrained[0] is None:
            raise FileNotFoundError(
                "[Analysis] Unable to resolve the pretrained checkpoint (analysis.pretrained_checkpoint / "
                "finetune.pretrained_checkpoint, else pretrain.* / model.*)."
            )
    paths = []
    for seed in seeds:
        if template:
            paths.append(template.format(seed=seed))
            continue
        run_cfg = cfg.clone()
        run_cfg.seed = int(seed)
        paths.append(FinetuneRunner(run_cfg, *pretrained).get_checkpoint_path_for_metrics())
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"[Analysis] Finetuned checkpoint(s) not found: {missing}")
    if template and "{seed}" not in template:
        seeds = [int(torch.load(template, map_location="cpu")["cfg"]["seed"])]
    return list(zip(seeds, paths))


def restore_predictor(cfg, checkpoint: str) -> tuple[FinetuneRunner, dict]:
    """Return the set-up runner of *checkpoint* with its trained state, and the payload."""
    payload = torch.load(checkpoint, map_location="cpu")
    run_cfg = cfg.clone()
    saved = {key: value for key, value in payload["cfg"].items() if key not in _INVOCATION_KEYS}
    FinetuneRunner._merge_dict_into_cfg(run_cfg, saved)
    run_cfg.seed = int(saved["seed"])  # set per run by the finetune runtime, not a declared key
    run_cfg.finetune.skip_if_exists = False
    source = payload["extra"]["pretrained_from"]
    pretrained = str(cfg.analysis.pretrained_checkpoint).strip() or source["checkpoint"]
    runner = FinetuneRunner(run_cfg, pretrained, source["run_name"])
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
