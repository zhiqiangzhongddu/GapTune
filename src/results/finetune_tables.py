"""Backfill finetune result rows and render LaTeX summary tables.

The finetune Slurm jobs often skip completed checkpoints.  When
``save_results.save_skipped`` is false, those skipped jobs print summaries but
do not append rows to ``outputs/results/finetune.tsv``.  This module rebuilds
those missing rows from saved finetune artifacts, then renders the paper tables
from the TSV as the single source of truth.

Three kinds of LaTeX tables are produced per shot (f5, f100):

* ``finetune_<model>_<shot>_table.tex`` — one per backbone model.
* ``finetune_backbone_<shot>_table.tex`` — cross-backbone summary that, for
  each logical cell, reports the best backbone's ``mean±std`` with the
  winning backbone as a superscript.
* ``finetune_<shot>_table.tex`` — cross-method summary with one row per
  pretrain method in the supervised section and one row per prompt method in
  the prompting section; each cell picks the best combination across all
  backbones (and, for the prompting section, across all pretrain methods).
* ``finetune_<method>_<shot>_table.tex`` — one per finetune method
  (supervised + each prompt method): sections per pretrain method with one
  row per backbone model (the per-backbone tables regrouped by method).

``test_mae`` cells pick the lowest value; all other metrics pick the highest.

Cells are keyed by the pretrain dataset as well (:class:`CellKey`), so
cross-dataset rows (e.g. ZINC/PubMed checkpoints) never collide with
same-dataset rows; the grids above are the same-dataset ones (pretrain
dataset == target dataset).

The GapTune paper tables are rendered as well; bold/underline mark the
best/second distinct displayed mean per column within a ranking group:

* ``finetune_cross_<shot>_table.tex`` — Tables 1/14: one block per
  :data:`CROSS_DATASET_SOURCES` checkpoint with its architecture-matched
  scratch control (from ``train.tsv``), the twelve baselines, GapTune and
  GapTune+; ranked within each block.
* ``finetune_same_<shot>_table.tex`` — Tables 2/15: the seven scratch
  controls, then each baseline and GapTune+ at its best observed test mean
  over all backbones and pretrain methods (paper Eq. 78-79).
* ``finetune_ablation_<name>_table.tex`` — Tables 3-6 (:data:`ABLATION_TABLES`)
  on the ZINC/GCN/EdgePred checkpoint.

GapTune ablation arms are launched with explicit ``finetune.gaptune.*``
overrides (``gaptune_plus False`` for the proxy arms), which the result
TSV records as columns.  Their non-default values form the cell's
``variant`` key component, so ablation rows never overwrite the main
GapTune/GapTune+ cells (``variant == ""``).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import io
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.results import train_tables
from src.results.metric_policy import TABLE_TASKS, eval_metric

if False:  # pragma: no cover — type-only; torch deps are lazy-loaded inside backfill_model().
    from src.finetune.finetuner import FinetuneRunner


RESULTS_DIR = Path("outputs/results")
RESULTS_TSV = RESULTS_DIR / "finetune.tsv"
SLURM_DIR = Path("slurm")
LOG_PREFIX = "[Results][Finetune]"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label: str
    task_label: str
    f5_split: tuple[float, ...]
    f100_split: tuple[float, ...]

    @property
    def metric(self) -> str:
        return eval_metric(self.name, *TABLE_TASKS[self.task_label])


@dataclass(frozen=True)
class MethodSpec:
    method: str
    label: str


@dataclass(frozen=True)
class FinetuneSpec:
    method: str
    label: str
    plus: bool | None = None


@dataclass(frozen=True)
class CellKey:
    model: str
    dataset: str
    split: tuple[float, ...]
    pretrain_dataset: str
    pretrain_method: str
    finetune_method: str
    plus: bool | None
    variant: str = ""  # non-default finetune.gaptune.* settings of a GapTune ablation arm


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("photo", "Photo", "NC", (5.0, 0.0, 1.0), (100.0, 0.0, 1.0)),
    DatasetSpec("ogbn-arxiv", "Ogbn-arxiv", "NC", (5.0, 0.0, 1.0), (100.0, 0.0, 1.0)),
    DatasetSpec("dblp", "DBLP", "LP", (0.05, 0.1, 0.1), (0.1, 0.05, 0.1)),
    DatasetSpec("airports", "Airports", "NC", (5.0, 0.0, 1.0), (100.0, 0.0, 1.0)),
    DatasetSpec("chameleon", "Chameleon", "NC", (5.0, 0.0, 1.0), (100.0, 0.0, 1.0)),
    DatasetSpec("cornell", "Cornell", "LP", (0.05, 0.1, 0.1), (0.1, 0.05, 0.1)),
    DatasetSpec("qm7b", "QM7b", "GR", (5.0, 0.0, 1.0), (100.0, 0.0, 1.0)),
    DatasetSpec("toxcast", "Toxcast", "GC", (5.0, 0.0, 1.0), (100.0, 0.0, 1.0)),
    DatasetSpec("mnist", "MNIST", "GC", (5.0, 0.0, 1.0), (100.0, 0.0, 1.0)),
)

PRETRAIN_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("attr_masking", "AttrMasking"),
    MethodSpec("context_pred", "ContextPred"),
    MethodSpec("dgi", "DGI"),
    MethodSpec("edge_pred", "EdgePred"),
    MethodSpec("graphcl", "GraphCL"),
    MethodSpec("infograph", "InfoGraph"),
)

PROMPT_METHODS: tuple[FinetuneSpec, ...] = (
    FinetuneSpec("all_in_one", "All-in-One"),
    FinetuneSpec("edgeprompt", "EdgePrompt", False),
    FinetuneSpec("edgeprompt", "EdgePrompt Plus", True),
    FinetuneSpec("gaptune", "GapTune", False),
    FinetuneSpec("gaptune", "GapTune Plus", True),
    FinetuneSpec("gpf", "GPF", False),
    FinetuneSpec("gpf", "GPF Plus", True),
    FinetuneSpec("gppt", "GPPT"),
    FinetuneSpec("graphprompt", "GraphPrompt", False),
    FinetuneSpec("graphprompt", "GraphPrompt Plus", True),
    FinetuneSpec("igap", "IGAP"),
    FinetuneSpec("mtg", "MTG"),
    FinetuneSpec("pronog", "ProNoG"),
    FinetuneSpec("supt", "SUPT"),
)

#: Supervised fine-tune rendered as its own per-method table alongside the
#: prompt methods in the per-method table family.
SUPERVISED_TABLE_SPEC = FinetuneSpec("supervised", "Supervised Fine-tune")

#: Cell-key method for the head-only control (``finetune.method supervised``
#: with ``finetune.supervised.freeze_encoder True``), so head-only rows never
#: overwrite full fine-tuning (``supervised``) cells.
HEAD_ONLY_METHOD = "head_only"

MODEL_LABELS = {
    "gcn": "GCN",
    "gin": "GIN",
    "gat": "GAT",
    "fagcn": "FAGCN",
    "h2gcn": "H2GCN",
    "nodeformer": "NodeFormer",
    "transformer": "Transformer",
}

PLUS_COLUMNS = {
    "edgeprompt": "finetune.edgeprompt.plus",
    "gaptune": "finetune.gaptune.plus",
    "gpf": "finetune.gpf.plus",
    "graphprompt": "finetune.graphprompt.plus",
}

# Early finetune.tsv exports stored every prompt method's plus flag in this
# column.  New rows use method-specific columns; table matching accepts both.
LEGACY_PLUS_COLUMN = "finetune.edgeprompt.plus"
STATUS_COLUMNS = ("result_status", "status", "failure_reason")
OOM_STATUS = "OOM"
OOM_STATUS_VALUES = {
    "cuda oom",
    "cuda_oom",
    "cuda out of memory",
    "out of memory",
    "out_of_memory",
    "outofmemoryerror",
    "oom",
}

GAPTUNE_COLUMN_PREFIX = "finetune.gaptune."
_GAPTUNE_DEFAULTS = set_cfg(CN()).finetune.gaptune


@dataclass(frozen=True)
class SourceSpec:
    """A cross-dataset source checkpoint block (paper Tables 1 and 14)."""

    dataset: str
    label: str
    model: str
    pretrain_method: str


CROSS_DATASET_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("zinc", "ZINC", "gcn", "edge_pred"),
    SourceSpec("pubmed", "PubMed", "gin", "graphcl"),
)
#: Checkpoint of the GapTune ablations (paper App. C).
ABLATION_SOURCE = CROSS_DATASET_SOURCES[0]

#: Paper row order of the twelve prompting baselines (Tables 1, 2, 14, 15).
_BASELINE_ORDER = ("all_in_one", "edgeprompt", "gpf", "gppt", "graphprompt", "pronog", "igap", "mtg", "supt")
BASELINE_METHODS: tuple[FinetuneSpec, ...] = tuple(
    spec for method in _BASELINE_ORDER for spec in PROMPT_METHODS if spec.method == method
)
GAPTUNE_METHODS: tuple[FinetuneSpec, ...] = tuple(spec for spec in PROMPT_METHODS if spec.method == "gaptune")


@dataclass(frozen=True)
class GapTuneArm:
    """One ablation row: the GapTune+ flag and its non-default ``finetune.gaptune.*`` settings."""

    label: str
    settings: tuple[tuple[str, Any], ...] = ()
    plus: bool = True

    @property
    def variant(self) -> str:
        return _gaptune_variant(dict(self.settings), plus=self.plus)


@dataclass(frozen=True)
class AblationTable:
    name: str
    caption: str
    columns: tuple[tuple[str, str], ...]  # (dataset name, shot)
    arms: tuple[GapTuneArm, ...]


_ABLATION_TARGETS = (("photo", "f5"), ("chameleon", "f5"), ("dblp", "f5"), ("mnist", "f5"))
_HEAD_ONLY_ARM = GapTuneArm("Head only", (("prompt_locations", "none"),))
_FREE = ("value_mode", "free")

ABLATION_TABLES: tuple[AblationTable, ...] = (
    AblationTable(
        "value",
        "Prompt-value ablation (paper Table 3).",
        _ABLATION_TARGETS,
        (
            _HEAD_ONLY_ARM,
            GapTuneArm("Target", (("value_mode", "target"),)),
            GapTuneArm("Source", (("value_mode", "source"),)),
            GapTuneArm("Paired mean", (("value_mode", "paired_mean"),)),
            GapTuneArm("Free vectors", (_FREE,)),
            GapTuneArm("GapTune Plus"),
        ),
    ),
    AblationTable(
        "insertion",
        "Insertion ablation; free/gap parameter counts match within each pattern (paper Table 4).",
        _ABLATION_TARGETS,
        (
            _HEAD_ONLY_ARM,
            GapTuneArm("Free values: node only", (_FREE, ("prompt_locations", "node"))),
            GapTuneArm("Free values: message only", (_FREE, ("prompt_locations", "message"))),
            GapTuneArm("Free values: node + message", (_FREE,)),
            GapTuneArm("Gap values: node only", (("prompt_locations", "node"),)),
            GapTuneArm("Gap values: message only", (("prompt_locations", "message"),)),
            GapTuneArm("GapTune Plus: node + message"),
        ),
    ),
    AblationTable(
        "composition",
        "Shared queries and local signed composition (paper Table 5).",
        (("photo", "f5"), ("photo", "f100"), ("chameleon", "f5"), ("chameleon", "f100")),
        (
            GapTuneArm("Frozen shared queries", (("query_mode", "frozen"),)),
            GapTuneArm("Untied source/target queries", (("query_mode", "untied"),)),
            GapTuneArm("Uniform mixture weights", (("mixture", "uniform"),)),
            GapTuneArm("Learned global mixture weights", (("mixture", "global"),)),
            GapTuneArm("Nonnegative gates", (("gate", "nonnegative"),)),
            GapTuneArm("GapTune Plus"),
        ),
    ),
    AblationTable(
        "source",
        "Source-context construction; $B$ is the number of proxy graphs (paper Table 6).",
        _ABLATION_TARGETS,
        (
            GapTuneArm("Random proxies ($B=4$)", (("proxy.mode", "random"), ("proxy.num_graphs", 4)), plus=False),
            GapTuneArm("Random proxies ($B=16$)", (("proxy.mode", "random"),), plus=False),
            GapTuneArm("Random proxies ($B=64$)", (("proxy.mode", "random"), ("proxy.num_graphs", 64)), plus=False),
            GapTuneArm("Inverted proxies ($B=4$)", (("proxy.num_graphs", 4),), plus=False),
            GapTuneArm("GapTune ($B=16$)", plus=False),
            GapTuneArm("Inverted proxies ($B=64$)", (("proxy.num_graphs", 64),), plus=False),
            GapTuneArm("GapTune Plus (source available)"),
        ),
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    results_tsv = Path(args.results_tsv)
    results_dir = Path(args.results_dir)
    models = _resolve_models(args.models)

    if args.backfill:
        total = 0
        for model in models:
            total += backfill_model(
                model=model,
                results_tsv=results_tsv,
                results_dir=results_dir,
                tasks_tsv=Path(args.tasks_tsv_template.format(model=model)),
                dry_run=bool(args.dry_run),
                max_cells=args.max_backfill_cells,
                verbose_runner=bool(args.verbose_runner),
                min_seeds=args.min_seeds,
            ).get("appended", 0)
        print(f"{LOG_PREFIX} Backfill appended rows: {total}")

    if args.render:
        rendered = render_tables(
            models=models,
            results_tsv=results_tsv,
            results_dir=results_dir,
            train_results_tsv=Path(args.train_results_tsv),
            dry_run=bool(args.dry_run),
        )
        print(f"{LOG_PREFIX} Rendered tables: {rendered}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill finetune.tsv from saved artifacts and render finetune LaTeX tables."
    )
    parser.add_argument(
        "--models",
        default="auto",
        help="Comma-separated model names, or 'auto' for all backbones in MODEL_LABELS.",
    )
    parser.add_argument("--results-tsv", default=str(RESULTS_TSV))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--train-results-tsv",
        default=str(train_tables.RESULTS_TSV),
        help="train.tsv supplying the scratch-control rows of the cross- and same-dataset paper tables.",
    )
    parser.add_argument("--tasks-tsv-template", default=str(SLURM_DIR / "finetune.tsv"))
    parser.add_argument("--max-backfill-cells", type=int, default=0, help="Stop after this many appended rows (0 = no limit).")
    parser.add_argument("--min-seeds", type=int, default=0, help="Minimum seeds required to append a row (0 = require all requested seeds).")
    parser.add_argument("--dry-run", action="store_true", help="Report actions without writing TSV or TeX files.")
    parser.add_argument("--verbose-runner", action="store_true", help="Keep FinetuneRunner skip/setup messages visible.")
    parser.add_argument("--backfill", dest="backfill", action="store_true", default=True)
    parser.add_argument("--no-backfill", dest="backfill", action="store_false")
    parser.add_argument("--render", dest="render", action="store_true", default=True)
    parser.add_argument("--no-render", dest="render", action="store_false")
    return parser


def backfill_model(
    *,
    model: str,
    results_tsv: Path,
    results_dir: Path,
    tasks_tsv: Path,
    dry_run: bool = False,
    max_cells: int = 0,
    verbose_runner: bool = False,
    min_seeds: int = 0,
) -> dict[str, int]:
    from src.finetune.run import build_finetune_cfg
    from src.finetune.utils import collect_pretrained_checkpoints, parse_finetune_tasks
    from src.utils.run_helpers import resolve_seeds

    model = _normalize_name(model)
    if not tasks_tsv.is_file():
        print(f"{LOG_PREFIX} Missing tasks TSV for {model}: {tasks_tsv}")
        return {"missing_tasks_tsv": 1, "appended": 0}

    cfg = build_finetune_cfg(
        [
            "finetune.run_tasks_tsv",
            "True",
            "finetune.tasks_tsv",
            str(tasks_tsv),
            "save_results.output_dir",
            str(results_dir),
        ]
    )
    cfg.save_results.enabled = not dry_run

    tasks = [task for task in parse_finetune_tasks(str(tasks_tsv)) if _normalize_name(task.get("model")) == model]
    if not tasks:
        print(f"{LOG_PREFIX} No tasks for model={model} in {tasks_tsv}")
        return {"tasks": 0, "appended": 0}

    checkpoint_root = getattr(getattr(cfg, "pretrain", None), "checkpoint_dir", "outputs/pretrained_models")
    log_root = getattr(getattr(cfg, "pretrain", None), "log_dir", None) or None
    checkpoints = collect_pretrained_checkpoints(checkpoint_root, log_root=log_root)
    if not checkpoints:
        print(f"{LOG_PREFIX} No pretrained checkpoints found under {checkpoint_root}")
        return {"tasks": len(tasks), "appended": 0, "missing_pretrain_checkpoints": len(tasks)}

    requested_runs = int(getattr(getattr(cfg, "finetune", None), "num_runs", 0) or 0)
    seeds = resolve_seeds(cfg, requested_count=requested_runs)
    pretrain_seed = int(resolve_seeds(cfg, requested_count=1)[0])
    required_seeds = min_seeds if min_seeds > 0 else len(seeds)

    present = _valid_result_cells(_read_result_rows(results_tsv), models={model})
    expected = _expected_cells_for_model(model)

    appended = 0
    skipped_present = 0
    no_artifact = 0
    failures = 0
    for task in tasks:
        key = _cell_key_from_task(model, task)
        if key is None or key not in expected:
            continue
        if key in present:
            skipped_present += 1
            continue
        if max_cells and appended >= max_cells:
            break

        try:
            result = _backfill_task(
                cfg=cfg,
                task=task,
                checkpoints=checkpoints,
                seeds=seeds,
                pretrain_seed=pretrain_seed,
                results_tsv=results_tsv,
                dry_run=dry_run,
                verbose_runner=verbose_runner,
                required_seeds=required_seeds,
            )
        except Exception as exc:
            failures += 1
            print(f"{LOG_PREFIX} Failed {model}/{task.get('dataset')}/{task.get('pretrain_method')}/{task.get('finetune_method')}: {exc}")
            continue

        if not result:
            no_artifact += 1
            continue

        appended += 1
        present.add(key)

    print(
        f"{LOG_PREFIX} model={model} tasks={len(tasks)} already_present={skipped_present} "
        f"appended={appended} no_artifact={no_artifact} failures={failures}"
    )
    return {
        "tasks": len(tasks),
        "already_present": skipped_present,
        "appended": appended,
        "no_artifact": no_artifact,
        "failures": failures,
    }


def _backfill_task(
    *,
    cfg,
    task: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    pretrain_seed: int,
    results_tsv: Path,
    dry_run: bool,
    verbose_runner: bool,
    required_seeds: int,
) -> bool:
    from src.finetune.finetuner import FinetuneRunner
    from src.finetune.utils import (
        _build_task_cfg,
        _checkpoint_label,
        _select_checkpoint_for_pretrain_seed,
        _select_checkpoints_for_task,
    )
    from src.utils.run_helpers import (
        aggregate_run_metrics,
        checkpoint_path_for_runner,
        collect_run_metrics,
    )
    from src.utils.save_results import append_workflow_result

    selected_checkpoints = _select_checkpoints_for_task(list(checkpoints), dict(task))
    if not selected_checkpoints:
        requested = str(task.get("pretrained_run_name") or "").strip()
        label = f"run '{requested}'" if requested else f"task {task.get('dataset')}"
        print(f"{LOG_PREFIX} No pretrained checkpoint matched {label}")
        return False

    requested_run_name = str(task.get("pretrained_run_name") or "").strip()
    if requested_run_name:
        if len(selected_checkpoints) != 1:
            names = [_checkpoint_label(ckpt) for ckpt in selected_checkpoints]
            raise ValueError(f"multiple checkpoints matched explicit run {requested_run_name}: {names}")
        ckpt = dict(selected_checkpoints[0])
    else:
        ckpt = dict(_select_checkpoint_for_pretrain_seed(list(selected_checkpoints), pretrain_seed, task=dict(task)))

    base_cfg = _build_task_cfg(cfg, dict(task), ckpt)
    run_metrics: list[dict[str, float]] = []
    completed_seeds: list[int] = []
    source_paths: list[str] = []
    checkpoint_paths: list[str] = []
    oom_paths: list[str] = []

    for seed in seeds:
        run_cfg = base_cfg.clone()
        run_cfg.seed = int(seed)
        stdout_context = contextlib.nullcontext() if verbose_runner else contextlib.redirect_stdout(io.StringIO())
        with stdout_context:
            runner = FinetuneRunner(
                cfg=run_cfg,
                pretrained_checkpoint=ckpt["path"],
                pretrained_run_name=ckpt.get("run_name"),
            )
        checkpoint_path = checkpoint_path_for_runner(runner)
        metrics = collect_run_metrics(runner, log_prefix=LOG_PREFIX)
        source_path = checkpoint_path if metrics and os.path.isfile(checkpoint_path) else ""
        if not metrics:
            metrics, source_path = _load_log_metrics(runner)
        if not metrics:
            status, status_path = _load_log_status(runner)
            if status == OOM_STATUS and status_path:
                oom_paths.append(status_path)
        if hasattr(runner, "_loaded_checkpoint"):
            delattr(runner, "_loaded_checkpoint")

        if not metrics:
            continue
        run_metrics.append(metrics)
        completed_seeds.append(int(seed))
        if checkpoint_path:
            checkpoint_paths.append(checkpoint_path)
        if source_path:
            source_paths.append(source_path)

    if len(run_metrics) < required_seeds:
        label = (
            f"{base_cfg.model.name}/{base_cfg.pretrain.method}->{base_cfg.finetune.dataset.name}/"
            f"{base_cfg.finetune.method}/split={list(_split_key(base_cfg.finetune.dataset.fixed_split))}"
        )
        if oom_paths:
            started_at, ended_at = _artifact_time_window(oom_paths)
            if dry_run:
                print(f"{LOG_PREFIX} DRY-RUN would append OOM {label} sources={len(oom_paths)}")
                return True

            _append_status_result(
                results_tsv=results_tsv,
                cfg=base_cfg,
                status=OOM_STATUS,
                started_at=started_at,
                ended_at=ended_at,
                seeds=seeds,
            )
            print(f"{LOG_PREFIX} appended OOM {label} sources={len(oom_paths)}")
            return True

        print(
            f"{LOG_PREFIX} Skipping {task.get('model')}/{task.get('dataset')}/"
            f"{task.get('pretrain_method')}/{task.get('finetune_method')}: "
            f"only {len(run_metrics)}/{required_seeds} seeds have complete artifacts "
            f"(completed={completed_seeds})"
        )
        return False

    summary = aggregate_run_metrics(run_metrics)
    metric_name = _metric_for_task(task)
    metric_stats = summary["metric_stats"]
    if not _valid_metric_values(metric_stats.get(metric_name, {}).get("mean"), metric_stats.get(metric_name, {}).get("std")):
        return False

    started_at, ended_at = _artifact_time_window(source_paths or checkpoint_paths)
    label = (
        f"{base_cfg.model.name}/{base_cfg.pretrain.method}->{base_cfg.finetune.dataset.name}/"
        f"{base_cfg.finetune.method}/split={list(_split_key(base_cfg.finetune.dataset.fixed_split))}"
    )
    if dry_run:
        print(f"{LOG_PREFIX} DRY-RUN would append {label} seeds={completed_seeds}")
        return True

    append_workflow_result(
        cfg=base_cfg,
        workflow="finetune",
        started_at=started_at,
        ended_at=ended_at,
        checkpoint_save_paths=checkpoint_paths,
        seeds=completed_seeds,
        best_epochs=summary["epoch_values"].get("best_epoch"),
        metric_summary=metric_stats,
    )
    print(f"{LOG_PREFIX} appended {label} seeds={completed_seeds}")
    return True


def render_tables(
    *,
    models: Sequence[str],
    results_tsv: Path,
    results_dir: Path,
    train_results_tsv: Path = train_tables.RESULTS_TSV,
    dry_run: bool = False,
) -> int:
    rows = _read_result_rows(results_tsv)
    latest = _latest_rows(rows)
    scratch = train_tables._latest_rows(train_tables._read_result_rows(train_results_tsv))
    rendered = 0
    normalized_models = [_normalize_name(model) for model in models]
    for model in normalized_models:
        for shot in ("f5", "f100"):
            text = _render_table(model=model, shot=shot, latest=latest)
            out_path = results_dir / f"finetune_{model}_{shot}_table.tex"
            rendered += _write_table(out_path, text, dry_run=dry_run)

    for shot in ("f5", "f100"):
        text = _render_backbone_summary_table(models=normalized_models, shot=shot, latest=latest)
        out_path = results_dir / f"finetune_backbone_{shot}_table.tex"
        rendered += _write_table(out_path, text, dry_run=dry_run)

    for shot in ("f5", "f100"):
        text = _render_method_summary_table(models=normalized_models, shot=shot, latest=latest)
        out_path = results_dir / f"finetune_{shot}_table.tex"
        rendered += _write_table(out_path, text, dry_run=dry_run)

    # Per-finetune-method regrouping of the per-backbone tables.
    rendered += _render_finetune_method_tables(
        models=normalized_models,
        latest=latest,
        results_dir=results_dir,
        dry_run=dry_run,
    )

    # GapTune paper tables.
    for shot in ("f5", "f100"):
        text = _render_cross_dataset_table(shot=shot, latest=latest, scratch=scratch)
        rendered += _write_table(results_dir / f"finetune_cross_{shot}_table.tex", text, dry_run=dry_run)
        text = _render_same_dataset_table(models=normalized_models, shot=shot, latest=latest, scratch=scratch)
        rendered += _write_table(results_dir / f"finetune_same_{shot}_table.tex", text, dry_run=dry_run)
    for table in ABLATION_TABLES:
        text = _render_ablation_table(table=table, latest=latest)
        rendered += _write_table(results_dir / f"finetune_ablation_{table.name}_table.tex", text, dry_run=dry_run)

    return rendered


def _write_table(out_path: Path, text: str, *, dry_run: bool) -> int:
    if dry_run:
        print(f"{LOG_PREFIX} DRY-RUN would write {out_path}")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return 1


def _read_result_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            {key: value or "" for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        ]
    return rows


def _is_renderable_metric_row(row: Mapping[str, str], metric: str) -> bool:
    """Return whether a latest row may contribute a rendered metric cell."""
    if _is_oom_row(row):
        return False
    status = _row_status(row).strip().lower()
    if status and status not in {"complete", "completed", "ok", "success"}:
        return False
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if not _valid_metric_values(mean, std):
        return False
    mean_f = float(mean)
    std_f = float(std)
    if std_f < 0.0:
        return False
    if metric in {"test_acc", "test_auc"}:
        return 0.0 <= mean_f <= 1.0
    if metric == "test_mae":
        return mean_f >= 0.0
    return True


def _latest_rows(rows: Sequence[Mapping[str, str]]) -> dict[CellKey, Mapping[str, str]]:
    latest: dict[CellKey, Mapping[str, str]] = {}
    for row in rows:
        key = _cell_key_from_result_row(row)
        if key is not None:
            latest[key] = row
    return latest


def _valid_result_cells(rows: Sequence[Mapping[str, str]], *, models: set[str]) -> set[CellKey]:
    valid: set[CellKey] = set()
    for row in rows:
        key = _cell_key_from_result_row(row)
        if key is None or key.model not in models:
            continue
        if _is_oom_row(row):
            valid.add(key)
            continue
        metric = _metric_for_dataset(key.dataset)
        if _valid_metric_values(row.get(f"{metric}_mean"), row.get(f"{metric}_std")):
            valid.add(key)
    return valid


def _row_status(row: Mapping[str, str]) -> str:
    for column in STATUS_COLUMNS:
        status = str(row.get(column) or "").strip()
        if status:
            return status
    return ""


def _is_oom_status(status: str) -> bool:
    normalized = status.strip().lower().replace("-", " ").replace("_", " ")
    if normalized in OOM_STATUS_VALUES:
        return True
    compact = normalized.replace(" ", "")
    return compact in OOM_STATUS_VALUES or "outofmemory" in compact


def _is_oom_row(row: Mapping[str, str]) -> bool:
    return _is_oom_status(_row_status(row))


def _expected_cells_for_model(model: str) -> set[CellKey]:
    # Same-dataset grids (pretrain dataset == target dataset) plus the
    # cross-dataset blocks whose source checkpoint uses this backbone.
    keys: set[CellKey] = set()
    for dataset in DATASETS:
        for split in (dataset.f5_split, dataset.f100_split):
            for pretrain in PRETRAIN_METHODS:
                keys.add(CellKey(model, dataset.name, split, dataset.name, pretrain.method, "supervised", None))
                for finetune in PROMPT_METHODS:
                    keys.add(CellKey(model, dataset.name, split, dataset.name, pretrain.method, finetune.method, finetune.plus))
            for source in CROSS_DATASET_SOURCES:
                if source.model != model:
                    continue
                for finetune in PROMPT_METHODS:
                    keys.add(CellKey(model, dataset.name, split, source.dataset, source.pretrain_method, finetune.method, finetune.plus))
    return keys


def _cell_key_from_task(model: str, task: Mapping[str, Any]) -> CellKey | None:
    dataset = _normalize_name(task.get("dataset"))
    pretrain_dataset = _normalize_name(task.get("pretrain_dataset"))
    pretrain_method = _normalize_name(task.get("pretrain_method"))
    finetune_method = _normalize_method(task.get("finetune_method") or "supervised")
    split = _split_key(task.get("fixed_split"))
    if not dataset or not pretrain_dataset or not pretrain_method or not finetune_method or not split:
        return None
    return CellKey(
        _normalize_name(model),
        dataset,
        split,
        pretrain_dataset,
        pretrain_method,
        finetune_method,
        _plus_from_task(task, finetune_method),
    )


def _cell_key_from_result_row(row: Mapping[str, str]) -> CellKey | None:
    model = _normalize_name(row.get("model.name"))
    dataset = _normalize_name(row.get("finetune.dataset.name"))
    pretrain_dataset = _normalize_name(row.get("pretrain.dataset.name"))
    pretrain_method = _normalize_name(row.get("pretrain.method"))
    finetune_method = _normalize_method(row.get("finetune.method") or "supervised")
    if finetune_method == "supervised" and _parse_bool(row.get("finetune.supervised.freeze_encoder")):
        finetune_method = HEAD_ONLY_METHOD
    split = _split_key(row.get("finetune.dataset.fixed_split"))
    if not model or not dataset or not pretrain_dataset or not pretrain_method or not finetune_method or not split:
        return None
    plus = _plus_from_result_row(row, finetune_method)
    variant = ""
    if finetune_method == "gaptune":
        settings = {
            column[len(GAPTUNE_COLUMN_PREFIX):]: value
            for column, value in row.items()
            if column and column.startswith(GAPTUNE_COLUMN_PREFIX)
        }
        variant = _gaptune_variant(settings, plus=bool(plus))
    return CellKey(
        model,
        dataset,
        split,
        pretrain_dataset,
        pretrain_method,
        finetune_method,
        plus,
        variant,
    )


def _gaptune_variant(settings: Mapping[str, Any], *, plus: bool) -> str:
    """Canonical ``key=value`` list of a GapTune run's non-default settings.

    ``settings`` maps ``finetune.gaptune.*`` suffixes to TSV text or Python
    values; a blank value is a row written before that column existed, i.e.
    the default.  ``plus`` is its own key component, and the proxy settings
    only matter for source-free runs (as in ``GapTune.variant_tag``).
    """
    parts = []
    for key in sorted(settings):
        if key == "plus" or (plus and key.startswith("proxy.")):
            continue
        default = _GAPTUNE_DEFAULTS
        for part in key.split("."):
            default = getattr(default, part, None)
        value = _parse_setting(settings[key], default)
        if value != default:
            parts.append(f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}")
    return ",".join(parts)


def _parse_setting(value: Any, default: Any) -> Any:
    """Parse one TSV/config value with the type of its default (blank -> default)."""
    text = "" if value is None else str(value).strip()
    if not text:
        return default
    if isinstance(default, bool):
        parsed = _parse_bool(text)
        return text if parsed is None else parsed
    if default is None or isinstance(default, (int, float)):
        try:
            number = float(text)
        except ValueError:
            return None if text.lower() in {"none", "null"} else text
        return int(number) if isinstance(default, int) and number.is_integer() else number
    return text.lower()


def _plus_from_task(task: Mapping[str, Any], finetune_method: str) -> bool | None:
    if finetune_method == "edgeprompt":
        return _coerce_bool(task.get("edgeprompt_plus"), default=True)
    if finetune_method == "gpf":
        return _coerce_bool(task.get("gpf_plus"), default=False)
    if finetune_method == "graphprompt":
        return _coerce_bool(task.get("graphprompt_plus"), default=False)
    if finetune_method == "gaptune":
        return _coerce_bool(task.get("gaptune_plus"), default=True)
    return None


def _plus_from_result_row(row: Mapping[str, str], finetune_method: str) -> bool | None:
    if finetune_method not in PLUS_COLUMNS:
        return None
    values = [row.get(PLUS_COLUMNS[finetune_method], "")]
    if PLUS_COLUMNS[finetune_method] != LEGACY_PLUS_COLUMN:
        values.append(row.get(LEGACY_PLUS_COLUMN, ""))
    default = finetune_method in ("edgeprompt", "gaptune")
    for value in values:
        parsed = _parse_bool(value)
        if parsed is not None:
            return parsed
    return default


def _render_table(model: str, shot: str, latest: Mapping[CellKey, Mapping[str, str]]) -> str:
    model_label = MODEL_LABELS.get(model, model.upper())
    shot_label = "5" if shot == "f5" else "100"
    split_comment = (
        "[5, 0.0, 1.0] for NC/GC/GR datasets; LP datasets use [0.05, 0.1, 0.1]."
        if shot == "f5"
        else "[100, 0.0, 1.0] for NC/GC/GR datasets; LP datasets use [0.1, 0.05, 0.1]."
    )
    lp_split_text = "10$\\%$-5$\\%$-10$\\%$" if shot == "f100" else "5$\\%$-10$\\%$-10$\\%$"
    lines = [
        "% Generated from outputs/results/finetune.tsv.",
        f"% Selection: model.name={model}; latest matching row by file order per logical table cell.",
        f"% Splits: {split_comment}",
        "% Metrics: test_acc for NC/MNIST GC, test_auc for LP/Toxcast GC, test_mae for QM7b GR.",
        "% Missing cells are shown as --; explicit OOM status rows are shown as OOM.",
        "\\begin{table*}[t]",
        "\\caption{",
        f"Performance of existing approaches using {model_label} as the backbone model.",
        f"For node classification (NC) and MNIST graph classification (GC), we report \\emph{{{shot_label}-shot}} accuracy scores.",
        f"For Toxcast multilabel graph classification, we report macro ROC-AUC using \\emph{{{shot_label}}} labeled training graphs.",
        f"For graph regression (GR), we report MAE using \\emph{{{shot_label}}} labeled training graphs.",
        f"For the link prediction (LP) task, we present AUC scores following the split \\emph{{{lp_split_text}}}.",
    ]
    lines.extend(
        [
            "}",
            f"\\label{{table:results_{model}_{shot}}}",
            "% \\vskip 0.15in",
            "% \\vspace{-1.mm}",
            "\\begin{center}",
            "\\begin{small}",
            "% \\begin{sc}",
            "\\resizebox{1.\\linewidth}{!}{",
            "\\begin{tabular}{l|ccc|ccc|ccc}",
            "\\toprule",
            "\\multirow{5}{*}{\\textbf{Models}} & \\multicolumn{9}{c}{\\textbf{Datasets}} \\\\",
            "& \\multicolumn{3}{c|}{\\textbf{Homophily}} & \\multicolumn{3}{c}{\\textbf{Heterophily}} & \\multicolumn{3}{c}{\\textbf{Graph}} \\\\",
            "& \\textbf{Social} & \\textbf{Academic} & \\textbf{Academic}",
            "& \\textbf{Transport} & \\textbf{Web} & \\textbf{Web}",
            "& \\textbf{Molecular} & \\textbf{Chemistry} & \\textbf{Vision} \\\\",
            "& \\textbf{Photo} & \\textbf{Ogbn-arxiv} & \\textbf{DBLP}",
            "& \\textbf{Airports} & \\textbf{Chameleon} & \\textbf{Cornell}",
            "& \\textbf{QM7b} & \\textbf{Toxcast} & \\textbf{MNIST} \\\\",
            "& NC & NC & LP",
            "& NC & NC & LP",
            "& GR & GC & GC \\\\",
            "\\midrule",
            f"& \\multicolumn{{9}}{{c}}{{Pre-train + Supervised Fine-tune \\emph{{{shot_label}-shot}}}} \\\\",
            "\\midrule",
        ]
    )

    for pretrain in PRETRAIN_METHODS:
        lines.extend(
            _render_metric_row(
                model=model,
                shot=shot,
                pretrain=pretrain.method,
                label=pretrain.label,
                finetune=FinetuneSpec("supervised", "supervised"),
                latest=latest,
            )
        )

    for pretrain in PRETRAIN_METHODS:
        lines.extend(
            [
                "\\midrule",
                f"& \\multicolumn{{9}}{{c}}{{Pre-train ({pretrain.label}) + Prompting \\emph{{{shot_label}-shot}}}} \\\\",
                "\\midrule",
            ]
        )
        for finetune in PROMPT_METHODS:
            lines.extend(
                _render_metric_row(
                    model=model,
                    shot=shot,
                    pretrain=pretrain.method,
                    label=finetune.label,
                    finetune=finetune,
                    latest=latest,
                )
            )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "% \\end{sc}",
            "\\end{small}",
            "\\end{center}",
            "% \\vskip -0.1in",
            "\\vspace{-2mm}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_metric_row(
    *,
    model: str,
    shot: str,
    pretrain: str,
    label: str,
    finetune: FinetuneSpec,
    latest: Mapping[CellKey, Mapping[str, str]],
) -> list[str]:
    values = []
    for dataset in DATASETS:
        split = dataset.f5_split if shot == "f5" else dataset.f100_split
        key = CellKey(model, dataset.name, split, dataset.name, pretrain, finetune.method, finetune.plus)
        values.append(_format_cell(latest.get(key), dataset.metric))
    return [
        label,
        f"    & {' & '.join(values[:3])}",
        f"    & {' & '.join(values[3:6])}",
        f"    & {' & '.join(values[6:])} \\\\",
    ]


def _format_cell(row: Mapping[str, str] | None, metric: str) -> str:
    if row is None:
        return "--"
    if _is_oom_row(row):
        return OOM_STATUS
    if not _is_renderable_metric_row(row, metric):
        return "--"
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    mean_f = float(mean)
    std_f = float(std)
    if metric == "test_mae":
        return f"{mean_f:.2f}$_{{\\pm{std_f:.2f}}}$"
    return f"{100.0 * mean_f:.2f}$_{{\\pm{100.0 * std_f:.2f}}}$"


def _render_backbone_summary_table(
    *,
    models: Sequence[str],
    shot: str,
    latest: Mapping[CellKey, Mapping[str, str]],
) -> str:
    shot_label = "5" if shot == "f5" else "100"
    split_comment = (
        "[5, 0.0, 1.0] for NC/GC/GR datasets; LP datasets use [0.05, 0.1, 0.1]."
        if shot == "f5"
        else "[100, 0.0, 1.0] for NC/GC/GR datasets; LP datasets use [0.1, 0.05, 0.1]."
    )
    lp_split_text = "10$\\%$-5$\\%$-10$\\%$" if shot == "f100" else "5$\\%$-10$\\%$-10$\\%$"
    model_labels = ", ".join(MODEL_LABELS.get(model, model.upper()) for model in models)
    lines = [
        "% Generated from outputs/results/finetune.tsv.",
        f"% Selection: best backbone per cell across models={list(models)}; latest matching row per (model, cell).",
        f"% Splits: {split_comment}",
        "% Metrics: test_acc for NC/MNIST GC, test_auc for LP/Toxcast GC, test_mae for QM7b GR (lower is better).",
        "% Missing cells are shown as --; explicit OOM status rows are shown as OOM.",
        "\\begin{table*}[t]",
        "\\caption{",
        f"Summary of best backbone performance among \\{{{model_labels}\\}} for each pre-train / prompt combination.",
        f"For node classification (NC) and MNIST graph classification (GC), we report \\emph{{{shot_label}-shot}} accuracy scores.",
        f"For Toxcast multilabel graph classification, we report macro ROC-AUC using \\emph{{{shot_label}}} labeled training graphs.",
        f"For graph regression (GR), we report MAE using \\emph{{{shot_label}}} labeled training graphs.",
        f"For the link prediction (LP) task, we present AUC scores following the split \\emph{{{lp_split_text}}}.",
        "Each cell shows the winning backbone's $\\text{mean}_{\\pm\\text{std}}^{\\text{backbone}}$.",
        "}",
        f"\\label{{table:results_backbone_summary_{shot}}}",
        "% \\vskip 0.15in",
        "% \\vspace{-1.mm}",
        "\\begin{center}",
        "\\begin{small}",
        "% \\begin{sc}",
        "\\resizebox{1.\\linewidth}{!}{",
        "\\begin{tabular}{l|ccc|ccc|ccc}",
        "\\toprule",
        "\\multirow{5}{*}{\\textbf{Models}} & \\multicolumn{9}{c}{\\textbf{Datasets}} \\\\",
        "& \\multicolumn{3}{c|}{\\textbf{Homophily}} & \\multicolumn{3}{c}{\\textbf{Heterophily}} & \\multicolumn{3}{c}{\\textbf{Graph}} \\\\",
        "& \\textbf{Social} & \\textbf{Academic} & \\textbf{Academic}",
        "& \\textbf{Transport} & \\textbf{Web} & \\textbf{Web}",
        "& \\textbf{Molecular} & \\textbf{Chemistry} & \\textbf{Vision} \\\\",
        "& \\textbf{Photo} & \\textbf{Ogbn-arxiv} & \\textbf{DBLP}",
        "& \\textbf{Airports} & \\textbf{Chameleon} & \\textbf{Cornell}",
        "& \\textbf{QM7b} & \\textbf{Toxcast} & \\textbf{MNIST} \\\\",
        "& NC & NC & LP",
        "& NC & NC & LP",
        "& GR & GC & GC \\\\",
        "\\midrule",
        f"& \\multicolumn{{9}}{{c}}{{Pre-train + Supervised Fine-tune \\emph{{{shot_label}-shot}}}} \\\\",
        "\\midrule",
    ]

    for pretrain in PRETRAIN_METHODS:
        lines.extend(
            _render_summary_row(
                models=models,
                shot=shot,
                pretrain=pretrain.method,
                label=pretrain.label,
                finetune=FinetuneSpec("supervised", "supervised"),
                latest=latest,
            )
        )

    for pretrain in PRETRAIN_METHODS:
        lines.extend(
            [
                "\\midrule",
                f"& \\multicolumn{{9}}{{c}}{{Pre-train ({pretrain.label}) + Prompting \\emph{{{shot_label}-shot}}}} \\\\",
                "\\midrule",
            ]
        )
        for finetune in PROMPT_METHODS:
            lines.extend(
                _render_summary_row(
                    models=models,
                    shot=shot,
                    pretrain=pretrain.method,
                    label=finetune.label,
                    finetune=finetune,
                    latest=latest,
                )
            )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "% \\end{sc}",
            "\\end{small}",
            "\\end{center}",
            "% \\vskip -0.1in",
            "\\vspace{-2mm}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_summary_row(
    *,
    models: Sequence[str],
    shot: str,
    pretrain: str,
    label: str,
    finetune: FinetuneSpec,
    latest: Mapping[CellKey, Mapping[str, str]],
) -> list[str]:
    values = []
    for dataset in DATASETS:
        split = dataset.f5_split if shot == "f5" else dataset.f100_split
        values.append(
            _format_best_cell(
                models=models,
                dataset=dataset,
                split=split,
                pretrain=pretrain,
                finetune=finetune,
                latest=latest,
            )
        )
    return [
        label,
        f"    & {' & '.join(values[:3])}",
        f"    & {' & '.join(values[3:6])}",
        f"    & {' & '.join(values[6:])} \\\\",
    ]


def _best_for_cell(
    *,
    models: Sequence[str],
    pretrain_methods: Sequence[str],
    dataset: DatasetSpec,
    split: tuple[float, ...],
    finetune: FinetuneSpec,
    latest: Mapping[CellKey, Mapping[str, str]],
) -> tuple[float, float, str] | None:
    lower_is_better = dataset.metric == "test_mae"
    best: tuple[float, float, str] | None = None
    for model in models:
        for pretrain in pretrain_methods:
            key = CellKey(model, dataset.name, split, dataset.name, pretrain, finetune.method, finetune.plus)
            row = latest.get(key)
            if row is None:
                continue
            if not _is_renderable_metric_row(row, dataset.metric):
                continue
            mean = row.get(f"{dataset.metric}_mean")
            std = row.get(f"{dataset.metric}_std")
            mean_f = float(mean)
            std_f = float(std)
            if best is None or (mean_f < best[0] if lower_is_better else mean_f > best[0]):
                best = (mean_f, std_f, model)
    return best


def _has_oom_for_cell(
    *,
    models: Sequence[str],
    pretrain_methods: Sequence[str],
    dataset: DatasetSpec,
    split: tuple[float, ...],
    finetune: FinetuneSpec,
    latest: Mapping[CellKey, Mapping[str, str]],
) -> bool:
    for model in models:
        for pretrain in pretrain_methods:
            key = CellKey(model, dataset.name, split, dataset.name, pretrain, finetune.method, finetune.plus)
            row = latest.get(key)
            if row is not None and _is_oom_row(row):
                return True
    return False


def _format_metric_value(mean_f: float, std_f: float, metric: str, *, superscript: str = "") -> str:
    sup = f"^{{{superscript}}}" if superscript else ""
    if metric == "test_mae":
        return f"{mean_f:.2f}$_{{\\pm{std_f:.2f}}}{sup}$"
    return f"{100.0 * mean_f:.2f}$_{{\\pm{100.0 * std_f:.2f}}}{sup}$"


def _format_best_cell(
    *,
    models: Sequence[str],
    dataset: DatasetSpec,
    split: tuple[float, ...],
    pretrain: str,
    finetune: FinetuneSpec,
    latest: Mapping[CellKey, Mapping[str, str]],
) -> str:
    best = _best_for_cell(
        models=models,
        pretrain_methods=[pretrain],
        dataset=dataset,
        split=split,
        finetune=finetune,
        latest=latest,
    )
    if best is None:
        if _has_oom_for_cell(
            models=models,
            pretrain_methods=[pretrain],
            dataset=dataset,
            split=split,
            finetune=finetune,
            latest=latest,
        ):
            return OOM_STATUS
        return "--"
    mean_f, std_f, winner = best
    label_tex = MODEL_LABELS.get(winner, winner.upper())
    return _format_metric_value(mean_f, std_f, dataset.metric, superscript=f"\\text{{{label_tex}}}")


def _format_best_cell_plain(
    *,
    models: Sequence[str],
    pretrain_methods: Sequence[str],
    dataset: DatasetSpec,
    split: tuple[float, ...],
    finetune: FinetuneSpec,
    latest: Mapping[CellKey, Mapping[str, str]],
) -> str:
    best = _best_for_cell(
        models=models,
        pretrain_methods=pretrain_methods,
        dataset=dataset,
        split=split,
        finetune=finetune,
        latest=latest,
    )
    if best is None:
        if _has_oom_for_cell(
            models=models,
            pretrain_methods=pretrain_methods,
            dataset=dataset,
            split=split,
            finetune=finetune,
            latest=latest,
        ):
            return OOM_STATUS
        return "--"
    mean_f, std_f, _ = best
    return _format_metric_value(mean_f, std_f, dataset.metric)


def _render_method_summary_table(
    *,
    models: Sequence[str],
    shot: str,
    latest: Mapping[CellKey, Mapping[str, str]],
) -> str:
    shot_label = "5" if shot == "f5" else "100"
    split_comment = (
        "[5, 0.0, 1.0] for NC/GC/GR datasets; LP datasets use [0.05, 0.1, 0.1]."
        if shot == "f5"
        else "[100, 0.0, 1.0] for NC/GC/GR datasets; LP datasets use [0.1, 0.05, 0.1]."
    )
    lp_split_text = "10$\\%$-5$\\%$-10$\\%$" if shot == "f100" else "5$\\%$-10$\\%$-10$\\%$"
    model_labels = ", ".join(MODEL_LABELS.get(model, model.upper()) for model in models)
    pretrain_methods = [pretrain.method for pretrain in PRETRAIN_METHODS]
    lines = [
        "% Generated from outputs/results/finetune.tsv.",
        f"% Selection: supervised rows pick best across models={list(models)}; prompting rows pick best across models and pretrain methods={pretrain_methods}.",
        f"% Splits: {split_comment}",
        "% Metrics: test_acc for NC/MNIST GC, test_auc for LP/Toxcast GC, test_mae for QM7b GR (lower is better).",
        "% Missing cells are shown as --; explicit OOM status rows are shown as OOM.",
        "\\begin{table*}[t]",
        "\\caption{",
        f"Summary of best performance per pre-train / prompt method across backbones \\{{{model_labels}\\}}.",
        f"For node classification (NC) and MNIST graph classification (GC), we report \\emph{{{shot_label}-shot}} accuracy scores.",
        f"For Toxcast multilabel graph classification, we report macro ROC-AUC using \\emph{{{shot_label}}} labeled training graphs.",
        f"For graph regression (GR), we report MAE using \\emph{{{shot_label}}} labeled training graphs.",
        f"For the link prediction (LP) task, we present AUC scores following the split \\emph{{{lp_split_text}}}.",
        "}",
        f"\\label{{table:results_summary_{shot}}}",
        "% \\vskip 0.15in",
        "% \\vspace{-1.mm}",
        "\\begin{center}",
        "\\begin{small}",
        "% \\begin{sc}",
        "\\resizebox{1.\\linewidth}{!}{",
        "\\begin{tabular}{l|ccc|ccc|ccc}",
        "\\toprule",
        "\\multirow{5}{*}{\\textbf{Models}} & \\multicolumn{9}{c}{\\textbf{Datasets}} \\\\",
        "& \\multicolumn{3}{c|}{\\textbf{Homophily}} & \\multicolumn{3}{c}{\\textbf{Heterophily}} & \\multicolumn{3}{c}{\\textbf{Graph}} \\\\",
        "& \\textbf{Social} & \\textbf{Academic} & \\textbf{Academic}",
        "& \\textbf{Transport} & \\textbf{Web} & \\textbf{Web}",
        "& \\textbf{Molecular} & \\textbf{Chemistry} & \\textbf{Vision} \\\\",
        "& \\textbf{Photo} & \\textbf{Ogbn-arxiv} & \\textbf{DBLP}",
        "& \\textbf{Airports} & \\textbf{Chameleon} & \\textbf{Cornell}",
        "& \\textbf{QM7b} & \\textbf{Toxcast} & \\textbf{MNIST} \\\\",
        "& NC & NC & LP",
        "& NC & NC & LP",
        "& GR & GC & GC \\\\",
        "\\midrule",
        f"& \\multicolumn{{9}}{{c}}{{Pre-train + Fine-tune \\emph{{{shot_label}-shot}}}} \\\\",
        "\\midrule",
    ]

    for pretrain in PRETRAIN_METHODS:
        lines.extend(
            _render_method_summary_row(
                models=models,
                pretrain_methods=[pretrain.method],
                shot=shot,
                label=pretrain.label,
                finetune=FinetuneSpec("supervised", "supervised"),
                latest=latest,
            )
        )

    lines.extend(
        [
            "\\midrule",
            f"& \\multicolumn{{9}}{{c}}{{Pre-train + Prompting \\emph{{{shot_label}-shot}}}} \\\\",
            "\\midrule",
        ]
    )
    for finetune in PROMPT_METHODS:
        lines.extend(
            _render_method_summary_row(
                models=models,
                pretrain_methods=pretrain_methods,
                shot=shot,
                label=finetune.label,
                finetune=finetune,
                latest=latest,
            )
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "% \\end{sc}",
            "\\end{small}",
            "\\end{center}",
            "% \\vskip -0.1in",
            "\\vspace{-2mm}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_method_summary_row(
    *,
    models: Sequence[str],
    pretrain_methods: Sequence[str],
    shot: str,
    label: str,
    finetune: FinetuneSpec,
    latest: Mapping[CellKey, Mapping[str, str]],
) -> list[str]:
    values = []
    for dataset in DATASETS:
        split = dataset.f5_split if shot == "f5" else dataset.f100_split
        values.append(
            _format_best_cell_plain(
                models=models,
                pretrain_methods=pretrain_methods,
                dataset=dataset,
                split=split,
                finetune=finetune,
                latest=latest,
            )
        )
    return [
        label,
        f"    & {' & '.join(values[:3])}",
        f"    & {' & '.join(values[3:6])}",
        f"    & {' & '.join(values[6:])} \\\\",
    ]


_TABLE_COLUMN_HEADER_LINES: tuple[str, ...] = (
    "\\begin{tabular}{l|ccc|ccc|ccc}",
    "\\toprule",
    "\\multirow{5}{*}{\\textbf{Models}} & \\multicolumn{9}{c}{\\textbf{Datasets}} \\\\",
    "& \\multicolumn{3}{c|}{\\textbf{Homophily}} & \\multicolumn{3}{c}{\\textbf{Heterophily}} & \\multicolumn{3}{c}{\\textbf{Graph}} \\\\",
    "& \\textbf{Social} & \\textbf{Academic} & \\textbf{Academic}",
    "& \\textbf{Transport} & \\textbf{Web} & \\textbf{Web}",
    "& \\textbf{Molecular} & \\textbf{Chemistry} & \\textbf{Vision} \\\\",
    "& \\textbf{Photo} & \\textbf{Ogbn-arxiv} & \\textbf{DBLP}",
    "& \\textbf{Airports} & \\textbf{Chameleon} & \\textbf{Cornell}",
    "& \\textbf{QM7b} & \\textbf{Toxcast} & \\textbf{MNIST} \\\\",
    "& NC & NC & LP",
    "& NC & NC & LP",
    "& GR & GC & GC \\\\",
)

#: Paper Tables 1, 2, 14 and 15: the grid header with the paper's domain row
#: (DBLP is Academic and Cornell is Web there).
_PAPER_COLUMN_HEADER_LINES: tuple[str, ...] = (
    *_TABLE_COLUMN_HEADER_LINES[:4],
    "& \\textbf{E-commerce} & \\textbf{Academic} & \\textbf{Academic}",
    "& \\textbf{Transport} & \\textbf{Web} & \\textbf{Web}",
    "& \\textbf{Chemistry} & \\textbf{Chemistry} & \\textbf{Vision} \\\\",
    *_TABLE_COLUMN_HEADER_LINES[7:],
)

_TABLE_FOOTER_LINES: tuple[str, ...] = (
    "\\bottomrule",
    "\\end{tabular}",
    "}",
    "% \\end{sc}",
    "\\end{small}",
    "\\end{center}",
    "% \\vskip -0.1in",
    "\\vspace{-2mm}",
    "\\end{table*}",
    "",
)


def _finetune_method_slug(finetune: FinetuneSpec) -> str:
    return f"{finetune.method}_plus" if finetune.plus else finetune.method


def _render_finetune_method_tables(
    *,
    models: Sequence[str],
    latest: Mapping[CellKey, Mapping[str, str]],
    results_dir: Path,
    dry_run: bool,
) -> int:
    rendered = 0
    for finetune in (SUPERVISED_TABLE_SPEC, *PROMPT_METHODS):
        slug = _finetune_method_slug(finetune)
        for shot in ("f5", "f100"):
            text = _render_finetune_method_table(
                models=models, shot=shot, finetune=finetune, latest=latest
            )
            out_path = results_dir / f"finetune_{slug}_{shot}_table.tex"
            rendered += _write_table(out_path, text, dry_run=dry_run)
    return rendered


def _render_finetune_method_table(
    *,
    models: Sequence[str],
    shot: str,
    finetune: FinetuneSpec,
    latest: Mapping[CellKey, Mapping[str, str]],
) -> str:
    shot_label = "5" if shot == "f5" else "100"
    split_comment = (
        "[5, 0.0, 1.0] for NC/GC/GR datasets; LP datasets use [0.05, 0.1, 0.1]."
        if shot == "f5"
        else "[100, 0.0, 1.0] for NC/GC/GR datasets; LP datasets use [0.1, 0.05, 0.1]."
    )
    lp_split_text = "10$\\%$-5$\\%$-10$\\%$" if shot == "f100" else "5$\\%$-10$\\%$-10$\\%$"
    ordered_models = [model for model in MODEL_LABELS if model in set(models)]
    ordered_models += [model for model in models if model not in MODEL_LABELS]
    model_labels = ", ".join(MODEL_LABELS.get(model, model.upper()) for model in ordered_models)
    slug = _finetune_method_slug(finetune)
    section_suffix = (
        "Supervised Fine-tune"
        if finetune.method == "supervised"
        else f"{finetune.label} Prompting"
    )
    lines = [
        "% Generated from outputs/results/finetune.tsv.",
        f"% Selection: finetune method={slug}; latest matching row by file order per logical table cell.",
        f"% Splits: {split_comment}",
        "% Metrics: test_acc for NC/MNIST GC, test_auc for LP/Toxcast GC, test_mae for QM7b GR.",
        "% Missing cells are shown as --; explicit OOM status rows are shown as OOM.",
        "\\begin{table*}[t]",
        "\\caption{",
        f"Performance of {finetune.label} across backbone models \\{{{model_labels}\\}}.",
        f"For node classification (NC) and MNIST graph classification (GC), we report \\emph{{{shot_label}-shot}} accuracy scores.",
        f"For Toxcast multilabel graph classification, we report macro ROC-AUC using \\emph{{{shot_label}}} labeled training graphs.",
        f"For graph regression (GR), we report MAE using \\emph{{{shot_label}}} labeled training graphs.",
        f"For the link prediction (LP) task, we present AUC scores following the split \\emph{{{lp_split_text}}}.",
        "}",
        f"\\label{{table:results_method_{slug}_{shot}}}",
        "% \\vskip 0.15in",
        "% \\vspace{-1.mm}",
        "\\begin{center}",
        "\\begin{small}",
        "% \\begin{sc}",
        "\\resizebox{1.\\linewidth}{!}{",
        *_TABLE_COLUMN_HEADER_LINES,
    ]

    for pretrain in PRETRAIN_METHODS:
        lines.extend(
            [
                "\\midrule",
                f"& \\multicolumn{{9}}{{c}}{{Pre-train ({pretrain.label}) + {section_suffix} \\emph{{{shot_label}-shot}}}} \\\\",
                "\\midrule",
            ]
        )
        for model in ordered_models:
            lines.extend(
                _render_metric_row(
                    model=model,
                    shot=shot,
                    pretrain=pretrain.method,
                    label=MODEL_LABELS.get(model, model.upper()),
                    finetune=finetune,
                    latest=latest,
                )
            )

    lines.extend(_TABLE_FOOTER_LINES)
    return "\n".join(lines)


def _render_cross_dataset_table(
    *,
    shot: str,
    latest: Mapping[CellKey, Mapping[str, str]],
    scratch: Mapping[train_tables.CellKey, Mapping[str, str]],
) -> str:
    shot_label = "5" if shot == "f5" else "100"
    splits = [dataset.f5_split if shot == "f5" else dataset.f100_split for dataset in DATASETS]
    metrics = [dataset.metric for dataset in DATASETS]
    pretrain_labels = {pretrain.method: pretrain.label for pretrain in PRETRAIN_METHODS}
    body: list[str] = []
    for source in CROSS_DATASET_SOURCES:
        model_label = MODEL_LABELS[source.model]
        rows = [
            (
                f"{model_label} (scratch)",
                [
                    _cell_value(scratch.get(train_tables.CellKey(source.model, dataset.name, split)), dataset.metric)
                    for dataset, split in zip(DATASETS, splits)
                ],
            )
        ]
        for finetune in (*BASELINE_METHODS, *GAPTUNE_METHODS):
            cells = []
            for dataset, split in zip(DATASETS, splits):
                key = CellKey(source.model, dataset.name, split, source.dataset, source.pretrain_method, finetune.method, finetune.plus)
                cells.append(_cell_value(latest.get(key), dataset.metric))
            rows.append((finetune.label, cells))
        title = f"Source dataset: {source.label}, GNN: {model_label}, Pretraining: {pretrain_labels[source.pretrain_method]}"
        body.extend(_ranked_group_lines([(title, rows)], metrics))
    return _paper_table(
        selection="one block per source checkpoint (pretrain dataset/model/method); latest matching row per cell; scratch rows from train.tsv; ranked within each block.",
        caption=(
            f"Cross-dataset adaptation at \\emph{{{shot_label}-shot}}. Each block repeats the architecture-matched scratch control. "
            "Bold/underline mark the best/second test means within a block, including scratch."
        ),
        label=f"table:results_cross_{shot}",
        header=_PAPER_COLUMN_HEADER_LINES,
        body=body,
    )


def _render_same_dataset_table(
    *,
    models: Sequence[str],
    shot: str,
    latest: Mapping[CellKey, Mapping[str, str]],
    scratch: Mapping[train_tables.CellKey, Mapping[str, str]],
) -> str:
    shot_label = "5" if shot == "f5" else "100"
    splits = [dataset.f5_split if shot == "f5" else dataset.f100_split for dataset in DATASETS]
    metrics = [dataset.metric for dataset in DATASETS]
    pretrain_methods = [pretrain.method for pretrain in PRETRAIN_METHODS]
    scratch_rows = [
        (
            model.label,
            [
                _cell_value(scratch.get(train_tables.CellKey(model.name, dataset.name, split)), dataset.metric)
                for dataset, split in zip(DATASETS, splits)
            ],
        )
        for model in train_tables.MODELS
        if model.name in models
    ]
    prompt_rows = []
    for finetune in (*BASELINE_METHODS, *(spec for spec in GAPTUNE_METHODS if spec.plus)):
        cells = []
        for dataset, split in zip(DATASETS, splits):
            best = _best_for_cell(
                models=models, pretrain_methods=pretrain_methods, dataset=dataset, split=split, finetune=finetune, latest=latest
            )
            if best is not None:
                cells.append(best[:2])
            elif _has_oom_for_cell(
                models=models, pretrain_methods=pretrain_methods, dataset=dataset, split=split, finetune=finetune, latest=latest
            ):
                cells.append(OOM_STATUS)
            else:
                cells.append(None)
        prompt_rows.append((finetune.label, cells))
    sections = [
        ("Target-supervised controls: training from scratch (no pretraining)", scratch_rows),
        ("Same-dataset pretraining + prompting (best-observed summaries)", prompt_rows),
    ]
    return _paper_table(
        selection=f"scratch rows from train.tsv; prompting rows pick the best test mean across models={list(models)} and pretrain methods={pretrain_methods} (same-dataset cells only).",
        caption=(
            f"Same-dataset \\emph{{{shot_label}-shot}} results with scratch controls; prompting rows report the best observed "
            "test mean over the architecture--objective grid. Bold/underline indicate the best/second-best displayed means."
        ),
        label=f"table:results_same_{shot}",
        header=_PAPER_COLUMN_HEADER_LINES,
        body=_ranked_group_lines(sections, metrics),
    )


def _render_ablation_table(*, table: AblationTable, latest: Mapping[CellKey, Mapping[str, str]]) -> str:
    source = ABLATION_SOURCE
    pretrain_label = next(pretrain.label for pretrain in PRETRAIN_METHODS if pretrain.method == source.pretrain_method)
    columns = [(next(spec for spec in DATASETS if spec.name == name), shot) for name, shot in table.columns]
    metrics = [dataset.metric for dataset, _ in columns]
    rows = []
    for arm in table.arms:
        cells = []
        for dataset, shot in columns:
            split = dataset.f5_split if shot == "f5" else dataset.f100_split
            key = CellKey(source.model, dataset.name, split, source.dataset, source.pretrain_method, "gaptune", arm.plus, arm.variant)
            cells.append(_cell_value(latest.get(key), dataset.metric))
        rows.append((arm.label, cells))
    header = [
        f"\\begin{{tabular}}{{l|{'c' * len(columns)}}}",
        "\\toprule",
        "\\textbf{Variant} & " + " & ".join(f"\\textbf{{{dataset.label}}}" for dataset, _ in columns) + " \\\\",
        "& " + " & ".join(f"{dataset.task_label} / {'5' if shot == 'f5' else '100'}-shot" for dataset, shot in columns) + " \\\\",
    ]
    return _paper_table(
        selection=(
            f"model.name={source.model}, pretrain {source.dataset}/{source.pretrain_method}, finetune.method=gaptune; "
            "arms keyed by gaptune plus flag + non-default finetune.gaptune.* columns; latest matching row per cell."
        ),
        caption=(
            f"{table.caption} {source.label}/{MODEL_LABELS[source.model]}/{pretrain_label} checkpoint. "
            "Bold/underline mark the best/second distinct displayed means."
        ),
        label=f"table:ablation_{table.name}",
        header=header,
        body=_ranked_group_lines([("", rows)], metrics),
    )


def _cell_value(row: Mapping[str, str] | None, metric: str) -> tuple[float, float] | str | None:
    """``(mean, std)`` of a renderable row, ``OOM_STATUS``, or ``None`` when missing."""
    if row is None:
        return None
    if _is_oom_row(row):
        return OOM_STATUS
    if not _is_renderable_metric_row(row, metric):
        return None
    return float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])


def _shown_mean(mean: float, metric: str) -> str:
    return f"{mean if metric == 'test_mae' else 100.0 * mean:.2f}"


def _ranked_group_lines(
    sections: Sequence[tuple[str, Sequence[tuple[str, Sequence[tuple[float, float] | str | None]]]]],
    metrics: Sequence[str],
) -> list[str]:
    """Rows of one ranking group; bold/underline mark each column's best/second distinct displayed mean."""
    all_cells = [cells for _, rows in sections for _, cells in rows]
    tops = []
    for column, metric in enumerate(metrics):
        shown = {float(_shown_mean(cells[column][0], metric)) for cells in all_cells if isinstance(cells[column], tuple)}
        tops.append(sorted(shown, reverse=metric != "test_mae")[:2])
    lines = []
    for title, rows in sections:
        lines.append("\\midrule")
        if title:
            lines.extend([f"& \\multicolumn{{{len(metrics)}}}{{c}}{{{title}}} \\\\", "\\midrule"])
        for label, cells in rows:
            values = []
            for cell, metric, top in zip(cells, metrics, tops):
                if not isinstance(cell, tuple):
                    values.append(cell or "--")
                    continue
                mean_text = _shown_mean(cell[0], metric)
                std_text = _shown_mean(cell[1], metric)
                if float(mean_text) == top[0]:
                    mean_text = f"\\textbf{{{mean_text}}}"
                elif len(top) > 1 and float(mean_text) == top[1]:
                    mean_text = f"\\underline{{{mean_text}}}"
                values.append(f"{mean_text}$_{{\\pm{std_text}}}$")
            lines.extend([label, f"    & {' & '.join(values)} \\\\"])
    return lines


def _paper_table(*, selection: str, caption: str, label: str, header: Sequence[str], body: Sequence[str]) -> str:
    lines = [
        "% Generated from outputs/results/finetune.tsv (scratch controls from outputs/results/train.tsv).",
        f"% Selection: {selection}",
        "% Metrics: test_acc for NC/MNIST GC, test_auc for LP/Toxcast GC, test_mae for QM7b GR (lower is better).",
        "% Missing cells are shown as --; explicit OOM status rows are shown as OOM.",
        "\\begin{table*}[t]",
        "\\caption{",
        caption,
        "}",
        f"\\label{{{label}}}",
        "% \\vskip 0.15in",
        "% \\vspace{-1.mm}",
        "\\begin{center}",
        "\\begin{small}",
        "% \\begin{sc}",
        "\\resizebox{1.\\linewidth}{!}{",
        *header,
        *body,
        *_TABLE_FOOTER_LINES,
    ]
    return "\n".join(lines)


def _load_log_metrics(runner: FinetuneRunner) -> tuple[dict[str, float], str]:
    log_path = runner._log_path()
    if not os.path.isfile(log_path):
        return {}, ""
    try:
        with open(log_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"{LOG_PREFIX} Failed to load log metrics: {log_path} ({exc})")
        return {}, ""

    history = payload.get("history") or []
    if not isinstance(history, list) or not history:
        return {}, log_path
    best_epoch = (payload.get("best") or {}).get("epoch")
    if best_epoch is None:
        print(f"{LOG_PREFIX} Skipping log with no best.epoch (training likely unfinished): {log_path}")
        return {}, log_path
    chosen = None
    for entry in history:
        if isinstance(entry, dict) and int(entry.get("epoch", -1)) == int(best_epoch):
            chosen = entry
            break
    if not isinstance(chosen, dict):
        return {}, log_path

    metrics: dict[str, float] = {}
    if chosen.get("epoch") is not None:
        metrics["best_epoch"] = float(chosen["epoch"])
    if chosen.get("loss") is not None:
        metrics["train_loss"] = float(chosen["loss"])
    for key, value in (chosen.get("metrics") or {}).items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics, log_path


def _load_log_status(runner: FinetuneRunner) -> tuple[str, str]:
    log_path = runner._log_path()
    if not os.path.isfile(log_path):
        return "", ""
    try:
        with open(log_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"{LOG_PREFIX} Failed to load log status: {log_path} ({exc})")
        return "", ""
    if _payload_has_oom(payload):
        return OOM_STATUS, log_path
    return "", log_path


def _payload_has_oom(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_payload_has_oom(part) for part in value.values())
    if isinstance(value, list):
        return any(_payload_has_oom(part) for part in value)
    if isinstance(value, str):
        return _is_oom_status(value)
    return False


def _append_status_result(
    *,
    results_tsv: Path,
    cfg,
    status: str,
    started_at: datetime,
    ended_at: datetime,
    seeds: Sequence[int],
) -> None:
    if not bool(getattr(getattr(cfg, "save_results", None), "enabled", False)):
        return

    # Same advisory lock as append_workflow_result: this writer must not
    # interleave its read-modify-rewrite with an active finetune sweep.
    from src.utils.save_results import exclusive_table_lock

    with exclusive_table_lock(results_tsv):
        _append_status_result_locked(
            results_tsv=results_tsv,
            cfg=cfg,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            seeds=seeds,
        )


def _append_status_result_locked(
    *,
    results_tsv: Path,
    cfg,
    status: str,
    started_at: datetime,
    ended_at: datetime,
    seeds: Sequence[int],
) -> None:
    header, rows = _load_status_table(results_tsv)
    if not header:
        header = [
            "started_at",
            "ended_at",
            "duration_sec",
            "finetune.num_runs",
            "finetune.dataset.name",
            "finetune.dataset.task_level",
            "finetune.dataset.induced",
            "finetune.dataset.task_type",
            "finetune.dataset.fixed_split",
            "finetune.method",
            "pretrain.dataset.name",
            "pretrain.dataset.task_level",
            "pretrain.dataset.induced",
            "pretrain.method",
            "model.name",
            "finetune.edgeprompt.plus",
            "finetune.gpf.plus",
            "finetune.graphprompt.plus",
            "finetune.gaptune.plus",
            "result_status",
            "seeds",
            "best_epochs",
        ]
    elif "result_status" not in header:
        insert_at = header.index("seeds") if "seeds" in header else len(header)
        header.insert(insert_at, "result_status")

    for row in rows:
        row.setdefault("result_status", "")

    perf_columns = {column for column in header if column.endswith(("_mean", "_std"))}
    row = {column: ("-1" if column in perf_columns else "") for column in header}
    row.update(
        {
            "started_at": started_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": ended_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": f"{max(0.0, (ended_at - started_at).total_seconds()):.2f}",
            "seeds": json.dumps([int(seed) for seed in seeds]),
            "result_status": status,
        }
    )
    for column in header:
        if row.get(column) or column in perf_columns or column in {"started_at", "ended_at", "duration_sec", "seeds", "best_epochs"}:
            continue
        row[column] = _cfg_value_text(cfg, column)
    rows.append(row)
    _write_status_table(results_tsv, header, rows)


def _load_status_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), [{key: value or "" for key, value in row.items()} for row in reader]


def _write_status_table(path: Path, header: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    # Atomic replace, mirroring src.utils.save_results._write_table.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(header), delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in header})
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _cfg_value_text(cfg, dotted_key: str) -> str:
    current = cfg
    for part in str(dotted_key).split("."):
        if not hasattr(current, part):
            return ""
        current = getattr(current, part)
    if current is None:
        return ""
    if isinstance(current, bool):
        return "True" if current else "False"
    if isinstance(current, (list, tuple)):
        return json.dumps(list(current))
    return str(current)


def _artifact_time_window(paths: Sequence[str]) -> tuple[datetime, datetime]:
    timestamps = [os.path.getmtime(path) for path in paths if path and os.path.exists(path)]
    if not timestamps:
        now = datetime.now().astimezone()
        return now, now
    return (
        datetime.fromtimestamp(min(timestamps)).astimezone(),
        datetime.fromtimestamp(max(timestamps)).astimezone(),
    )


def _metric_for_task(task: Mapping[str, Any]) -> str:
    return _metric_for_dataset(_normalize_name(task.get("dataset")))


def _metric_for_dataset(dataset_name: str) -> str:
    for dataset in DATASETS:
        if dataset.name == _normalize_name(dataset_name):
            return dataset.metric
    return "test_acc"


def _valid_metric_values(mean: Any, std: Any) -> bool:
    try:
        mean_f = float(mean)
        std_f = float(std)
    except (TypeError, ValueError):
        return False
    return math.isfinite(mean_f) and math.isfinite(std_f) and mean_f != -1.0 and std_f != -1.0


def _split_key(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return ()
    try:
        return tuple(round(float(part), 10) for part in value)
    except TypeError:
        return ()


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _coerce_bool(value: Any, *, default: bool) -> bool:
    parsed = _parse_bool(value)
    return default if parsed is None else parsed


def _normalize_method(value: Any) -> str:
    return _normalize_name(value).replace("-", "_")


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _resolve_models(raw: str) -> list[str]:
    if str(raw).strip().lower() == "auto":
        return list(MODEL_LABELS)
    return [_normalize_name(part) for part in str(raw).split(",") if _normalize_name(part)]


if __name__ == "__main__":
    raise SystemExit(main())
