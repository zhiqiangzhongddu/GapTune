"""Backfill train result rows and render LaTeX summary tables.

The supervised-from-scratch train jobs run with ``skip_if_exists=True`` and
the default ``cfg.save_results.save_skipped=False``, so any run whose 5 seed
checkpoints already exist short-circuits without appending a row to
``outputs/results/train.tsv``.  This module rebuilds those missing rows from
saved train artifacts (checkpoints under ``outputs/trained_models/<dataset>/`` and
training-log JSON under ``logs/training_models/<dataset>/``) and then renders
the two paper tables from the TSV as the single source of truth.

Unlike ``finetune_tables``, the train tables are single files — one ``f5`` and
one ``f100`` — covering all backbones as rows; there is no per-model table.
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

from src.results.metric_policy import TABLE_TASKS, eval_metric

if False:  # pragma: no cover — type-only; torch deps are lazy-loaded inside backfill().
    from src.train.trainer import TrainRunner


RESULTS_DIR = Path("outputs/results")
RESULTS_TSV = RESULTS_DIR / "train.tsv"
TASKS_TSV = Path("slurm/train.tsv")
LOG_PREFIX = "[Results][Train]"


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
class ModelSpec:
    name: str
    label: str


@dataclass(frozen=True)
class CellKey:
    model: str
    dataset: str
    split: tuple[float, ...]


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

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("gcn", "GCN"),
    ModelSpec("gat", "GAT"),
    ModelSpec("gin", "GIN"),
    ModelSpec("h2gcn", "H2GCN"),
    ModelSpec("fagcn", "FAGCN"),
    ModelSpec("transformer", "Transformer"),
    ModelSpec("nodeformer", "NodeFormer"),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    results_tsv = Path(args.results_tsv)
    results_dir = Path(args.results_dir)
    tasks_tsv = Path(args.tasks_tsv)
    models = _resolve_models(args.models)

    if args.backfill:
        result = backfill(
            models=models,
            results_tsv=results_tsv,
            results_dir=results_dir,
            tasks_tsv=tasks_tsv,
            dry_run=bool(args.dry_run),
            max_cells=args.max_backfill_cells,
            verbose_runner=bool(args.verbose_runner),
            min_seeds=args.min_seeds,
        )
        print(f"{LOG_PREFIX} Backfill appended rows: {result.get('appended', 0)}")

    if args.render:
        rendered = render_tables(
            results_tsv=results_tsv,
            results_dir=results_dir,
            dry_run=bool(args.dry_run),
        )
        print(f"{LOG_PREFIX} Rendered tables: {rendered}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill train.tsv from saved artifacts and render train LaTeX tables."
    )
    parser.add_argument(
        "--models",
        default="auto",
        help="Comma-separated model names, or 'auto' for all models in MODELS.",
    )
    parser.add_argument("--results-tsv", default=str(RESULTS_TSV))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--tasks-tsv", default=str(TASKS_TSV))
    parser.add_argument("--max-backfill-cells", type=int, default=0, help="Stop after this many appended rows (0 = no limit).")
    parser.add_argument("--min-seeds", type=int, default=0, help="Minimum seeds required to append a row (0 = require all requested seeds).")
    parser.add_argument("--dry-run", action="store_true", help="Report actions without writing TSV or TeX files.")
    parser.add_argument("--verbose-runner", action="store_true", help="Keep TrainRunner skip/setup messages visible.")
    parser.add_argument("--backfill", dest="backfill", action="store_true", default=True)
    parser.add_argument("--no-backfill", dest="backfill", action="store_false")
    parser.add_argument("--render", dest="render", action="store_true", default=True)
    parser.add_argument("--no-render", dest="render", action="store_false")
    return parser


def backfill(
    *,
    models: Sequence[str],
    results_tsv: Path,
    results_dir: Path,
    tasks_tsv: Path,
    dry_run: bool = False,
    max_cells: int = 0,
    verbose_runner: bool = False,
    min_seeds: int = 0,
) -> dict[str, int]:
    from src.train.run import build_train_cfg
    from src.train.utils import parse_train_tasks
    from src.utils.run_helpers import resolve_seeds

    if not tasks_tsv.is_file():
        print(f"{LOG_PREFIX} Missing tasks TSV: {tasks_tsv}")
        return {"missing_tasks_tsv": 1, "appended": 0}

    cfg = build_train_cfg(
        [
            "train.run_tasks_tsv",
            "True",
            "train.tasks_tsv",
            str(tasks_tsv),
            "save_results.output_dir",
            str(results_dir),
        ]
    )
    cfg.save_results.enabled = not dry_run

    model_filter = {_normalize_name(m) for m in models}
    all_tasks = parse_train_tasks(str(tasks_tsv))
    tasks = [task for task in all_tasks if _normalize_name(task.get("model")) in model_filter]
    if not tasks:
        print(f"{LOG_PREFIX} No tasks matched models={sorted(model_filter)} in {tasks_tsv}")
        return {"tasks": 0, "appended": 0}

    requested_runs = int(getattr(getattr(cfg, "train", None), "num_runs", 0) or 0)
    seeds = resolve_seeds(cfg, requested_count=requested_runs)
    required_seeds = min_seeds if min_seeds > 0 else len(seeds)

    present = _valid_result_cells(_read_result_rows(results_tsv), models=model_filter)
    expected = _expected_cells(model_filter)

    appended = 0
    skipped_present = 0
    no_artifact = 0
    failures = 0
    for task in tasks:
        key = _cell_key_from_task(task)
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
                seeds=seeds,
                dry_run=dry_run,
                verbose_runner=verbose_runner,
                required_seeds=required_seeds,
            )
        except Exception as exc:
            failures += 1
            print(f"{LOG_PREFIX} Failed {task.get('model')}/{task.get('dataset')}/{task.get('fixed_split')}: {exc}")
            continue

        if not result:
            no_artifact += 1
            continue

        appended += 1
        present.add(key)

    print(
        f"{LOG_PREFIX} tasks={len(tasks)} already_present={skipped_present} "
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
    seeds: Sequence[int],
    dry_run: bool,
    verbose_runner: bool,
    required_seeds: int,
) -> bool:
    from src.train.trainer import TrainRunner
    from src.train.utils import _build_task_cfg
    from src.utils.run_helpers import (
        aggregate_run_metrics,
        checkpoint_path_for_runner,
        collect_run_metrics,
    )
    from src.utils.save_results import append_workflow_result

    base_cfg = _build_task_cfg(cfg, dict(task))
    run_metrics: list[dict[str, float]] = []
    completed_seeds: list[int] = []
    source_paths: list[str] = []
    checkpoint_paths: list[str] = []

    for seed in seeds:
        run_cfg = base_cfg.clone()
        run_cfg.seed = int(seed)
        stdout_context = contextlib.nullcontext() if verbose_runner else contextlib.redirect_stdout(io.StringIO())
        with stdout_context:
            runner = TrainRunner(run_cfg)
        checkpoint_path = checkpoint_path_for_runner(runner)
        metrics = collect_run_metrics(runner, log_prefix=LOG_PREFIX)
        source_path = checkpoint_path if metrics and os.path.isfile(checkpoint_path) else ""
        if not metrics:
            metrics, source_path = _load_log_metrics(runner)

        if not metrics:
            continue
        run_metrics.append(metrics)
        completed_seeds.append(int(seed))
        if checkpoint_path:
            checkpoint_paths.append(checkpoint_path)
        if source_path:
            source_paths.append(source_path)

    if len(run_metrics) < required_seeds:
        print(
            f"{LOG_PREFIX} Skipping {task.get('model')}/{task.get('dataset')}/"
            f"{task.get('fixed_split')}: only {len(run_metrics)}/{required_seeds} "
            f"seeds have complete artifacts (completed={completed_seeds})"
        )
        return False

    summary = aggregate_run_metrics(run_metrics)
    metric_name = _metric_for_task(task)
    metric_stats = summary["metric_stats"]
    if not _valid_metric_values(metric_stats.get(metric_name, {}).get("mean"), metric_stats.get(metric_name, {}).get("std")):
        return False

    started_at, ended_at = _artifact_time_window(source_paths or checkpoint_paths)
    label = (
        f"{base_cfg.model.name}/{base_cfg.train.dataset.name}/"
        f"split={list(_split_key(base_cfg.train.dataset.fixed_split))}"
    )
    if dry_run:
        print(f"{LOG_PREFIX} DRY-RUN would append {label} seeds={completed_seeds}")
        return True

    append_workflow_result(
        cfg=base_cfg,
        workflow="train",
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
    results_tsv: Path,
    results_dir: Path,
    dry_run: bool = False,
) -> int:
    rows = _read_result_rows(results_tsv)
    latest = _latest_rows(rows)
    rendered = 0
    for shot in ("f5", "f100"):
        text = _render_table(shot=shot, latest=latest)
        out_path = results_dir / f"train_{shot}_table.tex"
        if dry_run:
            print(f"{LOG_PREFIX} DRY-RUN would write {out_path}")
            rendered += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        rendered += 1
    return rendered


def _read_result_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle, delimiter="\t")]


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
        metric = _metric_for_dataset(key.dataset)
        if _valid_metric_values(row.get(f"{metric}_mean"), row.get(f"{metric}_std")):
            valid.add(key)
    return valid


def _expected_cells(models: set[str]) -> set[CellKey]:
    keys: set[CellKey] = set()
    for dataset in DATASETS:
        for split in (dataset.f5_split, dataset.f100_split):
            for model in models:
                keys.add(CellKey(model, dataset.name, split))
    return keys


def _cell_key_from_task(task: Mapping[str, Any]) -> CellKey | None:
    model = _normalize_name(task.get("model"))
    dataset = _normalize_name(task.get("dataset"))
    split = _split_key(task.get("fixed_split"))
    if not model or not dataset or not split:
        return None
    return CellKey(model, dataset, split)


def _cell_key_from_result_row(row: Mapping[str, str]) -> CellKey | None:
    model = _normalize_name(row.get("model.name"))
    dataset = _normalize_name(row.get("train.dataset.name"))
    split = _split_key(row.get("train.dataset.fixed_split"))
    if not model or not dataset or not split:
        return None
    return CellKey(model, dataset, split)


def _render_table(shot: str, latest: Mapping[CellKey, Mapping[str, str]]) -> str:
    shot_label = "5" if shot == "f5" else "100"
    lp_split_text = (
        "5$\\%$-10$\\%$-10$\\%$" if shot == "f5" else "10$\\%$-5$\\%$-10$\\%$"
    )
    lines = [
        "% Generated from outputs/results/train.tsv.",
        "% Selection: latest matching row by file order per model/dataset cell.",
        f"% Splits: [{shot_label}, 0.0, 1.0] for NC/GC/GR datasets; LP datasets use "
        + ("[0.05, 0.1, 0.1]." if shot == "f5" else "[0.1, 0.05, 0.1]."),
        "% Metrics: test_acc for NC/MNIST GC, test_auc for LP/Toxcast GC, test_mae for QM7b GR.",
        "% Missing cells are shown as -- because no matching row was present in the TSV.",
        "\\begin{table*}[t]",
        "\\caption{",
        "Extensive comparison of supervised backbone models on datasets spanning multiple domains.",
        f"For node classification (NC) and MNIST graph classification (GC), we report \\emph{{{shot_label}-shot}} accuracy scores.",
        f"For Toxcast multilabel graph classification, we report macro ROC-AUC using \\emph{{{shot_label}}} labeled training graphs.",
        f"For the graph regression (GR) task, we report \\emph{{{shot_label}-shot}} MAE scores.",
        f"For the link prediction (LP) task, we present AUC scores following the split \\emph{{{lp_split_text}}}.",
        "}",
        f"\\label{{table:results_main_{shot}}}",
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
        "& \\textbf{Social} & \\textbf{Academic} & \\textbf{Web}",
        "& \\textbf{Transport} & \\textbf{Web} & \\textbf{Academic}",
        "& \\textbf{Molecular} & \\textbf{Chemistry} & \\textbf{Vision} \\\\",
        "& \\textbf{Photo} & \\textbf{Ogbn-arxiv} & \\textbf{DBLP}",
        "& \\textbf{Airports} & \\textbf{Chameleon} & \\textbf{Cornell}",
        "& \\textbf{QM7b} & \\textbf{Toxcast} & \\textbf{MNIST} \\\\",
        "& NC & NC & LP",
        "& NC & NC & LP",
        "& GR & GC & GC \\\\",
        "\\midrule",
        f"& \\multicolumn{{9}}{{c}}{{Supervised \\emph{{{shot_label}-shot}}}} \\\\",
        "\\midrule",
    ]
    for model in MODELS:
        lines.extend(_render_model_row(shot=shot, model=model, latest=latest))
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


def _render_model_row(
    *,
    shot: str,
    model: ModelSpec,
    latest: Mapping[CellKey, Mapping[str, str]],
) -> list[str]:
    values = []
    for dataset in DATASETS:
        split = dataset.f5_split if shot == "f5" else dataset.f100_split
        key = CellKey(model.name, dataset.name, split)
        values.append(_format_cell(latest.get(key), dataset.metric))
    return [
        model.label,
        f"    & {' & '.join(values[:3])}",
        f"    & {' & '.join(values[3:6])}",
        f"    & {' & '.join(values[6:])} \\\\",
    ]


def _format_cell(row: Mapping[str, str] | None, metric: str) -> str:
    if row is None:
        return "--"
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if not _valid_metric_values(mean, std):
        return "--"
    mean_f = float(mean)
    std_f = float(std)
    if metric == "test_mae":
        return f"{mean_f:.2f}$_{{\\pm{std_f:.2f}}}$"
    return f"{100.0 * mean_f:.2f}$_{{\\pm{100.0 * std_f:.2f}}}$"


def _load_log_metrics(runner: TrainRunner) -> tuple[dict[str, float], str]:
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


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _resolve_models(raw: str) -> list[str]:
    if str(raw).strip().lower() == "auto":
        return [m.name for m in MODELS]
    return [_normalize_name(part) for part in str(raw).split(",") if _normalize_name(part)]


if __name__ == "__main__":
    raise SystemExit(main())
