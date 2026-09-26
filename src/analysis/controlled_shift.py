"""App. C.7: prompt values under controlled context shifts (paper Table 9, Fig. 7).

Per dataset of ``analysis.controlled_shift.datasets`` (Photo, Chameleon): the
full graph (node classification) with its own within-dataset DGI GCN, resolved
like the finetune runner from ``pretrain.*`` / ``model.*`` with
``pretrain.dataset.name`` set to the dataset. The support of repetition ``s``
is the ``fixed_split`` few-shot split of seed ``s`` under
``analysis.split_root``, fixed across views.
The views and the discrepancy samples draw from two streams seeded by
``np.random.SeedSequence(s).generate_state(2)``, so they share no draws with
GapTune's initialiser stream ``s`` or with other repetitions.
Per repetition:

- views: :func:`controlled_views` (``CONTROL`` + family x strength) on the
  simple undirected edge set (App. A.2); for a directed loaded graph such as
  Chameleon this adds the reverse edges to every view, the control included;
- discrepancies for the Fig. 7 x-axis: :func:`view_context_gaps`;
- arms, each a fresh ``FinetuneGapTune`` fit on the view with the same support:
  head (``prompt_locations none``), target (``c_t``), free (learned values in
  the norm envelope) and gap (``c_s - c_t``); all other ``finetune.gaptune``
  settings are shared. The head is drawn first from stream ``s``, so every arm
  starts from the same head. ``updates`` full-graph updates with the GapTune
  optimiser and gradient clip; the state the finetune runner's monitor selects
  is restored (training loss without a validation split) and scored on test.
- collections: the source bank is every unprompted observation of the
  unperturbed graph (``CONTROL``: nodes and directed messages, native
  self-loops included); the target collection is every observation of the
  view. The control reuses the source tensors, so its gap prompts are exactly
  zero and the gap arm follows the head-only trajectory (bitwise on CPU; on
  GPU up to the atomic summation order of the frozen encoder's aggregation).

Summary: per condition (the control once), mean eta / discrepancies /
accuracies and the paired effects ``D = Acc_gap - Acc_v`` for ``v`` in
{free, target} with pointwise t intervals (Eq. 73-74). Accuracies are in
percent, effects in percentage points. GCN is the paper's backbone; the
NodeFormer driver's fixed projections are not wired in.
Outputs in ``<analysis.output_dir>/controlled_shift/<checkpoint stem>/`` per
dataset: ``rep<seed>.json`` (with ``loader_edges`` / ``view_edges``, the
directed edge counts of the loaded graph and of every view), ``conditions.tsv``
(one row per repetition and view) and ``effects.tsv``. The two TSVs cover the
repetitions of one invocation only, so run all of ``analysis.repetitions`` in
one invocation.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
from torch_geometric.data import Data

from src.finetune.encoders.gaptune import get_gaptune_driver
from src.finetune.methods.gaptune import FinetuneGapTune
from src.finetune.monitoring import resolve_finetune_monitor_spec
from src.utils.checkpoint import save_json_atomic
from src.utils.dataset_helpers import make_workflow_loaders
from src.utils.monitoring import is_metric_improved, monitor_uses_train_split, resolve_monitor_value
from src.utils.training import run_epoch_loop

from .context_gap import view_context_gaps
from .perturbations import controlled_views, simple_undirected_edges
from .predictor import full_graph_setup, pretrained_checkpoint, study_dir, write_tsv
from .stats import paired_t

# arm -> finetune.gaptune overrides
ARMS = {
    "head": {"prompt_locations": "none"},
    "target": {"value_mode": "target"},
    "free": {"value_mode": "free"},
    "gap": {"value_mode": "gap"},
}
COMPARATORS = ("free", "target")


def source_bank(reference) -> dict[str, torch.Tensor]:
    """Every unprompted observation of a graph, keyed like ``ContextGapPrompt`` types."""
    obs = reference.obs
    return {"N": obs.h0, **{f"M{layer}": o.messages for layer, o in enumerate(obs.layers, start=1)}}


def _loss(task, model, graph, reference, mask_attr: str):
    """``(loss, accuracy)`` of the nodes in ``graph[mask_attr]``."""
    mask = graph[mask_attr]
    logits = task.head(task.encode(model, graph, reference=reference)[mask])
    return task.objective.loss_from_logits(logits=logits, labels=graph.y[mask])


@torch.no_grad()
def evaluate(task, model, graph, reference, mask_attr: str) -> float:
    task.eval()
    return float(_loss(task, model, graph, reference, mask_attr)[1])


def fit_arm(cfg, arm: str, model, graph, reference, source: dict, monitor):
    """A fresh GapTune task of ``arm`` fit on ``graph``; returns it with the monitor-selected state and update.

    ``reference`` is the view's unprompted pass (``collect=True``) and
    ``source`` the source bank; ``cfg.seed`` fixes the initialisation.
    """
    arm_cfg = cfg.clone()
    for key, value in ARMS[arm].items():
        setattr(arm_cfg.finetune.gaptune, key, value)
    task = FinetuneGapTune(arm_cfg)
    task.validate_encoder(model)
    if task.prompt.types:
        task.prompt.set_source_bank(source)
        task.prompt.refresh_retained()
    optimizer = task.build_optimizers(model)["primary"]

    def forward_fn(data, device):
        loss, acc = _loss(task, model, data, reference, "train_mask")
        return loss, {"train_acc": float(acc)}

    updates = int(cfg.analysis.controlled_shift.updates)
    best, best_state, best_update = monitor.best_metric, None, updates
    for update in range(1, updates + 1):
        task.train()
        loss, logs = run_epoch_loop(
            forward_fn=forward_fn,
            loader=[graph],
            optimizer=optimizer,
            device=graph.x.device,
            grad_clip=float(task.method_cfg.grad_clip),
        )
        task.prompt.refresh_retained()  # as GapTune's train_epoch: evaluation reads C*_s
        val = {} if monitor_uses_train_split(monitor.name) else {
            "val_acc": evaluate(task, model, graph, reference, "val_mask")
        }
        value = resolve_monitor_value(monitor.name, train_loss=loss, train_logs=logs, val_metrics=val, test_metrics={})
        if monitor.name is None or is_metric_improved(value, best, monitor.mode):
            best, best_state, best_update = value, copy.deepcopy(task.state_dict()), update
    if best_state is not None:
        task.load_state_dict(best_state)
    return task, best_update


def run_repetition(cfg, model, data, seed: int) -> list[dict]:
    """One row per view (``CONTROL`` first): eta, discrepancies, test accuracy (%) and selected update per arm.

    ``model`` is the frozen vanilla encoder in eval mode; ``data`` carries
    labels and the train / val / test masks.
    """
    cs_cfg = cfg.analysis.controlled_shift
    view_seed, gap_seed = (int(s) for s in np.random.SeedSequence(seed).generate_state(2))
    views = controlled_views(
        data.x, data.edge_index, list(cs_cfg.strengths), generator=torch.Generator().manual_seed(view_seed)
    )
    gaps = view_context_gaps(
        model,
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
        task_cls=FinetuneGapTune,
    )
    run_cfg = cfg.clone()
    run_cfg.seed = seed
    labels_and_split = {name: data[name] for name in ("y", "train_mask", "val_mask", "test_mask")}
    driver = get_gaptune_driver(model)
    source, rows = None, []
    for key, view in views.items():
        graph = Data(x=view.x, edge_index=view.edge_index, **labels_and_split)
        with torch.no_grad():
            reference = driver.forward(model, graph, collect=True)
        if source is None:  # CONTROL, the unperturbed graph, comes first
            source = source_bank(reference)
        row = {
            "seed": seed,
            "family": key[0],
            "alpha": key[1],
            "swaps": view.swaps,
            "eta": view.eta,
            "delta_H": gaps[key]["H"],
            "delta_M": gaps[key]["M"],
        }
        for arm in ARMS:
            task, best_update = fit_arm(run_cfg, arm, model, graph, reference, source, monitor)
            row[f"acc_{arm}"] = 100 * evaluate(task, model, graph, reference, "test_mask")
            row[f"update_{arm}"] = best_update
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Per condition: means over repetitions and paired ``Acc_gap - Acc_v`` effects with t intervals."""
    effects = []
    for family, alpha in dict.fromkeys((r["family"], r["alpha"]) for r in rows):
        cells = [r for r in rows if (r["family"], r["alpha"]) == (family, alpha)]
        row = {"family": family, "alpha": alpha, "n": len(cells)}
        for key in ("eta", "delta_H", "delta_M", *(f"acc_{arm}" for arm in ARMS)):
            row[f"{key}_mean"] = float(np.mean([c[key] for c in cells]))
        for v in COMPARATORS:
            t = paired_t([c["acc_gap"] - c[f"acc_{v}"] for c in cells])
            row.update({f"gap_{v}_{k}": t[k] for k in ("mean", "low", "high", "p")})
        effects.append(row)
    return effects


def run_dataset(cfg, name: str, path: str, device: torch.device) -> None:
    """Every repetition of ``analysis.repetitions`` on dataset ``name`` with checkpoint ``path``, then the summary."""
    cfg = cfg.clone()
    dataset, model = full_graph_setup(cfg, name, path, device)
    out_dir = study_dir(cfg, "controlled_shift", path)
    print(f"[Analysis][controlled_shift] dataset={name} checkpoint={path} output={out_dir}")

    rows = []
    for seed in (int(s) for s in cfg.analysis.repetitions):
        loader, _, _ = make_workflow_loaders(
            dataset=dataset,
            dataset_name=name,
            task_level_raw="node",
            effective_task_level="node",
            batch_size=1,
            num_workers=0,
            split=tuple(cfg.analysis.controlled_shift.fixed_split),
            seed=seed,
            induced=False,
            split_root=cfg.analysis.split_root,
        )
        data = next(iter(loader)).to(device)
        rep = run_repetition(cfg, model, data, seed)
        save_json_atomic(
            os.path.join(out_dir, f"rep{seed}.json"),
            {
                "checkpoint": path,
                "seed": seed,
                "loader_edges": int(data.edge_index.size(1)),
                "view_edges": 2 * int(simple_undirected_edges(data.edge_index).size(0)),
                "conditions": rep,
            },
        )
        print(f"[Analysis][controlled_shift] {name} seed={seed} " + " ".join(
            f"{r['family']}@{r['alpha']:g}:" + "/".join(f"{r[f'acc_{arm}']:.2f}" for arm in ARMS) for r in rep
        ) + " (head/target/free/gap %)")
        rows.extend(rep)

    effects = summarize(rows)
    write_tsv(os.path.join(out_dir, "conditions.tsv"), rows)
    write_tsv(os.path.join(out_dir, "effects.tsv"), effects)
    for r in effects:
        print(
            f"[Analysis][controlled_shift] {name} {r['family']}@{r['alpha']:g} delta_M={r['delta_M_mean']:.4f} "
            + " ".join(f"gap-{v}={r[f'gap_{v}_mean']:+.2f} [{r[f'gap_{v}_low']:+.2f}, {r[f'gap_{v}_high']:+.2f}]"
                       for v in COMPARATORS)
        )


def run_controlled_shift(cfg) -> int:
    """``analysis.study controlled_shift``: every dataset of ``analysis.controlled_shift.datasets``."""
    datasets = [str(name) for name in cfg.analysis.controlled_shift.datasets]
    if str(cfg.analysis.pretrained_checkpoint or "").strip() and len(datasets) != 1:
        print("[Analysis][controlled_shift] analysis.pretrained_checkpoint is one within-dataset checkpoint; "
              "select a single dataset with analysis.controlled_shift.datasets.")
        return 1
    paths = {name: pretrained_checkpoint(cfg, name)[0] for name in datasets}
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        print(f"[Analysis][controlled_shift] Unable to resolve the pretrained checkpoint of {missing} "
              "(pretrain.* / model.* or analysis.pretrained_checkpoint).")
        return 1
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    for name, path in paths.items():
        run_dataset(cfg, name, path, device)
    return 0
