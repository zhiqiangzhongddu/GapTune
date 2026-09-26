"""Parameter, time and memory accounting (paper App. C.10, Tables 12-13).

Workload: the bias-enabled 3-layer GCN ``100 -> 128 -> 128 -> 128`` without
normalization or dropout (random seeded weights), an eight-class affine head
on the focal-node readout, ``K_q = finetune.gaptune.num_queries`` (8), and one
fixed batch of 64 disjoint synthetic subgraphs with 8,192 nodes and 65,536
directed messages including the native GCN self-loops. Every method is a
GapTune task over the same driver, readout and head (``prompt_locations
none`` is the head-only control; full fine-tuning makes that single pass
trainable), so all share the batch sequence and evaluation calls:

- preparation: GapTune runs the source-free proxy inversion
  (``finetune.gaptune.proxy``: 16 graphs x 32 nodes, 1,000 updates, EdgePred
  dot-product pretext) and response extraction; GapTune+ only extracts
  responses, from a synthetic stand-in for the ZINC collection of a
  ZINC/GCN/EdgePred checkpoint (256 graphs of 23 nodes and 25 undirected
  edges; no dataset is loaded). Both use the source caps 512 / 2,048. The
  free-value controls read the same source bank for their norm envelope but,
  as in Table 12, are charged no preparation;
- adaptation: ``analysis.cost.updates`` Adam updates on the fixed batch plus
  ``analysis.cost.evaluations`` fixed-batch evaluations with the retained
  contexts; update and inference times are per call, total = preparation +
  adaptation.

Stage times are bracketed by ``torch.cuda.synchronize``; a stage's peak
memory is its maximum allocated bytes above the method's starting allocation
(MiB, the ``*_mib`` columns; NaN off CUDA). Every method first runs one
untimed warm-up block (one update, one evaluation, one inversion update).
Outputs in ``<analysis.output_dir>/cost/<run tag>/``: ``blocks.tsv`` (one row
per timing block and method), ``summary.tsv`` (Tables 12-13: census plus mean
and sample SD over blocks) and ``cost.json``.
"""

from __future__ import annotations

import copy
import gc
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from src.analysis.predictor import write_tsv
from src.finetune.methods.gaptune import FinetuneGapTune
from src.model import build_encoder_from_cfg
from src.utils.checkpoint import save_json_atomic
from src.utils.config_helpers import cfg_default, tag_if_nondefault
from src.utils.paths import ensure_dir
from src.utils.run_helpers import resolve_seeds

IN_DIM, WIDTH, NUM_LAYERS, NUM_CLASSES = 100, 128, 3, 8
# 64 * 128 = 8,192 nodes; 64 * (2 * 448 + 128) = 65,536 directed messages with self-loops
NUM_GRAPHS, GRAPH_NODES, GRAPH_EDGES = 64, 128, 448
# GapTune+ source: ZINC-sized graphs; extraction stops after two 64-graph chunks (4x the source caps)
SOURCE_GRAPHS, SOURCE_NODES, SOURCE_EDGES = 256, 23, 25
FULL_FINETUNING = "full_finetuning"
# name -> (Table 12 label, value_mode, prompt_locations, timed preparation)
METHODS = {
    "head_only": ("Head only", "gap", "none", None),
    FULL_FINETUNING: ("Full fine-tuning", "gap", "none", None),
    "free_node": ("Free: node only", "free", "node", None),
    "free_message": ("Free: message only", "free", "message", None),
    "free_node_message": ("Free: node + message", "free", "node_message", None),
    "gaptune": ("GapTune", "gap", "node_message", "proxy"),
    "gaptune_plus": ("GapTune+", "gap", "node_message", "source"),
}
METRICS = ("prep_s", "update_ms", "adapt_s", "infer_ms", "total_s", "prep_mib", "adapt_mib", "peak_mib")


class _FullFinetune(FinetuneGapTune):
    """Full fine-tuning control: the head-only readout and head over a trainable encoder pass."""

    def encode(self, model, data):
        return self.driver.forward(model, data, projections=self.projections).node_repr


def workload_cfg(cfg):
    """Clone of *cfg* with the App. C.10 encoder and a node-level (induced) eight-class task."""
    wcfg = cfg.clone()
    wcfg.seed = resolve_seeds(cfg, requested_count=1)[0]
    model = wcfg.model
    model.name, model.num_layers, model.in_dim, model.hidden_dim, model.out_dim = "gcn", NUM_LAYERS, IN_DIM, WIDTH, WIDTH
    model.dropout, model.use_batchnorm = 0.0, False
    # Source-free inversion with the dot-product EdgePred scorer (no retained pretext state needed).
    wcfg.pretrain.method = "edge_pred"
    wcfg.pretrain.edge_pred.use_mlp_scorer = False
    ds = wcfg.finetune.dataset
    ds.task_level, ds.induced, ds.task_type, ds.num_classes, ds.label_dim = "node", True, "classification", NUM_CLASSES, 1
    return wcfg


def synthetic_graphs(
    num_graphs: int, generator: torch.Generator, num_nodes: int = GRAPH_NODES, num_edges: int = GRAPH_EDGES
) -> list[Data]:
    """Simple undirected graphs (both directions, no self-loops) with N(0, 1) features,
    focal node 0 and a uniform class label."""
    row, col = torch.triu_indices(num_nodes, num_nodes, offset=1)
    graphs = []
    for _ in range(num_graphs):
        pick = torch.randperm(row.numel(), generator=generator)[:num_edges]
        u, v = row[pick], col[pick]
        graphs.append(
            Data(
                x=torch.randn(num_nodes, IN_DIM, generator=generator),
                edge_index=torch.stack([torch.cat([u, v]), torch.cat([v, u])]),
                y=torch.randint(NUM_CLASSES, (1,), generator=generator),
                target_node_index=torch.tensor([0]),
            )
        )
    return graphs


def workload_encoder(wcfg) -> torch.nn.Module:
    """Frozen random GCN (seeded by ``wcfg.seed``) in eval mode, on CPU."""
    torch.manual_seed(int(wcfg.seed))
    return build_encoder_from_cfg(wcfg, IN_DIM).eval().requires_grad_(False)


def build_method(wcfg, name: str, encoder):
    """``(task, trainable parameters)`` of one method; full fine-tuning also unfreezes *encoder*."""
    _, value_mode, prompt_locations, preparation = METHODS[name]
    mcfg = wcfg.clone()
    gt = mcfg.finetune.gaptune
    gt.value_mode, gt.prompt_locations, gt.plus = value_mode, prompt_locations, preparation != "proxy"
    task = (_FullFinetune if name == FULL_FINETUNING else FinetuneGapTune)(mcfg)
    task.validate_encoder(encoder)
    params = task.parameters_to_optimize()
    if name == FULL_FINETUNING:
        # GNNEncoder also allocates BatchNorm modules that it never uses without use_batchnorm.
        encoder_params = list(encoder.convs.parameters())
        for p in encoder_params:
            p.requires_grad_(True)
        params = encoder_params + params
    return task, params


def retained_scalars(task) -> int:
    """Scalars of the retained inference state (Eq. 18): ``C*_s`` for gaps, one source norm per free-value type."""
    return sum(
        p.retained_source_norm.numel() if p.value_mode == "free" else p.retained_source_context.numel()
        for p in task.prompt.types.values()
    )


def parameter_census(cfg) -> dict[str, dict[str, int]]:
    """Trainable parameters (head included) and retained scalars per method."""
    wcfg = workload_cfg(cfg)
    encoder = workload_encoder(wcfg)
    census = {}
    for name in METHODS:
        task, params = build_method(wcfg, name, copy.deepcopy(encoder))
        census[name] = {"trainable": sum(p.numel() for p in params), "retained_scalars": retained_scalars(task)}
    return census


def prepare_source(task, encoder, preparation: str, source_graphs, device) -> None:
    """Fill the fixed source bank: proxy inversion + extraction, or extraction from *source_graphs*."""
    if preparation == "proxy":
        task.prepare_with_encoder(model=encoder, device=device, pretrain_cfg={}, pretrain_extra={})
    else:
        task.set_source_from_graphs(encoder, source_graphs, device)


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stage_start(device) -> float:
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _stage_end(device, start: float, base: int) -> tuple[float, float]:
    """``(seconds, peak MiB above base)``."""
    _sync(device)
    seconds = time.perf_counter() - start
    if device.type != "cuda":
        return seconds, float("nan")
    return seconds, (torch.cuda.max_memory_allocated(device) - base) / 2**20


def run_block(wcfg, name: str, encoder, batch, source_graphs, device, *, updates: int, evaluations: int) -> dict:
    """One timing block of one method, from a fresh copy of *encoder* and a fresh head / prompt."""
    gc.collect()
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device) if cuda else 0
    encoder = copy.deepcopy(encoder).to(device)
    task, params = build_method(wcfg, name, encoder)
    preparation = METHODS[name][3]
    record = {"prep_s": 0.0, "prep_mib": 0.0 if cuda else float("nan")}
    if preparation is not None:
        start = _stage_start(device)
        prepare_source(task, encoder, preparation, source_graphs, device)
        record["prep_s"], record["prep_mib"] = _stage_end(device, start, base)
    elif task.prompt.types:
        prepare_source(task, encoder, "source", source_graphs, device)  # free-value envelope, not charged
    batch = batch.to(device)
    optimizer = torch.optim.Adam(params, lr=float(wcfg.finetune.lr), weight_decay=float(wcfg.finetune.weight_decay))
    grad_clip = float(wcfg.finetune.gaptune.grad_clip)
    update_ms, infer_ms = [], []
    start = _stage_start(device)
    task.train()
    for step in range(1, updates + 1):
        tick = time.perf_counter()
        loss, _ = task._forward(encoder, batch, device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        _sync(device)
        update_ms.append(1e3 * (time.perf_counter() - tick))
        if step % (updates // evaluations) == 0:
            task.eval()
            task.prompt.refresh_retained()
            _sync(device)
            tick = time.perf_counter()
            with torch.no_grad():
                task._forward(encoder, batch, device)
            _sync(device)
            infer_ms.append(1e3 * (time.perf_counter() - tick))
            task.train()
    record["adapt_s"], record["adapt_mib"] = _stage_end(device, start, base)
    record.update(
        update_ms=float(np.mean(update_ms)),
        infer_ms=float(np.mean(infer_ms)),
        total_s=record["prep_s"] + record["adapt_s"],
        peak_mib=max(record["prep_mib"], record["adapt_mib"]),
    )
    return record


def summarize(records: list[dict], census: dict) -> list[dict]:
    """Tables 12-13: the census plus mean and sample SD over the timing blocks of each method."""
    rows = []
    for name, (label, *_) in METHODS.items():
        row = {"method": name, "label": label, **census[name]}
        for key in METRICS:
            values = np.array([r[key] for r in records if r["method"] == name], dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_sd"] = float(values.std(ddof=1)) if values.size > 1 else float("nan")
        rows.append(row)
    return rows


def _run_tag(cfg, seed: int) -> str:
    """``seed<s>`` plus the non-default timing settings and ``finetune.gaptune`` options (proxy
    included), so shortened or resized runs never overwrite full ones."""
    cost = cfg.analysis.cost
    gaptune_cfg = cfg.clone()
    gaptune_cfg.finetune.gaptune.plus = False  # also tag the proxy options GapTune prepares with
    tags = [
        f"seed{seed}",
        tag_if_nondefault("b", int(cost.timing_blocks), cfg_default("analysis.cost.timing_blocks")),
        tag_if_nondefault("u", int(cost.updates), cfg_default("analysis.cost.updates")),
        tag_if_nondefault("e", int(cost.evaluations), cfg_default("analysis.cost.evaluations")),
        FinetuneGapTune.variant_tag(gaptune_cfg),
    ]
    return "-".join(tag for tag in tags if tag)


def _format(row: dict, key: str, digits: int) -> str:
    return f"{row[f'{key}_mean']:.{digits}f}+-{row[f'{key}_sd']:.{digits}f}"


def run_cost_study(cfg) -> int:
    """``analysis.study cost``: census, warm-up, then ``analysis.cost.timing_blocks`` blocks of every method."""
    cost = cfg.analysis.cost
    blocks, updates, evaluations = int(cost.timing_blocks), int(cost.updates), int(cost.evaluations)
    if blocks < 1 or evaluations < 1 or updates < evaluations or updates % evaluations:
        print(
            "[Analysis][cost] analysis.cost needs timing_blocks >= 1 and updates a positive multiple of "
            f"evaluations; got {blocks}, {updates}, {evaluations}.",
            file=sys.stderr,
        )
        return 1
    wcfg = workload_cfg(cfg)
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(int(wcfg.seed))
    batch = Batch.from_data_list(synthetic_graphs(NUM_GRAPHS, generator))
    source_graphs = synthetic_graphs(SOURCE_GRAPHS, generator, SOURCE_NODES, SOURCE_EDGES)
    encoder = workload_encoder(wcfg)
    census = parameter_census(cfg)
    out_dir = ensure_dir(os.path.join(cfg.analysis.output_dir, "cost", _run_tag(cfg, int(wcfg.seed))))
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"[Analysis][cost] device={device_name} output={out_dir}")

    warm_cfg = wcfg.clone()
    warm_cfg.finetune.gaptune.proxy.updates = 1
    for name in METHODS:
        run_block(warm_cfg, name, encoder, batch, source_graphs, device, updates=1, evaluations=1)
    records = []
    for block in range(blocks):
        for name in METHODS:
            record = run_block(wcfg, name, encoder, batch, source_graphs, device, updates=updates, evaluations=evaluations)
            records.append({"block": block, "method": name, **record})
            print(
                f"[Analysis][cost] block {block + 1}/{blocks} {METHODS[name][0]}: prep={record['prep_s']:.2f}s "
                f"update={record['update_ms']:.2f}ms adapt={record['adapt_s']:.2f}s infer={record['infer_ms']:.2f}ms "
                f"memory prep/adapt={record['prep_mib']:.1f}/{record['adapt_mib']:.1f}MiB"
            )

    rows = summarize(records, census)
    write_tsv(str(out_dir / "blocks.tsv"), records)
    write_tsv(str(out_dir / "summary.tsv"), rows)
    save_json_atomic(
        str(out_dir / "cost.json"),
        {
            "device": device_name,
            "torch": torch.__version__,
            "seed": int(wcfg.seed),
            "workload": {
                "widths": [IN_DIM] + [WIDTH] * NUM_LAYERS,
                "num_classes": NUM_CLASSES,
                "num_graphs": NUM_GRAPHS,
                "nodes": int(batch.num_nodes),
                "messages_with_self_loops": int(batch.num_edges + batch.num_nodes),
            },
            "settings": {
                "timing_blocks": blocks,
                "updates": updates,
                "evaluations": evaluations,
                "num_queries": int(wcfg.finetune.gaptune.num_queries),
                "proxy_graphs": int(wcfg.finetune.gaptune.proxy.num_graphs),
                "proxy_nodes": int(wcfg.finetune.gaptune.proxy.num_nodes),
                "proxy_updates": int(wcfg.finetune.gaptune.proxy.updates),
                "source_max_nodes": int(wcfg.finetune.gaptune.source_max_nodes),
                "source_max_messages": int(wcfg.finetune.gaptune.source_max_messages),
            },
            "summary": rows,
            "blocks": records,
        },
    )
    for row in rows:
        print(
            f"[Analysis][cost] {row['label']}: trainable={row['trainable']:,} retained={row['retained_scalars']:,} "
            f"prep={_format(row, 'prep_s', 2)}s update={_format(row, 'update_ms', 2)}ms "
            f"adapt={_format(row, 'adapt_s', 2)}s infer={_format(row, 'infer_ms', 2)}ms "
            f"total={_format(row, 'total_s', 2)}s memory prep/adapt/peak={_format(row, 'prep_mib', 1)}/"
            f"{_format(row, 'adapt_mib', 1)}/{_format(row, 'peak_mib', 1)}MiB"
        )
    return 0
