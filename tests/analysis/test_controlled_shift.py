"""App. C.7 controlled-shift study on a tiny synthetic graph (random GCN weights, CPU).

- zero shift: the gap arm's prompts vanish exactly and it follows the
  head-only trajectory (same selected update, head and predictions);
- the target arm excludes source observations (another source bank changes
  nothing), while the gap arm reads them and retains the source contexts of
  its selected queries (Prop. B.3);
- a repetition gives one row per view with zero control discrepancies, every
  arm of every view gets the unperturbed graph's source bank, and the
  summary's paired effects are the mean gap-minus-comparator differences;
- single-graph pooling / mixing in the GapTune prompt matches the per-graph path;
- CLI registration and the one-dataset rule for an explicit checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from yacs.config import CfgNode as CN

from src.analysis import controlled_shift
from src.analysis import run as analysis_run
from src.analysis.controlled_shift import (
    ARMS,
    COMPARATORS,
    evaluate,
    fit_arm,
    run_controlled_shift,
    run_repetition,
    source_bank,
    summarize,
)
from src.analysis.perturbations import CONTROL, FAMILIES, controlled_views
from src.config import set_cfg
from src.finetune.encoders.gaptune import get_gaptune_driver
from src.finetune.methods.gaptune import FinetuneGapTune
from src.finetune.monitoring import resolve_finetune_monitor_spec
from src.finetune.prompts.gaptune import ObservationTypePrompt, pool_source_context, pool_target_contexts
from src.model import build_encoder_from_cfg

IN_DIM, NUM_NODES, NUM_CLASSES = 6, 40, 3


def _cfg():
    cfg = CN()
    set_cfg(cfg)
    cfg.seed = 0
    cfg.model.in_dim, cfg.model.hidden_dim, cfg.model.out_dim, cfg.model.num_layers = IN_DIM, 8, 8, 2
    ds = cfg.finetune.dataset
    ds.task_level, ds.induced, ds.task_type = "node", False, "classification"
    ds.num_classes, ds.label_dim = NUM_CLASSES, 1
    cfg.analysis.controlled_shift.strengths = [0.0, 0.3]
    cfg.analysis.controlled_shift.updates = 15
    cfg.analysis.node_budget, cfg.analysis.message_budget = 16, 32
    return cfg


def _setup():
    cfg = _cfg()
    torch.manual_seed(0)
    model = build_encoder_from_cfg(cfg, IN_DIM).eval().requires_grad_(False)
    g = torch.Generator().manual_seed(0)
    ei = torch.randint(NUM_NODES, (2, 80), generator=g)
    train = torch.zeros(NUM_NODES, dtype=torch.bool)
    train[: 2 * NUM_CLASSES] = True  # two support nodes per class, no validation
    data = Data(
        x=torch.randn(NUM_NODES, IN_DIM, generator=g),
        edge_index=torch.cat([ei, ei.flip(0)], dim=1),
        y=torch.arange(NUM_NODES) % NUM_CLASSES,
        train_mask=train,
        val_mask=torch.zeros_like(train),
        test_mask=~train,
    )
    return cfg, model, data


def _monitor(cfg):
    return resolve_finetune_monitor_spec(
        cfg, task_level="node", label_dim=1, few_shot_without_validation=True, task_cls=FinetuneGapTune
    )


def _reference(model, graph):
    with torch.no_grad():
        return get_gaptune_driver(model).forward(model, graph, collect=True)


@torch.no_grad()
def _test_repr(task, model, graph, reference):
    task.eval()
    return task.encode(model, graph, reference=reference)


def test_zero_shift_gap_arm_follows_the_head_only_trajectory():
    cfg, model, data = _setup()
    reference = _reference(model, data)
    source, monitor = source_bank(reference), _monitor(cfg)
    head, head_update = fit_arm(cfg, "head", model, data, reference, source, monitor)
    gap, gap_update = fit_arm(cfg, "gap", model, data, reference, source, monitor)

    for key, prompt in gap.prompt.types.items():
        z = reference.obs.h0 if key == "N" else reference.obs.layers[int(key[1:]) - 1].messages
        graph_id = torch.zeros(z.size(0), dtype=torch.long)
        for use_retained in (False, True):
            assert torch.count_nonzero(prompt.values(z, graph_id, 1, use_retained=use_retained)) == 0
        # a nonzero gap, however small, would give the gates a gradient that Adam turns into a step
        assert torch.count_nonzero(prompt.gates) == 0
    assert gap_update == head_update
    assert torch.equal(gap.head.weight, head.head.weight) and torch.equal(gap.head.bias, head.head.bias)
    assert torch.equal(_test_repr(gap, model, data, reference), reference.node_repr)
    assert evaluate(gap, model, data, reference, "test_mask") == evaluate(head, model, data, reference, "test_mask")


def test_target_arm_excludes_source_observations():
    cfg, model, data = _setup()
    view = controlled_views(data.x, data.edge_index, [0.3], generator=torch.Generator().manual_seed(0))[("joint", 0.3)]
    split = {k: data[k] for k in ("y", "train_mask", "val_mask", "test_mask")}
    graph = Data(x=view.x, edge_index=view.edge_index, **split)
    reference, monitor = _reference(model, graph), _monitor(cfg)
    source = source_bank(_reference(model, data))
    other = {key: 3.0 * bank + 1.0 for key, bank in source.items()}

    a, update_a = fit_arm(cfg, "target", model, graph, reference, source, monitor)
    b, update_b = fit_arm(cfg, "target", model, graph, reference, other, monitor)
    assert update_a == update_b
    params_b = dict(b.named_parameters())
    assert all(torch.equal(p, params_b[name]) for name, p in a.named_parameters())
    assert torch.equal(_test_repr(a, model, graph, reference), _test_repr(b, model, graph, reference))

    gap_a, _ = fit_arm(cfg, "gap", model, graph, reference, source, monitor)
    gap_b, _ = fit_arm(cfg, "gap", model, graph, reference, other, monitor)
    assert not torch.equal(gap_a.head.weight, gap_b.head.weight)
    assert any(torch.count_nonzero(p.gates) > 0 for p in gap_a.prompt.types.values())
    # evaluation reads the source contexts pooled with the selected queries
    for p in gap_a.prompt.types.values():
        assert torch.equal(p.retained_source_context, p.source_context())


def test_repetition_rows_and_paired_effects(monkeypatch):
    cfg, model, data = _setup()
    cfg.analysis.controlled_shift.updates = 4
    control = controlled_views(data.x, data.edge_index, [], generator=torch.Generator())[CONTROL]
    control_bank = source_bank(_reference(model, Data(x=control.x, edge_index=control.edge_index)))
    banks = []

    def fit_arm_recording_source(run_cfg, arm, encoder, graph, reference, source, monitor):
        banks.append(source)
        return fit_arm(run_cfg, arm, encoder, graph, reference, source, monitor)

    monkeypatch.setattr(controlled_shift, "fit_arm", fit_arm_recording_source)
    rows = run_repetition(cfg, model, data, 0) + run_repetition(cfg, model, data, 1)
    conditions = [CONTROL] + [(family, 0.3) for family in FAMILIES]
    assert [(r["family"], r["alpha"]) for r in rows] == 2 * conditions
    # every arm of every view reads the unperturbed graph's observations as its source bank
    assert len(banks) == len(rows) * len(ARMS)
    for bank in banks:
        assert bank.keys() == control_bank.keys() and all(torch.equal(bank[k], control_bank[k]) for k in bank)
    for r in rows:
        assert all(0.0 <= r[f"acc_{arm}"] <= 100.0 for arm in ARMS)
        if (r["family"], r["alpha"]) == CONTROL:
            assert r["delta_H"] == r["delta_M"] == 0.0 and r["eta"] == 0.0
            assert r["acc_gap"] == r["acc_head"] and r["update_gap"] == r["update_head"]

    effects = summarize(rows)
    assert [(e["family"], e["alpha"]) for e in effects] == conditions
    for e in effects:
        cells = [r for r in rows if (r["family"], r["alpha"]) == (e["family"], e["alpha"])]
        assert e["n"] == 2
        for v in COMPARATORS:
            diffs = [c["acc_gap"] - c[f"acc_{v}"] for c in cells]
            assert e[f"gap_{v}_mean"] == pytest.approx(np.mean(diffs))
            assert e[f"gap_{v}_low"] <= e[f"gap_{v}_mean"] <= e[f"gap_{v}_high"]


def test_single_graph_pooling_and_mixing_match_the_per_graph_path():
    g = torch.Generator().manual_seed(0)
    z, queries = torch.randn(30, 5, generator=g), torch.randn(4, 5, generator=g)
    descriptors = torch.randn(30, 7, generator=g)
    graph_id = torch.zeros(30, dtype=torch.long)
    single = pool_target_contexts(queries, z, graph_id, 1, 0.5, 1e-6)
    assert torch.equal(single[0], pool_source_context(queries, z, 0.5, 1e-6))
    # two graphs (the second empty) take the per-graph segment path
    torch.testing.assert_close(single[0], pool_target_contexts(queries, z, graph_id, 2, 0.5, 1e-6)[0])

    prompt = ObservationTypePrompt(
        5, 7, num_queries=4, tau_c=0.5, tau_p=0.5, obs_eps=1e-6,
        value_mode="gap", query_mode="shared", mixture="local", generator=g,
    )
    prompt.set_source_bank(torch.randn(12, 5, generator=g))
    with torch.no_grad():
        prompt.gates.copy_(torch.randn(4, generator=g))
    torch.testing.assert_close(
        prompt(z, descriptors, graph_id, 1, use_retained=False),
        prompt(z, descriptors, graph_id, 2, use_retained=False),
    )


def test_cli_registration_and_explicit_checkpoint_rule(capsys):
    assert analysis_run.STUDIES["controlled_shift"] is run_controlled_shift
    cfg = _cfg()
    cfg.analysis.pretrained_checkpoint = "checkpoint.pt"
    assert run_controlled_shift(cfg) == 1  # photo and chameleon each need their own checkpoint
    assert "single dataset" in capsys.readouterr().out
