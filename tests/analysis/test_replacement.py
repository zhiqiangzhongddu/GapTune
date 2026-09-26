"""App. C.6 fixed-predictor source-context replacement on a tiny synthetic GapTune task.

- the identity replacement gives exactly zero context / prompt / output
  changes and 100 % agreement;
- perturbed contexts obey ``0 < delta_P,q <= eps_q`` (Prop. 3.3) and
  ``|dAcc| <= 100 - Agree``, and the predictor is left unchanged;
- proxy collections are pooled with the frozen queries without touching the
  predictor, and pooling the predictor's own bank reproduces ``C*_s``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch_geometric.data import Batch

from src.analysis.replacement import IDENTITY, _rows, _summary, proxy_contexts, replacement_measures
from tests.finetune.test_gaptune import _cfg, _graphs, _set_gates, _task

CPU = torch.device("cpu")


def _toy():
    cfg = _cfg("gcn")
    task, encoder = _task(cfg)
    _set_gates(task)
    task.eval()
    loader = [Batch.from_data_list(_graphs(seed=seed)) for seed in (0, 1, 2)]
    return cfg, task, encoder, loader


def _retained(task) -> dict:
    return {key: p.retained_source_context.clone() for key, p in task.prompt.types.items()}


def test_identity_replacement_is_exact():
    _, task, encoder, loader = _toy()
    m = replacement_measures(task, encoder, loader, CPU, {IDENTITY: _retained(task)})[IDENTITY]
    assert set(m["eps"]) == {"N", "M1", "M2"}
    assert all(value == 0.0 for value in m["eps"].values())
    assert all(value == 0.0 for value in m["delta_p"].values())
    assert m["ef_mean"] == m["ef_max"] == 0.0
    assert m["agreement"] == 100.0 and m["delta_acc"] == 0.0
    (row,) = _rows(42, {IDENTITY: m})
    assert row["contexts"] == IDENTITY and row["budget"] == "-" and row["ec"] == 0.0 and row["checks_ok"]


def test_perturbed_contexts_obey_the_prompt_and_accuracy_bounds():
    _, task, encoder, loader = _toy()
    original = _retained(task)
    generator = torch.Generator().manual_seed(5)
    replacements = {IDENTITY: _retained(task)}
    for scale in (0.1, 2.0):
        replacements[f"random_B{int(10 * scale)}"] = {
            key: value + scale * torch.randn(value.shape, generator=generator) for key, value in original.items()
        }
    measures = replacement_measures(task, encoder, loader, CPU, replacements)
    for name in ("random_B1", "random_B20"):
        m = measures[name]
        for key, eps in m["eps"].items():
            assert 0.0 < m["delta_p"][key] <= eps + 1e-6
        assert m["prompt_bound_ok"] and m["accuracy_bound_ok"]
        assert m["ef_max"] > 0.0 and abs(m["delta_acc"]) <= 100.0 - m["agreement"]
    # The fixed predictor keeps its retained contexts.
    for key, value in _retained(task).items():
        assert torch.equal(value, original[key])
    rows = _rows(42, measures) + _rows(0, measures)
    summary = {(entry["contexts"], entry["budget"]): entry for entry in _summary(rows)}
    assert summary[(IDENTITY, "-")]["delta_acc_low"] == summary[(IDENTITY, "-")]["delta_acc_high"] == 0.0
    assert summary[("random", "20")]["repetitions"] == 2 and summary[("random", "20")]["checks_ok"]


def test_proxy_contexts_are_pooled_with_the_frozen_queries():
    cfg, task, encoder, _ = _toy()
    for key, p in task.prompt.types.items():  # pooling the own bank reproduces C*_s
        torch.testing.assert_close(p.source_context(p.source_bank), p.retained_source_context, atol=0, rtol=0)
    cfg.pretrain.method = "edge_pred"
    proxy_cfg = cfg.finetune.gaptune.proxy.clone()
    for key, value in dict(mode="inverted", num_graphs=2, num_nodes=6, updates=3, final_edges=5, edge_hidden=16,
                           edgepred_pos_pairs=2, edgepred_neg_pairs=3).items():
        setattr(proxy_cfg, key, value)
    runner = SimpleNamespace(task=task, model=encoder, cfg=cfg, device=CPU, pretrain_cfg={})
    before = {name: value.clone() for name, value in task.state_dict().items()}
    contexts, metadata = proxy_contexts(runner, proxy_cfg, {})
    assert metadata["proxy"]["mode"] == "inverted" and metadata["proxy"]["num_graphs"] == 2
    assert metadata["bank_sizes"]["N"] == 12
    for key, value in contexts.items():
        assert value.shape == before[f"prompt.types.{key}.retained_source_context"].shape
        assert not torch.equal(value, before[f"prompt.types.{key}.retained_source_context"])
    for name, value in task.state_dict().items():
        assert torch.equal(value, before[name]), name
