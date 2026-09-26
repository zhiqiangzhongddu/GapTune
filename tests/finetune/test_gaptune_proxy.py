"""GapTune source-free proxy construction tests (paper App. B.8, D.2, Table 18).

Tiny configs (B=2 proxies, n=6 nodes, 3 updates) with random encoder weights:
- EdgePred (gcn; dot-product and retained MLP scorer) and GraphCL (gin)
  inversions give symmetric, loop-free graphs with the requested edge
  counts and features inside the radius ball; the encoder stays untouched;
- EdgePred conditioning pairs are fixed, masked out of every propagation
  adjacency (which keeps all off-diagonal pairs) and resolved by the final
  rule (positives kept, negatives excluded);
- random proxies share the initialization and take no optimizer step;
- determinism per seed; missing retained components are rejected;
- the runner hook fills the GapTune source bank from the proxy graphs.
"""

from __future__ import annotations

import json
import math

import pytest
import torch
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.encoders.gaptune import get_gaptune_driver
from src.finetune.encoders.gaptune.gnn import GCNDriver
from src.finetune.methods.gaptune import FinetuneGapTune
from src.finetune.methods.gaptune_proxy import _edgepred_templates, build_proxy_source_graphs
from src.model.encoder import build_encoder_from_cfg
from src.pretrain.methods import EdgePrediction, GraphCL
from src.utils.checkpoint import cfg_to_dict

IN_DIM = 6
NUM_GRAPHS, NUM_NODES, FINAL_EDGES = 2, 6, 5
# Pretrained payload of an AnyGraphAnyExpert checkpoint: no extra.pretrain_task_state.
LEGACY_PAYLOAD = {"cfg": {"seed": 42}}


def _cfg(model: str = "gcn", pretext: str = "edge_pred", seed: int = 3, **proxy):
    cfg = CN()
    set_cfg(cfg)
    cfg.seed = seed
    cfg.model.name = model
    cfg.model.num_layers = 2
    cfg.model.in_dim = IN_DIM
    cfg.model.hidden_dim = 8
    cfg.model.out_dim = 5
    cfg.pretrain.method = pretext
    cfg.finetune.gaptune.plus = False
    tiny = dict(
        num_graphs=NUM_GRAPHS,
        num_nodes=NUM_NODES,
        updates=3,
        final_edges=FINAL_EDGES,
        edge_hidden=16,
        edgepred_pos_pairs=2,
        edgepred_neg_pairs=3,
    )
    tiny.update(proxy)
    for key, value in tiny.items():
        setattr(cfg.finetune.gaptune.proxy, key, value)
    return cfg


def _encoder(cfg):
    torch.manual_seed(0)
    encoder = build_encoder_from_cfg(cfg, IN_DIM).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def _payload(cfg):
    """A checkpoint payload written by this repo's pretrain runner."""
    torch.manual_seed(1)
    task = GraphCL(cfg) if cfg.pretrain.method == "graphcl" else EdgePrediction(cfg)
    return {"cfg": {"seed": 42}, "extra": {"pretrain_task_state": task.state_dict()}}


def _build(cfg, *, encoder=None, driver=None, payload=None):
    encoder = _encoder(cfg) if encoder is None else encoder
    payload = _payload(cfg) if payload is None else payload
    return build_proxy_source_graphs(
        cfg=cfg,
        model=encoder,
        driver=get_gaptune_driver(encoder) if driver is None else driver,
        pretrain_cfg=cfg_to_dict(payload["cfg"]),
        pretrain_extra=payload.get("extra") or {},
        device=torch.device("cpu"),
    )


def _recording(base):
    calls = []

    class Recording(base):
        @classmethod
        def forward(cls, model, data, **kwargs):
            calls.append((data.x.detach().clone(), data.edge_index.clone(), kwargs["edge_weight"].detach().clone()))
            return super().forward(model, data, **kwargs)

    return Recording, calls


def _pairs(edge_index) -> set:
    return set(map(tuple, edge_index.t().tolist()))


@pytest.mark.parametrize(
    "model, pretext, mlp_scorer",
    [("gcn", "edge_pred", False), ("gcn", "edge_pred", True), ("gin", "graphcl", False)],
)
def test_inversion_produces_valid_proxy_graphs(model, pretext, mlp_scorer):
    cfg = _cfg(model, pretext, feature_radius=1.0)  # small radius so the projection is active
    cfg.pretrain.edge_pred.use_mlp_scorer = mlp_scorer
    encoder = _encoder(cfg)
    before = {key: value.clone() for key, value in encoder.state_dict().items()}
    graphs, meta = _build(cfg, encoder=encoder)

    assert len(graphs) == NUM_GRAPHS
    for graph in graphs:
        assert graph.x.shape == (NUM_NODES, IN_DIM)
        norms = graph.x.norm(dim=1)
        assert norms.max() <= 1.0 + 1e-5 and torch.isclose(norms, torch.ones_like(norms)).any()
        pairs = _pairs(graph.edge_index)
        assert len(pairs) == graph.edge_index.size(1) == 2 * FINAL_EDGES
        assert all((v, u) in pairs and u != v for u, v in pairs)
    assert meta["edges_per_graph"] == [FINAL_EDGES] * NUM_GRAPHS
    assert (meta["mode"], meta["pretext"], meta["updates"]) == ("inverted", pretext, 3)
    # tau_A(t) = tau_start * (tau_end / tau_start) ** (t / (updates - 1)); logged at t = 0 and the last update.
    assert [record["tau"] for record in meta["losses"]] == pytest.approx([1.0, 0.1])
    assert all(math.isfinite(record[key]) for record in meta["losses"] for key in ("loss", "pretext", "r_x", "r_a"))
    json.dumps(meta)
    assert all(p.grad is None for p in encoder.parameters())
    assert all(torch.equal(value, encoder.state_dict()[key]) for key, value in before.items())


def test_edgepred_conditioning_pairs_are_fixed_masked_and_resolved():
    cfg = _cfg("gcn", "edge_pred")
    driver, calls = _recording(GCNDriver)
    graphs, _ = _build(cfg, driver=driver)
    row, col = torch.triu_indices(NUM_NODES, NUM_NODES, offset=1)
    positive, negative = _edgepred_templates(
        NUM_GRAPHS, row.numel(), cfg.finetune.gaptune.proxy, torch.Generator().manual_seed(cfg.seed + 2)
    )
    assert len(calls) == 3
    for _x, edge_index, weight in calls:
        # Every off-diagonal pair stays in the backward pass; the forward is hard (0/1).
        assert edge_index.size(1) == NUM_GRAPHS * NUM_NODES * (NUM_NODES - 1)
        assert torch.all((weight.abs() < 1e-6) | ((weight - 1).abs() < 1e-6))
        weights = dict(zip(map(tuple, edge_index.t().tolist()), weight.tolist()))
        for b in range(NUM_GRAPHS):
            for p in torch.cat([positive[b], negative[b]]).tolist():
                u, v = int(row[p]) + b * NUM_NODES, int(col[p]) + b * NUM_NODES
                assert weights[(u, v)] == 0.0 and weights[(v, u)] == 0.0
    for b, graph in enumerate(graphs):
        pairs = _pairs(graph.edge_index)
        assert all((int(row[p]), int(col[p])) in pairs for p in positive[b].tolist())
        assert not any((int(row[p]), int(col[p])) in pairs for p in negative[b].tolist())


def test_random_mode_shares_the_initialization_and_takes_no_step(monkeypatch):
    cfg = _cfg("gcn", "edge_pred")
    driver, calls = _recording(GCNDriver)
    inverted, _ = _build(cfg, driver=driver)
    initial_features = calls[0][0].view(NUM_GRAPHS, NUM_NODES, IN_DIM)

    def no_step(*args, **kwargs):
        raise AssertionError("random proxies must not be optimized")

    monkeypatch.setattr(torch.optim.Adam, "step", no_step)
    cfg.finetune.gaptune.proxy.mode = "random"
    driver, calls = _recording(GCNDriver)
    graphs, meta = _build(cfg, driver=driver, payload=LEGACY_PAYLOAD)
    assert calls == [] and meta["updates"] == 0 and meta["losses"] == []
    assert torch.equal(torch.stack([graph.x for graph in graphs]), initial_features)
    assert not torch.equal(graphs[0].x, inverted[0].x)
    assert meta["edges_per_graph"] == [FINAL_EDGES] * NUM_GRAPHS


@pytest.mark.parametrize("model, pretext", [("gcn", "edge_pred"), ("gin", "graphcl")])
def test_proxies_are_deterministic_per_seed(model, pretext):
    first, first_meta = _build(_cfg(model, pretext))
    again, again_meta = _build(_cfg(model, pretext))
    other, _ = _build(_cfg(model, pretext, seed=4))
    for a, b in zip(first, again):
        assert torch.equal(a.x, b.x) and torch.equal(a.edge_index, b.edge_index)
    assert first_meta == again_meta
    assert not torch.equal(first[0].x, other[0].x)


def test_missing_pretrained_components_and_other_checkpoints_are_rejected():
    graphcl = _cfg("gin", "graphcl")
    with pytest.raises(ValueError, match="GraphCL projection.*re-pretrain the checkpoint with this repo"):
        _build(graphcl, payload=LEGACY_PAYLOAD)
    graphcl.finetune.gaptune.proxy.mode = "random"  # random proxies use no pretrained component
    assert len(_build(graphcl, payload=LEGACY_PAYLOAD)[0]) == NUM_GRAPHS

    mlp_scorer = _cfg("gcn", "edge_pred")
    mlp_scorer.pretrain.edge_pred.use_mlp_scorer = True
    with pytest.raises(ValueError, match="MLP edge scorer"):
        _build(mlp_scorer, payload=LEGACY_PAYLOAD)
    # The original dot-product scorer needs no retained state.
    assert len(_build(_cfg("gcn", "edge_pred"), payload=LEGACY_PAYLOAD)[0]) == NUM_GRAPHS

    with pytest.raises(ValueError, match="EdgePred or GraphCL"):
        _build(_cfg("gcn", "dgi"), payload=LEGACY_PAYLOAD)
    with pytest.raises(ValueError, match="edgepred_pos_pairs"):
        _build(_cfg("gcn", "edge_pred", edgepred_neg_pairs=11))


def test_pretext_settings_follow_the_checkpoint_cfg():
    # An explicit finetune.pretrained_checkpoint need not match cfg.pretrain.
    payload = {**_payload(_cfg("gin", "graphcl")), "cfg": {"seed": 42, "pretrain": {"method": "graphcl"}}}
    assert _build(_cfg("gin", "edge_pred"), payload=payload)[1]["pretext"] == "graphcl"
    mlp_payload = {**LEGACY_PAYLOAD, "cfg": {"pretrain": {"method": "edge_pred", "edge_pred": {"use_mlp_scorer": True}}}}
    with pytest.raises(ValueError, match="MLP edge scorer"):
        _build(_cfg("gcn", "edge_pred"), payload=mlp_payload)


def test_prepare_with_encoder_fills_the_bank_from_proxy_graphs():
    cfg = _cfg("gcn", "edge_pred")
    ds = cfg.finetune.dataset
    ds.name, ds.task_level, ds.task_level_raw, ds.task_level_effective = "toy", "node", "node", "graph"
    ds.induced, ds.task_type, ds.num_classes, ds.label_dim = True, "classification", 3, 1
    encoder = _encoder(cfg)
    task = FinetuneGapTune(cfg)
    task.validate_encoder(encoder)
    payload = _payload(cfg)
    task.prepare_with_encoder(
        model=encoder,
        device=torch.device("cpu"),
        pretrain_cfg=cfg_to_dict(payload["cfg"]),
        pretrain_extra=payload.get("extra") or {},
    )
    meta = task.initialization_metadata
    assert meta["source"] == "proxy" and meta["graphs_total"] == NUM_GRAPHS
    assert meta["proxy"]["pretext"] == "edge_pred" and meta["proxy"]["updates"] == 3
    # GCN messages include one native self-loop per node.
    per_graph_messages = 2 * FINAL_EDGES + NUM_NODES
    assert meta["bank_sizes"] == {"N": NUM_GRAPHS * NUM_NODES, "M1": NUM_GRAPHS * per_graph_messages, "M2": NUM_GRAPHS * per_graph_messages}
    json.dumps(meta)
    for prompt in task.prompt.types.values():
        assert not bool(prompt.source_empty)
