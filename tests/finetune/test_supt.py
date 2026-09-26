"""SUPT (subgraph-level universal prompt tuning) baseline tests.

- ``num_bases=1`` soft SUPT reduces to GPF (``x + b``).
- Soft mixture weights are a row-wise softmax over the bases (rows sum to 1).
- Hard SUPT adds each basis to the per-graph top-``ceil(r N_g)`` nodes and
  divides by ``1 + count`` (checked against a per-graph reference loop).
- The GCN scorer sees exactly the edge_index the frozen encoder sees, i.e.
  induced edge subgraphs without the query edge.
- The task trains prompt + head only, keeps induced node subgraphs, and its
  variant tag names every non-default option.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import softmax as graph_softmax
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.methods.supt import FinetuneSUPT
from src.finetune.prompts.gpf import GPFPrompt
from src.finetune.prompts.supt import SUPTPrompt
from src.model.encoder import build_encoder_from_cfg

D = 6


def _graph(num_nodes: int, num_edges: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, D, generator=g)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), generator=g)
    return x, edge_index


def _two_graph_batch():
    x1, e1 = _graph(5, 8, seed=1)
    x2, e2 = _graph(3, 4, seed=2)
    x = torch.cat([x1, x2])
    edge_index = torch.cat([e1, e2 + 5], dim=1)
    batch = torch.tensor([0] * 5 + [1] * 3)
    return x, edge_index, batch


def test_soft_single_basis_reduces_to_gpf():
    torch.manual_seed(0)
    x, edge_index = _graph(7, 12, seed=0)
    supt = SUPTPrompt(D, num_bases=1, variant="soft")
    gpf = GPFPrompt(D)
    with torch.no_grad():
        gpf.global_emb.copy_(supt.bases)
    batch = torch.zeros(7, dtype=torch.long)
    assert torch.allclose(supt.add(x, edge_index, batch), gpf.add(x), atol=1e-6)


def test_soft_rows_sum_to_one():
    torch.manual_seed(0)
    x, edge_index = _graph(7, 12, seed=0)
    prompt = SUPTPrompt(D, num_bases=4, variant="soft")
    out = prompt.add(x, edge_index, torch.zeros(7, dtype=torch.long))

    weights = torch.softmax(prompt.scorer(x + prompt.bases.sum(dim=0), edge_index), dim=1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(7), atol=1e-6)
    assert torch.allclose(out - x, weights @ prompt.bases, atol=1e-6)

    # Identical bases: any convex mixture returns that basis.
    with torch.no_grad():
        prompt.bases.copy_(prompt.bases[:1].expand_as(prompt.bases))
    out = prompt.add(x, edge_index, torch.zeros(7, dtype=torch.long))
    assert torch.allclose(out, x + prompt.bases[0], atol=1e-6)


@pytest.mark.parametrize("hard_score", ["tanh", "graph_softmax"])
def test_hard_topk_per_graph_and_one_plus_count(hard_score):
    torch.manual_seed(0)
    x, edge_index, batch = _two_graph_batch()
    ratio, k = 0.4, 3
    prompt = SUPTPrompt(D, num_bases=k, variant="hard", ratio=ratio, hard_score=hard_score)
    out = prompt.add(x, edge_index, batch)

    raw = prompt.scorer(x + prompt.bases.sum(dim=0), edge_index)
    score = torch.tanh(raw) if hard_score == "tanh" else graph_softmax(raw, batch)
    expected = x.clone()
    for g in range(2):
        nodes = (batch == g).nonzero().view(-1)
        top = math.ceil(ratio * nodes.numel())
        total = torch.zeros(nodes.numel(), D)
        count = torch.ones(nodes.numel())
        for j in range(k):
            chosen = torch.argsort(score[nodes, j], descending=True)[:top]
            assert chosen.numel() == top  # ceil(r N_g) nodes of THIS graph
            total[chosen] += score[nodes[chosen], j].unsqueeze(1) * prompt.bases[j]
            count[chosen] += 1
        expected[nodes] = x[nodes] + total / count.unsqueeze(1)
    assert torch.allclose(out, expected, atol=1e-6)


def _cfg(task_level: str, *, num_classes: int = 3, **supt) -> CN:
    cfg = set_cfg(CN())
    cfg.finetune.method = "supt"
    cfg.finetune.dataset.name = "toy"
    cfg.finetune.dataset.task_level = task_level
    cfg.finetune.dataset.induced = True
    cfg.finetune.dataset.task_type = "classification"
    cfg.finetune.dataset.label_dim = 1
    cfg.finetune.dataset.num_classes = num_classes
    cfg.model.name = "gcn"
    cfg.model.in_dim = D
    cfg.model.hidden_dim = 8
    cfg.model.out_dim = 8
    cfg.model.num_layers = 2
    for key, value in supt.items():
        setattr(cfg.finetune.supt, key, value)
    return cfg


def _frozen_encoder(cfg):
    model = build_encoder_from_cfg(cfg, in_dim=D)
    for param in model.parameters():
        param.requires_grad = False
    return model


def _edge_subgraph_batch() -> Batch:
    graphs = []
    for seed, n in ((3, 5), (4, 4)):
        x, edge_index = _graph(n, 6, seed=seed)
        keep = ~(((edge_index[0] == 0) & (edge_index[1] == 1)) | ((edge_index[0] == 1) & (edge_index[1] == 0)))
        graphs.append(
            Data(
                x=x,
                edge_index=edge_index[:, keep],  # query edge (0, 1) removed, as the loader does
                edge_label_index=torch.tensor([[0], [1]]),
                y=torch.tensor([seed % 2]),
            )
        )
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize("variant", ["soft", "hard"])
def test_scorer_sees_encoder_edge_index(variant):
    torch.manual_seed(0)
    cfg = _cfg("edge", num_classes=2, variant=variant)
    FinetuneSUPT.validate_cfg(cfg)
    task = FinetuneSUPT(cfg)
    model = _frozen_encoder(cfg)
    seen = {}
    task.prompt.scorer.register_forward_pre_hook(lambda _m, args: seen.__setitem__("scorer", args[1].clone()))
    model.register_forward_pre_hook(lambda _m, args: seen.__setitem__("encoder", args[0].edge_index.clone()))

    data = _edge_subgraph_batch()
    task.evaluate_split(model, [data], torch.device("cpu"), prefix="test", mask_attr="test_mask")

    assert torch.equal(seen["scorer"], seen["encoder"])
    pairs = set(map(tuple, seen["scorer"].t().tolist()))
    for u, v in data.edge_label_index.t().tolist():
        assert (u, v) not in pairs and (v, u) not in pairs


def test_task_trains_prompt_and_head_only():
    torch.manual_seed(0)
    cfg = _cfg("node", variant="hard")
    task = FinetuneSUPT(cfg)
    model = _frozen_encoder(cfg)
    encoder_before = [p.detach().clone() for p in model.parameters()]
    bases_before = task.prompt.bases.detach().clone()

    graphs = []
    for seed in range(4):
        x, edge_index = _graph(6, 10, seed=10 + seed)
        graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([seed % 3])))
    loader = [Batch.from_data_list(graphs)]

    optimizers = task.build_optimizers(model)
    trained = {id(p) for group in optimizers["primary"].param_groups for p in group["params"]}
    assert trained == {id(p) for p in task.parameters_to_optimize()}
    assert not any(id(p) in trained for p in model.parameters())

    model.train()
    loss, logs = task.train_epoch(model, loader, torch.device("cpu"), optimizers)
    assert math.isfinite(loss) and "train_acc" in logs
    assert not torch.equal(task.prompt.bases, bases_before)
    assert all(torch.equal(a, b) for a, b in zip(encoder_before, model.parameters()))


def test_keeps_induced_node_subgraphs():
    cfg = _cfg("node")
    params = {"task_level": "node", "induced": True}
    assert FinetuneSUPT.adjust_dataset_cfg(cfg, dict(params)) == params
    assert FinetuneSUPT.resolve_frozen_encoder_mode(cfg) == "train_bn_eval"
    assert FinetuneSUPT.supports_early_stopping is False


def test_variant_tag_names_non_default_options():
    assert FinetuneSUPT.variant_tag(_cfg("graph")) == ""
    assert FinetuneSUPT.variant_tag(_cfg("graph", ratio=0.2)) == ""  # ratio unused by soft
    assert FinetuneSUPT.variant_tag(_cfg("graph", variant="hard")) == "hard"
    assert FinetuneSUPT.variant_tag(_cfg("graph", variant="hard", ratio=0.2, num_bases=3)) == "hard-k3-r0.2"
    assert FinetuneSUPT.variant_tag(_cfg("graph", variant="hard", hard_score="graph_softmax")) == (
        "hard-score_graph_softmax"
    )
    assert FinetuneSUPT.variant_tag(_cfg("graph", orth_loss=True, gcn_bias=False)) == "orth-nogcnbias"
    assert FinetuneSUPT.variant_tag(_cfg("graph", head_layers=2, head_dropout=0.1)) == "head2-hdo0.1"
    assert FinetuneSUPT.variant_tag(_cfg("graph", lr=1e-2)) == "mlr0.01"


@pytest.mark.parametrize(
    "override",
    [{"variant": "mixed"}, {"num_bases": 0}, {"ratio": 1.0}, {"hard_score": "sigmoid"}, {"head_layers": 0}],
)
def test_validate_cfg_rejects_bad_options(override):
    with pytest.raises(ValueError):
        FinetuneSUPT.validate_cfg(_cfg("graph", **override))
