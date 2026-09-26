"""MTG (message tuning) baseline tests.

- zero-prototype parity: with ``M = 0`` the driver reproduces the vanilla
  encoder for all 7 backbones (NodeFormer: given the same fixed projections),
  on a batch with a pre-existing self-loop and on a batch-free graph;
- injection points: one fusion per layer, the last included, reading the
  state that layer consumes with the EdgePrompt ``dim_list`` widths;
  FAGCN's ``eps * x0`` reference stays unprompted;
- active path: default-initialized prototypes change the output and every
  prototype/router receives a gradient; the frozen encoder receives none;
- census ``sum_l (2 m d_l + m)`` plus the head;
- the task trains prototypes + head only, NodeFormer projections come from
  ``cfg.seed``, and cfg validation / variant tags.
"""

from __future__ import annotations

import math
from unittest import mock

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.encoders.edgeprompt import resolve_edgeprompt_prompt_spec
from src.finetune.encoders.mtg import forward_with_mtg
from src.finetune.encoders.nodeformer_fixed import build_fixed_projections
from src.finetune.methods.mtg import FinetuneMTG
from src.finetune.prompts.mtg import MessagePrototypeFusion, MTGPrompt
from src.model.encoder import build_encoder_from_cfg

IN_DIM = 6
MODELS = ("gcn", "gin", "gat", "transformer", "fagcn", "h2gcn", "nodeformer")


def _cfg(name: str = "gcn", use_batchnorm: bool = False, task_level: str = "graph", **mtg) -> CN:
    cfg = set_cfg(CN())
    cfg.finetune.method = "mtg"
    cfg.finetune.dataset.name = "toy"
    cfg.finetune.dataset.task_level = task_level
    cfg.finetune.dataset.induced = False
    cfg.finetune.dataset.task_type = "classification"
    cfg.finetune.dataset.label_dim = 1
    cfg.finetune.dataset.num_classes = 3
    cfg.model.name = name
    cfg.model.num_layers = 3
    cfg.model.in_dim = IN_DIM
    cfg.model.hidden_dim = 8
    cfg.model.out_dim = 5
    cfg.model.dropout = 0.3  # must be inactive in eval mode
    cfg.model.use_batchnorm = use_batchnorm
    cfg.model.gat.heads = 3
    cfg.model.nodeformer.heads = 2
    cfg.model.nodeformer.num_random_features = 6
    for key, value in mtg.items():
        setattr(cfg.finetune.mtg, key, value)
    return cfg


def _encoder(cfg):
    torch.manual_seed(0)
    encoder = build_encoder_from_cfg(cfg, IN_DIM)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in encoder.parameters():
            p.copy_(0.4 * torch.randn(p.shape, generator=g))
        for name, buf in encoder.named_buffers():
            if name.endswith("running_mean"):
                buf.copy_(0.2 * torch.randn(buf.shape, generator=g))
            elif name.endswith("running_var"):
                buf.copy_(0.5 + torch.rand(buf.shape, generator=g))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def _setup(name: str, use_batchnorm: bool = False):
    cfg = _cfg(name, use_batchnorm)
    encoder = _encoder(cfg)
    torch.manual_seed(1)
    prompt = MTGPrompt(resolve_edgeprompt_prompt_spec(cfg).dim_list, num_prototypes=4)
    projections = build_fixed_projections(encoder, seed=42) if name == "nodeformer" else None
    return cfg, encoder, prompt, projections


def _batch() -> Batch:
    g = torch.Generator().manual_seed(0)
    graphs = []
    for i, (n, e) in enumerate(((5, 8), (4, 6), (6, 9))):
        graphs.append(Data(
            x=torch.randn(n, IN_DIM, generator=g),
            edge_index=torch.randint(0, n, (2, e), generator=g),
            y=torch.tensor([i % 3]),
        ))
    # Guarantee a pre-existing self-loop.
    graphs[1].edge_index = torch.cat([graphs[1].edge_index, torch.tensor([[2], [2]])], dim=1)
    return Batch.from_data_list(graphs)


def _vanilla(encoder, data, projections=None):
    """Vanilla forward; NodeFormer gets the given projections in layer order."""
    if projections is None:
        return encoder(data)
    it = iter(projections)
    with mock.patch("src.model.nodeformer.create_projection_matrix", side_effect=lambda *a, **k: next(it)):
        return encoder(data)


@pytest.mark.parametrize("use_batchnorm", [False, True])
@pytest.mark.parametrize("name", MODELS)
def test_zero_prototype_parity(name, use_batchnorm):
    _, encoder, prompt, proj = _setup(name, use_batchnorm)
    with torch.no_grad():
        for fuse in prompt.fuse:
            fuse.prototypes.zero_()
    batch = _batch()
    single = Data(x=batch.x[:5], edge_index=batch.edge_index[:, :8])
    with torch.no_grad():
        for data in (batch, single):
            ref_node, ref_graph = _vanilla(encoder, data, proj)
            node, graph = forward_with_mtg(encoder, data, prompt.fuse, projections=proj)
            torch.testing.assert_close(node, ref_node, atol=1e-5, rtol=1e-4)
            if ref_graph is None:
                assert graph is None
            else:
                torch.testing.assert_close(graph, ref_graph, atol=1e-5, rtol=1e-4)


class _Record(nn.Module):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def forward(self, h):
        self.sink.append(h)
        return h


@pytest.mark.parametrize("name", MODELS)
def test_one_fusion_per_layer_at_layer_input(name):
    cfg, encoder, _, proj = _setup(name)
    data = _batch()
    seen = []
    with torch.no_grad():
        dims = resolve_edgeprompt_prompt_spec(cfg).dim_list
        forward_with_mtg(encoder, data, [_Record(seen) for _ in dims], projections=proj)
        assert [h.size(-1) for h in seen] == list(dims)
        assert all(h.size(-2) == data.num_nodes for h in seen)
        if name == "fagcn":
            first = encoder.lin_in(data.x)
        elif name == "nodeformer":
            inner = encoder.model
            first = inner.activation(inner.bns[0](inner.fcs[0](data.x.unsqueeze(0))))
        else:
            first = data.x
        torch.testing.assert_close(seen[0], first)
        if name in ("gcn", "gin", "gat"):  # later layers read the cached native layer outputs
            encoder(data)
            for h, cached in zip(seen[1:], encoder.get_layer_node_reprs()):
                torch.testing.assert_close(h, cached)
    with pytest.raises(ValueError):
        forward_with_mtg(encoder, data, [_Record(seen)], projections=proj)


def test_fagcn_initial_reference_is_unprompted():
    _, encoder, prompt, _ = _setup("fagcn")
    data = _batch()
    refs = []
    for conv in encoder.convs:
        conv.register_forward_pre_hook(lambda _m, args: refs.append(args[1]))
    with torch.no_grad():
        forward_with_mtg(encoder, data, prompt.fuse)
        x0 = encoder.lin_in(data.x)
    assert len(refs) == len(encoder.convs)
    for ref in refs:
        torch.testing.assert_close(ref, x0)


@pytest.mark.parametrize("name", MODELS)
def test_active_path_gradients(name):
    _, encoder, prompt, proj = _setup(name)
    data = _batch()
    with torch.no_grad():
        ref_node, _ = _vanilla(encoder, data, proj)
    node, graph = forward_with_mtg(encoder, data, prompt.fuse, projections=proj)
    assert not torch.allclose(node, ref_node, atol=1e-4)
    (node.pow(2).sum() + graph.pow(2).sum()).backward()
    for pname, p in prompt.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, pname
    assert all(p.grad is None for p in encoder.parameters())


def test_parameter_census():
    # Spec example: in 100, hidden 128, L = 2, m = 10 -> 4,580.
    assert sum(p.numel() for p in MTGPrompt([100, 128], 10).parameters()) == 4580
    fusion = MessagePrototypeFusion(4096, 10)
    assert fusion.router.bias is not None
    expected_std = math.sqrt(2.0 / (1 + 0.01**2)) / math.sqrt(4096)
    assert abs(fusion.prototypes.std().item() / expected_std - 1) < 0.05
    for name in MODELS:
        cfg = _cfg(name)
        task = FinetuneMTG(cfg)
        dims = resolve_edgeprompt_prompt_spec(cfg).dim_list
        m = int(cfg.finetune.mtg.num_prototypes)
        census = sum(2 * m * d + m for d in dims)
        assert sum(p.numel() for p in task.prompt.parameters()) == census, name
        head = sum(p.numel() for p in task.supervised_head.classifier.parameters())
        assert sum(p.numel() for p in task.parameters_to_optimize()) == census + head, name


@pytest.mark.parametrize("name", ["gcn", "nodeformer"])
def test_task_trains_prototypes_and_head_only(name):
    torch.manual_seed(0)
    cfg = _cfg(name)
    cfg.seed = 5
    FinetuneMTG.validate_cfg(cfg)
    task = FinetuneMTG(cfg)
    model = _encoder(cfg)
    loader = [_batch()]
    encoder_before = [p.detach().clone() for p in model.parameters()]
    prompt_before = [p.detach().clone() for p in task.prompt.parameters()]

    optimizers = task.build_optimizers(model)
    trained = {id(p) for group in optimizers["primary"].param_groups for p in group["params"]}
    assert trained == {id(p) for p in task.parameters_to_optimize()}

    loss, logs = task.train_epoch(model, loader, torch.device("cpu"), optimizers)
    assert math.isfinite(loss) and "train_acc" in logs
    assert all(not torch.equal(a, b) for a, b in zip(prompt_before, task.prompt.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(encoder_before, model.parameters()))

    metrics = task.evaluate_split(model, loader, torch.device("cpu"), prefix="val", mask_attr="val_mask")
    assert math.isfinite(metrics["val_loss"])
    if name == "nodeformer":
        expected = build_fixed_projections(model, seed=5)
        assert all(torch.equal(a, b) for a, b in zip(task._projections, expected))
        with torch.no_grad():
            first = task.encode(model, loader[0])[0]
            second = task.encode(model, loader[0])[0]
            direct = forward_with_mtg(model, loader[0], task.prompt.fuse, projections=expected)[0]
        assert torch.equal(first, second) and torch.equal(first, direct)


def test_policy_validation_and_variant_tag():
    cfg = _cfg()
    assert FinetuneMTG.resolve_frozen_encoder_mode(cfg) == "eval"
    assert FinetuneMTG.supports_early_stopping is True
    assert FinetuneMTG.requires_frozen_encoder is True
    FinetuneMTG.validate_cfg(_cfg(task_level="node"))
    with pytest.raises(ValueError, match="MTG requires node/graph batches"):
        FinetuneMTG.validate_cfg(_cfg(task_level="edge"))
    with pytest.raises(ValueError, match="unsupported backbone"):
        FinetuneMTG.validate_cfg(_cfg("mlp"))
    with pytest.raises(ValueError, match="num_prototypes"):
        FinetuneMTG.validate_cfg(_cfg(num_prototypes=0))
    with pytest.raises(ValueError, match="no forward driver"):
        forward_with_mtg(build_encoder_from_cfg(_cfg("mlp"), IN_DIM), _batch(), [])
    assert FinetuneMTG.variant_tag(cfg) == ""
    assert FinetuneMTG.variant_tag(_cfg(num_prototypes=5, lr=0.01)) == "m5-mlr0.01"
