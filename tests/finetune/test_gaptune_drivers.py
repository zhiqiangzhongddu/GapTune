"""GapTune forward-driver tests (paper App. B.10 implementation checks).

- zero-prompt parity: ``None`` and all-zero prompts reproduce the vanilla
  encoder (NodeFormer: given the same fixed projections) on a batch of
  graphs that contains a pre-existing self-loop;
- active path: non-zero node/message prompts change the output and receive
  gradients; the frozen encoder receives none;
- observation shapes, per-graph message rows and self-loop conventions;
- post-weighting insertion: one message-prompt row lands unscaled on its
  receiver's aggregate;
- gcn/gin ``edge_weight`` parity (source-free inversion path);
- ``nodeformer_fixed`` parity against the native ``NodeFormerConv``.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.encoders.gaptune import (
    build_gaptune_driver,
    get_gaptune_driver,
    resolve_gaptune_spec,
)
from src.finetune.encoders.nodeformer_fixed import (
    build_fixed_projections,
    nodeformer_conv_aggregate,
    nodeformer_conv_forward,
)
from src.model.encoder import build_encoder_from_cfg
from src.model.nodeformer import NodeFormerConv, create_projection_matrix


ATOL = 1e-5
RTOL = 1e-4
IN_DIM = 6
MODELS = ("gcn", "gin", "gat", "transformer", "fagcn", "h2gcn", "nodeformer")


def _cfg(name: str, use_batchnorm: bool = False, num_layers: int = 3):
    cfg = CN()
    set_cfg(cfg)
    cfg.model.name = name
    cfg.model.num_layers = num_layers
    cfg.model.in_dim = IN_DIM
    cfg.model.hidden_dim = 8
    cfg.model.out_dim = 5
    cfg.model.dropout = 0.3  # must be inactive in eval mode
    cfg.model.use_batchnorm = use_batchnorm
    cfg.model.gat.heads = 3
    cfg.model.nodeformer.heads = 2
    cfg.model.nodeformer.num_random_features = 6
    return cfg


def _randomize(module: torch.nn.Module) -> None:
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(0.4 * torch.randn(p.shape, generator=g))
        for name, buf in module.named_buffers():
            if name.endswith("running_mean"):
                buf.copy_(0.2 * torch.randn(buf.shape, generator=g))
            elif name.endswith("running_var"):
                buf.copy_(0.5 + torch.rand(buf.shape, generator=g))


def _setup(name: str, use_batchnorm: bool = False, num_layers: int = 3):
    cfg = _cfg(name, use_batchnorm, num_layers)
    torch.manual_seed(0)
    encoder = build_encoder_from_cfg(cfg, IN_DIM)
    _randomize(encoder)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    projections = build_fixed_projections(encoder, seed=42) if name == "nodeformer" else None
    return encoder, build_gaptune_driver(cfg), projections


def _batch() -> Batch:
    g = torch.Generator().manual_seed(0)
    graphs = []
    for n, e in ((5, 8), (4, 6), (6, 9)):
        graphs.append(Data(
            x=torch.randn(n, IN_DIM, generator=g),
            edge_index=torch.randint(0, n, (2, e), generator=g),
        ))
    # Guarantee a pre-existing self-loop.
    graphs[1].edge_index = torch.cat([graphs[1].edge_index, torch.tensor([[2], [2]])], dim=1)
    return Batch.from_data_list(graphs)


def _vanilla(encoder, data, projections=None):
    """Vanilla forward; NodeFormer gets the given projections in layer order."""
    if projections is None:
        return encoder(data)
    it = iter(projections)
    with mock.patch(
        "src.model.nodeformer.create_projection_matrix", side_effect=lambda *a, **k: next(it)
    ):
        return encoder(data)


class GapTuneDriverTest(unittest.TestCase):
    def assertClose(self, a, b, msg=""):
        diff = (a - b).abs().max().item()
        self.assertTrue(torch.allclose(a, b, atol=ATOL, rtol=RTOL), f"{msg} max abs diff {diff:.3e}")

    def test_factory_dispatch(self):
        for name in MODELS:
            encoder, driver, _ = _setup(name)
            self.assertTrue(driver.supports_model(encoder), name)
            self.assertIs(get_gaptune_driver(encoder), driver)
        with self.assertRaises(ValueError):
            build_gaptune_driver(_cfg("mlp"))
        with self.assertRaises(ValueError):
            get_gaptune_driver(build_encoder_from_cfg(_cfg("mlp"), IN_DIM))

    def test_zero_prompt_parity(self):
        data = _batch()
        for name in MODELS:
            for use_bn in (False, True):
                with self.subTest(model=name, batchnorm=use_bn):
                    encoder, driver, proj = _setup(name, use_bn)
                    with torch.no_grad():
                        ref_node, ref_graph = _vanilla(encoder, data, proj)
                        out_none = driver.forward(encoder, data, projections=proj)
                        obs = driver.forward(encoder, data, collect=True, projections=proj).obs
                        out_zero = driver.forward(
                            encoder,
                            data,
                            node_prompt=torch.zeros_like(obs.h0),
                            message_prompts=[torch.zeros_like(l.messages) for l in obs.layers],
                            projections=proj,
                        )
                    for out in (out_none, out_zero):
                        self.assertClose(out.node_repr, ref_node, f"{name} node")
                        self.assertClose(out.graph_repr, ref_graph, f"{name} graph")

    def test_observation_shapes_and_self_loops(self):
        data = _batch()
        N, E = data.num_nodes, data.edge_index.size(1)
        n_loops = int((data.edge_index[0] == data.edge_index[1]).sum())
        for name in MODELS:
            with self.subTest(model=name):
                encoder, driver, proj = _setup(name)
                adds_loops = resolve_gaptune_spec(_cfg(name)).adds_self_loops
                with torch.no_grad():
                    out = driver.forward(encoder, data, collect=True, projections=proj)
                obs = out.obs
                dims = driver.observation_dims(encoder)
                desc = driver.descriptor_dims(encoder)
                self.assertEqual(tuple(obs.h0.shape), (N, dims["N"]))
                self.assertTrue(torch.equal(obs.h_final, out.node_repr))
                self.assertEqual(desc["N"], dims["N"] + out.node_repr.size(1))
                self.assertEqual(len(obs.layers), len(dims["M"]))
                if name == "h2gcn":  # loops appended, existing kept
                    rows, loops = E + N, N + n_loops
                elif adds_loops:  # existing loops replaced by one per node
                    rows, loops = E - n_loops + N, N
                else:
                    rows, loops = E, n_loops
                for layer, d_m, e_m in zip(obs.layers, dims["M"], desc["M"]):
                    self.assertEqual(tuple(layer.messages.shape), (rows, d_m))
                    self.assertEqual(tuple(layer.sender.shape), (rows,))
                    self.assertEqual(tuple(layer.receiver.shape), (rows,))
                    self.assertEqual(layer.inputs.size(0), N)
                    self.assertEqual(e_m, 2 * layer.inputs.size(1) + d_m)
                    self.assertEqual(int((layer.sender == layer.receiver).sum()), loops)
                    self.assertTrue(torch.equal(data.batch[layer.sender], data.batch[layer.receiver]))

    def test_cost_study_dims(self):
        cfg = _cfg("gcn")
        cfg.model.in_dim, cfg.model.hidden_dim, cfg.model.out_dim = 100, 128, 128
        encoder = build_encoder_from_cfg(cfg, 100)
        driver = build_gaptune_driver(cfg)
        dims, desc = driver.observation_dims(encoder), driver.descriptor_dims(encoder)
        self.assertEqual(dims, {"N": 100, "M": [128, 128, 128]})
        self.assertEqual(desc, {"N": 228, "M": [328, 384, 384]})
        K = 8  # queries + relevance vectors + gates per type
        census = K * (dims["N"] + desc["N"] + 1) + sum(K * (d + e + 1) for d, e in zip(dims["M"], desc["M"]))
        self.assertEqual(census, 14496)

    def test_active_path_gradients(self):
        data = _batch()
        for name in MODELS:
            with self.subTest(model=name):
                encoder, driver, proj = _setup(name)
                with torch.no_grad():
                    base = driver.forward(encoder, data, collect=True, projections=proj)
                g = torch.Generator().manual_seed(1)
                node_prompt = (0.5 * torch.randn(base.obs.h0.shape, generator=g)).requires_grad_()
                message_prompts = [
                    (0.5 * torch.randn(l.messages.shape, generator=g)).requires_grad_()
                    for l in base.obs.layers
                ]
                for kwargs in ({"node_prompt": node_prompt}, {"message_prompts": message_prompts}):
                    with torch.no_grad():
                        out = driver.forward(encoder, data, projections=proj, **kwargs)
                    self.assertFalse(torch.allclose(out.node_repr, base.node_repr, atol=1e-4))
                out = driver.forward(
                    encoder, data, node_prompt=node_prompt, message_prompts=message_prompts, projections=proj
                )
                out.node_repr.pow(2).sum().backward()
                for p in (node_prompt, *message_prompts):
                    self.assertIsNotNone(p.grad)
                    self.assertGreater(p.grad.abs().sum().item(), 0.0)
                for p in encoder.parameters():
                    self.assertIsNone(p.grad)

    def test_node_prompt_enters_h0_once(self):
        data = _batch()
        for name in MODELS:
            with self.subTest(model=name):
                encoder, driver, proj = _setup(name)
                with torch.no_grad():
                    base = driver.forward(encoder, data, collect=True, projections=proj)
                    p = torch.randn_like(base.obs.h0)
                    out = driver.forward(encoder, data, node_prompt=p, collect=True, projections=proj)
                self.assertClose(out.obs.h0, base.obs.h0 + p)
                self.assertClose(out.obs.layers[0].inputs, base.obs.h0 + p)

    def test_message_prompt_is_added_after_weighting(self):
        # One-layer encoders whose UPD is affine: a single prompt row must
        # reach its receiver unscaled by the structural/attention weight.
        data = _batch()
        for name in ("gcn", "gat", "transformer", "fagcn"):
            with self.subTest(model=name):
                encoder, driver, _ = _setup(name, num_layers=1)
                with torch.no_grad():
                    base = driver.forward(encoder, data, collect=True)
                    layer = base.obs.layers[0]
                    e = int((layer.sender != layer.receiver).nonzero()[0])
                    v = torch.randn(layer.messages.size(1))
                    prompt = torch.zeros_like(layer.messages)
                    prompt[e] = v
                    out = driver.forward(encoder, data, message_prompts=[prompt])
                if name in ("gat", "transformer"):
                    effect = v.view(encoder.convs[0].heads, -1).mean(dim=0)
                elif name == "fagcn":
                    effect = encoder.out_lin.weight @ v
                else:
                    effect = v
                expected = torch.zeros_like(base.node_repr)
                expected[layer.receiver[e]] = effect
                self.assertClose(out.node_repr - base.node_repr, expected)

    def test_edge_weight_gcn_gin(self):
        data = _batch()
        E = data.edge_index.size(1)
        for name in ("gcn", "gin"):
            with self.subTest(model=name):
                encoder, driver, _ = _setup(name)
                ones = torch.ones(E, requires_grad=True)
                out_w = driver.forward(encoder, data, edge_weight=ones, collect=True)
                with torch.no_grad():
                    ref = driver.forward(encoder, data, collect=True)
                self.assertClose(out_w.node_repr, ref.node_repr)
                for lw, lr in zip(out_w.obs.layers, ref.obs.layers):
                    self.assertClose(lw.messages, lr.messages)
                out_w.node_repr.pow(2).sum().backward()
                self.assertGreater(ones.grad.abs().sum().item(), 0.0)

                # Zero weights on non-loop edges == removing those edges.
                keep = torch.rand(E, generator=torch.Generator().manual_seed(3)) < 0.6
                keep |= data.edge_index[0] == data.edge_index[1]
                pruned = data.clone()
                pruned.edge_index = data.edge_index[:, keep]
                with torch.no_grad():
                    out_mask = driver.forward(encoder, data, edge_weight=keep.float())
                    out_pruned = driver.forward(encoder, pruned)
                self.assertClose(out_mask.node_repr, out_pruned.node_repr)

    def test_edge_weight_rejected_elsewhere(self):
        data = _batch()
        for name in ("gat", "transformer", "fagcn", "h2gcn", "nodeformer"):
            with self.subTest(model=name):
                encoder, driver, proj = _setup(name)
                with self.assertRaises(NotImplementedError):
                    driver.forward(
                        encoder, data, edge_weight=torch.ones(data.edge_index.size(1)), projections=proj
                    )

    def test_message_prompt_count_checked(self):
        encoder, driver, _ = _setup("gcn")
        with self.assertRaises(ValueError):
            driver.forward(encoder, _batch(), message_prompts=[None])


class NodeFormerFixedTest(unittest.TestCase):
    def test_driver_requires_projections(self):
        encoder, driver, _ = _setup("nodeformer")
        with self.assertRaises(ValueError):
            driver.forward(encoder, _batch())

    def test_build_fixed_projections(self):
        encoder, _, _ = _setup("nodeformer")
        state = torch.get_rng_state()
        p1 = build_fixed_projections(encoder, seed=42)
        p2 = build_fixed_projections(encoder, seed=42)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        self.assertEqual(len(p1), len(encoder.model.convs))
        for a, b in zip(p1, p2):
            self.assertEqual(tuple(a.shape), (6, 8))
            self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(p1[0], p1[1]))
        self.assertTrue(torch.equal(p1[1], create_projection_matrix(6, 8, seed=42 + 1009 * 2)))

    def _conv_inputs(self, rb_order: int, use_edge_loss: bool = False):
        torch.manual_seed(0)
        conv = NodeFormerConv(8, 6, num_heads=2, nb_random_features=5, rb_order=rb_order,
                              use_edge_loss=use_edge_loss)
        _randomize(conv)
        data = _batch()
        z = torch.randn(1, data.num_nodes, 8, generator=torch.Generator().manual_seed(2))
        adjs = [(data.edge_index[0], data.edge_index[1])]
        return conv, z, adjs, data.batch, create_projection_matrix(5, 6, seed=11)

    def test_conv_parity_with_native(self):
        for use_edge_loss in (False, True):
            for training in (False, True):  # training exercises the Gumbel branch
                with self.subTest(use_edge_loss=use_edge_loss, training=training):
                    conv, z, adjs, batch, proj = self._conv_inputs(1, use_edge_loss)
                    conv.train(training)
                    with mock.patch("src.model.nodeformer.create_projection_matrix", return_value=proj):
                        torch.manual_seed(5)
                        native = conv(z, adjs, 0.5, batch=batch)
                    native = native[0] if use_edge_loss else native
                    torch.manual_seed(5)
                    ours, weight = nodeformer_conv_forward(conv, z, adjs, 0.5, batch, proj, return_weight=True)
                    self.assertTrue(torch.allclose(ours, native, atol=ATOL, rtol=RTOL))
                    self.assertEqual(tuple(weight.shape), (1, adjs[0][0].numel(), 2))

    def test_edge_weights_are_native_attention(self):
        # On the complete per-graph edge set the sparse coefficients sum to
        # one per receiver and their weighted values rebuild the all-pair
        # aggregate (per-graph denominators, same kernel features).
        conv, z, _, batch, proj = self._conv_inputs(0)
        conv.eval()
        same = batch.view(-1, 1) == batch.view(1, -1)
        sender, receiver = same.nonzero().t()
        agg, value, weight = nodeformer_conv_aggregate(
            conv, z, [(sender, receiver)], 1.0, batch, proj, return_weight=True
        )
        n = batch.numel()
        self.assertTrue(torch.allclose(scatter(weight[0], receiver, 0, dim_size=n), torch.ones(n, 2), atol=ATOL))
        messages = weight[0].unsqueeze(-1) * value[0, sender]
        self.assertTrue(torch.allclose(scatter(messages, receiver, 0, dim_size=n), agg[0], atol=ATOL, rtol=RTOL))


if __name__ == "__main__":
    unittest.main()
