"""IGAP prompt finetuning tests.

Covers the pieces specific to this baseline:

- ``laplacian_eigvecs``: per-graph eigenpairs of the loop-free combinatorial
  Laplacian, ascending, sign-canonical, zero-padded for mixed graph sizes;
- the projection sandwich ``U_K P_t U_K^T f(A, U_K P_t^T U_K^T X~)`` equals
  IGAP Eq. 12 for a linear spectral filter, and reduces to the plain encoder
  output for ``P_t = I``, zero ``P_s`` and ``K >= M``;
- the head/label-prompt output dims per task type through the shared
  ``FinetuneSupervised`` readout, and gradient routing (prompts + head only).
"""

from __future__ import annotations

import unittest
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import remove_self_loops, to_dense_adj, to_undirected
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.methods.igap import FinetuneIGAP
from src.finetune.prompts.igap import IGAPLabelPrompt, IGAPPrompt, laplacian_eigvecs
from src.model.encoder import build_encoder_from_cfg

IN_DIM = 6


def igap_cfg(task_level="graph", induced=False, task_type="classification", label_dim=1, num_classes=4, **igap):
    cfg = set_cfg(CN())
    cfg.finetune.method = "igap"
    cfg.finetune.dataset.name = "toy"
    cfg.finetune.dataset.task_level = task_level
    cfg.finetune.dataset.induced = induced
    cfg.finetune.dataset.task_type = task_type
    cfg.finetune.dataset.label_dim = label_dim
    cfg.finetune.dataset.num_classes = num_classes
    cfg.model.name = "gcn"
    cfg.model.in_dim = IN_DIM
    cfg.model.hidden_dim = 8
    cfg.model.out_dim = 5
    cfg.model.num_layers = 2
    for key, value in igap.items():
        setattr(cfg.finetune.igap, key, value)
    return cfg


def random_graph(num_nodes, gen, extra_edges=3, self_loop=False, **attrs):
    """Random connected graph: a random tree plus a few extra undirected edges."""
    edges = [(int(torch.randint(0, i, (1,), generator=gen)), i) for i in range(1, num_nodes)]
    for _ in range(extra_edges):
        u, v = torch.randint(0, num_nodes, (2,), generator=gen).tolist()
        if u != v:
            edges.append((u, v))
    edge_index = to_undirected(torch.tensor(edges).t(), num_nodes=num_nodes)
    if self_loop:
        edge_index = torch.cat([edge_index, torch.tensor([[0], [0]])], dim=1)
    x = torch.randn(num_nodes, IN_DIM, generator=gen)
    return Data(x=x, edge_index=edge_index, **attrs)


def dense_laplacian(edge_index, num_nodes):
    edge_index, _ = remove_self_loops(edge_index)
    adj = to_dense_adj(to_undirected(edge_index, num_nodes=num_nodes), max_num_nodes=num_nodes)[0].double()
    return torch.diag(adj.sum(-1)) - adj


def mixed_batch(sizes=(3, 7, 5), seed=0):
    gen = torch.Generator().manual_seed(seed)
    graphs = [random_graph(n, gen, self_loop=(i == 0)) for i, n in enumerate(sizes)]
    return Batch.from_data_list(graphs), graphs


def frozen_encoder(cfg):
    torch.manual_seed(0)
    model = build_encoder_from_cfg(cfg, in_dim=cfg.model.in_dim).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


class LinearSpectralFilter(nn.Module):
    """``f(A, X) = g(L) X W`` with ``g(L) = I - 0.3 L + 0.05 L^2`` (loop-free ``D - A``)."""

    @staticmethod
    def g(lam):
        return 1.0 - 0.3 * lam + 0.05 * lam**2

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, data):
        lap = dense_laplacian(data.edge_index, data.num_nodes).to(data.x.dtype)
        eye = torch.eye(data.num_nodes, dtype=lap.dtype)
        return (eye - 0.3 * lap + 0.05 * lap @ lap) @ self.lin(data.x), None


class LaplacianBasisTest(unittest.TestCase):
    def check_basis(self, eigvecs, graphs, num_eigvecs):
        self.assertEqual(eigvecs.shape, (len(graphs), max(g.num_nodes for g in graphs), num_eigvecs))
        for b, graph in enumerate(graphs):
            m = graph.num_nodes
            kept = min(m, num_eigvecs)
            u = eigvecs[b, :m, :kept]
            lap = dense_laplacian(graph.edge_index, m)
            full = torch.linalg.eigvalsh(lap)
            # Columns are orthonormal eigenvectors with the K smallest eigenvalues, ascending.
            torch.testing.assert_close(u.t() @ u, torch.eye(kept, dtype=u.dtype))
            torch.testing.assert_close(lap @ u, u * full[:kept])
            # Sign canonicalization: largest-|.| entry of each eigenvector is positive.
            peak = u.gather(0, u.abs().argmax(dim=0, keepdim=True))
            self.assertTrue(bool((peak > 0).all()))
            # Padded rows and the columns beyond the graph's size are exactly zero.
            self.assertTrue(bool((eigvecs[b, m:] == 0).all()))
            self.assertTrue(bool((eigvecs[b, :, kept:] == 0).all()))

    def test_mixed_sizes_truncated_and_padded(self):
        batch, graphs = mixed_batch()
        for num_eigvecs in (4, 6, 10):
            with self.subTest(K=num_eigvecs):
                eigvecs = laplacian_eigvecs(batch.edge_index, batch.batch, num_eigvecs)
                self.check_basis(eigvecs, graphs, num_eigvecs)

    def test_batched_projection_matches_per_graph(self):
        batch, graphs = mixed_batch()
        prompt = IGAPPrompt(IN_DIM, 2, 4).double()
        with torch.no_grad():
            prompt.alignment.add_(0.3 * torch.randn(4, 4, generator=torch.Generator().manual_seed(1)))
        eigvecs = laplacian_eigvecs(batch.edge_index, batch.batch, 4)
        x = batch.x.double()
        projected = prompt.project(x, eigvecs, batch.batch, transpose=True)
        for b, graph in enumerate(graphs):
            u = eigvecs[b, : graph.num_nodes]
            expected = u @ prompt.alignment.t() @ u.t() @ x[batch.ptr[b] : batch.ptr[b + 1]]
            torch.testing.assert_close(projected[batch.ptr[b] : batch.ptr[b + 1]], expected)


class SandwichTest(unittest.TestCase):
    def test_equals_eq12_for_linear_spectral_filter(self):
        # K=4 truncates the 7- and 5-node graphs and zero-pads the 3-node graph.
        batch, graphs = mixed_batch()
        gen = torch.Generator().manual_seed(2)
        prompt = IGAPPrompt(IN_DIM, 3, 4).double()
        with torch.no_grad():
            prompt.alignment.add_(0.3 * torch.randn(4, 4, generator=gen))
        model = LinearSpectralFilter(IN_DIM, 3).double()
        x = batch.x.double()
        eigvecs = prompt.basis(batch.edge_index, batch.batch, x.dtype)
        prompted = batch.clone()
        prompted.x = prompt.prompt_input(x, eigvecs, batch.batch)
        z = prompt.align_output(model(prompted)[0], eigvecs, batch.batch)

        x_signal = prompt.signal.add(x)
        p_t = prompt.alignment
        for b, graph in enumerate(graphs):
            rows = slice(int(batch.ptr[b]), int(batch.ptr[b + 1]))
            u = eigvecs[b, : graph.num_nodes]
            lam = torch.diagonal(u.t() @ dense_laplacian(graph.edge_index, graph.num_nodes) @ u)
            g = LinearSpectralFilter.g(lam) * (u.norm(dim=0) > 0)
            # Eq. 12: Z~ = U_K P_t g(Lambda_K) P_t^T U_K^T X~ (then the filter's feature map W).
            expected = u @ p_t @ torch.diag(g) @ p_t.t() @ u.t() @ model.lin(x_signal[rows])
            torch.testing.assert_close(z[rows], expected)

    def test_identity_alignment_and_zero_signal_reduce_to_encoder(self):
        cfg = igap_cfg()
        task = FinetuneIGAP(cfg)
        model = frozen_encoder(cfg)
        with torch.no_grad():
            task.prompt.signal.p_list.zero_()
        batch, _ = mixed_batch(sizes=(5, 9, 12))
        node_repr, graph_repr = task.encode(model, batch)
        expected_node, expected_graph = model(batch)
        torch.testing.assert_close(node_repr, expected_node, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(graph_repr, expected_graph, atol=1e-5, rtol=1e-5)

        # K < M truncates the spectrum, so the view no longer equals the encoder.
        truncated = FinetuneIGAP(igap_cfg(num_eigvecs=4))
        with torch.no_grad():
            truncated.prompt.signal.p_list.zero_()
        self.assertFalse(torch.allclose(truncated.encode(model, batch)[0], expected_node, atol=1e-3))


class TaskTest(unittest.TestCase):
    def batch_for(self, kind):
        gen = torch.Generator().manual_seed(3)
        graphs = []
        for i, n in enumerate((4, 6, 5)):
            if kind == "node":
                attrs = {"y": torch.tensor([i % 4]), "target_node_index": torch.tensor([1])}
            elif kind == "edge":
                attrs = {"y": torch.tensor([i % 2]), "edge_label_index": torch.tensor([[0], [2]])}
            elif kind == "multilabel":
                attrs = {"y": torch.tensor([[1.0, 0.0, float("nan")]])}
            else:
                attrs = {"y": torch.randn(1, 14, generator=gen)}
            graphs.append(random_graph(n, gen, **attrs))
        return Batch.from_data_list(graphs)

    CASES = {
        "node": (dict(task_level="node", induced=True, num_classes=4), 4, False),
        "edge": (dict(task_level="edge", induced=True, num_classes=2), 1, True),
        "multilabel": (dict(label_dim=3, num_classes=2), 3, True),
        "regression": (dict(task_type="regression", label_dim=14), 14, None),
    }

    def test_output_dims_and_gradient_routing(self):
        for kind, (kwargs, out_dim, pairwise) in self.CASES.items():
            with self.subTest(kind=kind):
                cfg = igap_cfg(**kwargs)
                FinetuneIGAP.validate_cfg(cfg)
                task = FinetuneIGAP(cfg)
                model = frozen_encoder(cfg)
                head = task.supervised_head.classifier
                self.assertIsInstance(head[1], nn.ReLU)
                if pairwise is None:
                    self.assertEqual(len(head), 3)
                    self.assertEqual(head[-1].out_features, out_dim)
                else:
                    self.assertIsInstance(head[-1], IGAPLabelPrompt)
                    self.assertEqual(head[-1].pairwise, pairwise)
                loss, _primary, logits, _labels = task.supervised_head.evaluate(
                    model=partial(task.encode, model), data=self.batch_for(kind),
                    device=torch.device("cpu"), return_outputs=True,
                )
                self.assertEqual(logits.view(3, -1).shape, (3, out_dim))
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                for param in task.parameters_to_optimize():
                    self.assertIsNotNone(param.grad)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                optimizer = task.build_optimizers(model)["primary"]
                n_opt = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
                self.assertEqual(n_opt, sum(p.numel() for p in task.parameters_to_optimize()))

    def test_pairwise_label_prompt_is_two_prototype_softmax(self):
        torch.manual_seed(0)
        prompt = IGAPLabelPrompt(dim=5, num_outputs=3, pairwise=True, tau=0.1)
        h = torch.randn(4, 5)
        cos = F.normalize(h, dim=-1) @ F.normalize(prompt.prototypes, dim=-1).t()
        two_class = torch.softmax(cos.view(4, 3, 2) / 0.1, dim=-1)[..., 1]
        torch.testing.assert_close(torch.sigmoid(prompt(h)), two_class)

    def test_ablation_switches_drop_components(self):
        task = FinetuneIGAP(igap_cfg(use_signal_prompt=False, use_spectral_prompt=False, use_label_prompt=False))
        self.assertIsNone(task.prompt.signal)
        self.assertIsNone(task.prompt.alignment)
        self.assertEqual(len(task.supervised_head.classifier), 3)
        model = frozen_encoder(igap_cfg())
        batch, _ = mixed_batch()
        torch.testing.assert_close(task.encode(model, batch)[0], model(batch)[0])

    def test_validate_cfg_requires_graph_level_batches(self):
        with self.assertRaisesRegex(ValueError, "IGAP requires graph-level"):
            FinetuneIGAP.validate_cfg(igap_cfg(task_level="node", induced=False))

    def test_variant_tag(self):
        self.assertEqual(FinetuneIGAP.variant_tag(igap_cfg()), "")
        cfg = igap_cfg(num_eigvecs=16, use_label_prompt=False, lr=0.01)
        self.assertEqual(FinetuneIGAP.variant_tag(cfg), "k16-nopl-mlr0.01")


if __name__ == "__main__":
    unittest.main()
