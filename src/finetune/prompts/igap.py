"""Prompt modules for IGAP finetuning (Yan et al., WWW 2024).

IGAP combines three prompts around a frozen encoder:

- graph-signal prompt ``P_s`` (Eq. 9-10) on the input features, with
  GPF+-style input-conditioned coefficients so it applies to unseen graphs;
- spectral-alignment prompt ``P_t`` (Eq. 11-12), a ``K x K`` matrix acting on
  the coordinates of the ``K`` lowest-frequency eigenvectors of the
  combinatorial Laplacian ``L = D - A`` of each input graph;
- label prompt ``P_l`` (Eq. 17-18), cosine-similarity class prototypes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import remove_self_loops, to_dense_adj, to_dense_batch, to_undirected

from .gpf import GPFPlusPrompt


@torch.no_grad()
def laplacian_eigvecs(edge_index: Tensor, batch: Tensor, num_eigvecs: int) -> Tensor:
    """Return ``[B, M_max, K]`` lowest-frequency eigenvectors of each graph's ``D - A``.

    The adjacency is unweighted, symmetrized and loop-free; ``eigh`` runs in
    float64 and every eigenvector is sign-canonicalized (largest-magnitude
    entry positive). Row ``m`` of graph ``b`` is its ``m``-th node in
    ``to_dense_batch`` order. Rows beyond a graph's size and columns beyond its
    node count (``M_b < K``) are exactly zero.
    """
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=batch.numel())
    adj = to_dense_adj(edge_index, batch=batch).double()
    counts = torch.bincount(batch, minlength=adj.size(0))
    max_nodes = adj.size(-1)
    node_mask = torch.arange(max_nodes, device=batch.device).unsqueeze(0) < counts.unsqueeze(1)
    deg = adj.sum(-1)
    lap = torch.diag_embed(deg) - adj
    # lambda_max(D - A) <= 2 * max degree, so padded rows sort after every real eigenpair.
    lap = lap + torch.diag_embed((~node_mask).double() * (2.0 * deg.max() + 1.0))
    _, eigvecs = torch.linalg.eigh(lap)
    eigvecs = eigvecs * (node_mask.unsqueeze(2) & node_mask.unsqueeze(1))
    peak = eigvecs.gather(1, eigvecs.abs().argmax(dim=1, keepdim=True))
    eigvecs = eigvecs * torch.where(peak < 0, -1.0, 1.0)
    if max_nodes < num_eigvecs:
        eigvecs = F.pad(eigvecs, (0, num_eigvecs - max_nodes))
    return eigvecs[..., :num_eigvecs]


class IGAPPrompt(nn.Module):
    """Graph-signal prompt ``P_s`` and spectral-alignment prompt ``P_t``.

    ``prompt_input`` builds ``U_K P_t^T U_K^T X~`` and ``align_output`` builds
    ``U_K P_t U_K^T H``; around a linear spectral filter ``f = U g(Lambda) U^T``
    the sandwich equals IGAP Eq. 12 ``U_K P_t g(Lambda_K) P_t^T U_K^T X~``.
    """

    def __init__(
        self,
        in_channels: int,
        num_signal_prompts: int,
        num_eigvecs: int,
        use_signal_prompt: bool = True,
        use_spectral_prompt: bool = True,
    ):
        super().__init__()
        self.num_eigvecs = int(num_eigvecs)
        self.signal = GPFPlusPrompt(in_channels, num_signal_prompts) if use_signal_prompt else None
        self.alignment = nn.Parameter(torch.eye(self.num_eigvecs)) if use_spectral_prompt else None

    def project(self, y: Tensor, eigvecs: Tensor, batch: Tensor, transpose: bool) -> Tensor:
        """Return ``U_K P U_K^T y`` per graph (``P = P_t^T`` when ``transpose``)."""
        dense, node_mask = to_dense_batch(y, batch)
        coeffs = eigvecs.transpose(1, 2) @ dense
        align = self.alignment.t() if transpose else self.alignment
        return (eigvecs @ (align @ coeffs))[node_mask]

    def basis(self, edge_index: Tensor, batch: Tensor, dtype: torch.dtype) -> Tensor | None:
        if self.alignment is None:
            return None
        return laplacian_eigvecs(edge_index, batch, self.num_eigvecs).to(dtype)

    def prompt_input(self, x: Tensor, eigvecs: Tensor | None, batch: Tensor) -> Tensor:
        if self.signal is not None:
            x = self.signal.add(x)
        if eigvecs is not None:
            x = self.project(x, eigvecs, batch, transpose=True)
        return x

    def align_output(self, h: Tensor, eigvecs: Tensor | None, batch: Tensor) -> Tensor:
        if eigvecs is None:
            return h
        return self.project(h, eigvecs, batch, transpose=False)


class IGAPLabelPrompt(nn.Module):
    """Label prompt ``P_l``: logits from cosine similarity to learnable prototypes.

    ``pairwise=False``: one prototype per logit, ``cos(h, p_j) / tau``
    (multi-class softmax). ``pairwise=True``: two prototypes per logit,
    ``(cos(h, p_pos) - cos(h, p_neg)) / tau``, i.e. the two-prototype softmax
    written as one BCE logit (binary graph/edge tasks, each multilabel task).
    """

    def __init__(self, dim: int, num_outputs: int, pairwise: bool, tau: float):
        super().__init__()
        self.num_outputs = int(num_outputs)
        self.pairwise = bool(pairwise)
        self.tau = float(tau)
        per_output = 2 if self.pairwise else 1
        self.prototypes = nn.Parameter(torch.empty(self.num_outputs * per_output, int(dim)))
        glorot(self.prototypes)

    def forward(self, h: Tensor) -> Tensor:
        cos = F.normalize(h, dim=-1) @ F.normalize(self.prototypes, dim=-1).t()
        if self.pairwise:
            cos = cos.view(h.size(0), self.num_outputs, 2)
            return (cos[..., 1] - cos[..., 0]) / self.tau
        return cos / self.tau
