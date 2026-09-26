"""Prompt module for SUPT finetuning (Lee et al., subgraph-level universal prompt tuning)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.utils import softmax as graph_softmax


class SUPTPrompt(nn.Module):
    """SUPT prompt: ``k`` learnable bases added to the encoder input features.

    Both variants score nodes with a one-hop ``GCNConv(d, k)`` applied to
    ``x + sum_j b_j`` over the encoder's own ``edge_index`` (official code).

    - ``soft`` (official ``DiffPoolPrompt``): ``x + softmax(scores, dim=1) @ B``,
      a row-wise softmax over the bases for every node.
    - ``hard`` (official ``SAGPoolPrompt``): basis ``j``, scaled by its score,
      is added to the top-``ceil(ratio * N_g)`` nodes of every graph; each
      node's prompt sum is divided by ``1 + count`` because the official
      counter starts at one.
    """

    def __init__(
        self,
        in_channels: int,
        num_bases: int,
        variant: str = "soft",
        ratio: float = 0.4,
        hard_score: str = "tanh",
        orth_loss: bool = False,
        gcn_bias: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_bases = int(num_bases)
        self.variant = str(variant)
        self.ratio = float(ratio)
        self.hard_score = str(hard_score)
        # Official code only enables the orthogonality term for k > 1.
        self.orth_loss = bool(orth_loss) and self.num_bases > 1
        self.bases = nn.Parameter(torch.empty(self.num_bases, self.in_channels))
        self.scorer = GCNConv(self.in_channels, self.num_bases, bias=bool(gcn_bias))
        self.register_buffer("eye", torch.eye(self.num_bases), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.scorer.reset_parameters()
        if self.orth_loss:
            nn.init.orthogonal_(self.bases)
        else:
            glorot(self.bases)

    def orthogonal_loss(self) -> Tensor:
        return torch.linalg.matrix_norm(self.bases @ self.bases.t() - self.eye)

    def add(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        score = self.scorer(x + self.bases.sum(dim=0), edge_index)
        if self.variant == "soft":
            return x + torch.softmax(score, dim=1) @ self.bases

        if self.hard_score == "graph_softmax":
            score = graph_softmax(score, batch)
        else:
            score = torch.tanh(score)
        prompt = torch.zeros_like(x)
        count = torch.ones(x.size(0), dtype=x.dtype, device=x.device)
        for j in range(self.num_bases):
            perm = topk(score[:, j], self.ratio, batch)
            p_j = score[perm, j].unsqueeze(1) * self.bases[j]
            if self.orth_loss:
                p_j = p_j / self.num_bases  # official bio code
            prompt = prompt.index_add(0, perm, p_j)
            count[perm] += 1
        return x + prompt / count.unsqueeze(1)
