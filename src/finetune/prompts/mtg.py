"""Prompt modules for MTG finetuning (Chen et al., "Message Tuning Outshines
Graph Prompt Tuning", ICML 2026).

Before every frozen message-passing layer ``l`` MTG fuses ``m`` learnable
message prototypes into the layer input (Eq. 16-17):
``H_M = H + softmax(H W_p + b) M``, with a per-node softmax over the ``m``
prototypes.  Initialization follows the official code: Kaiming-normal
prototypes (fan_in, leaky_relu, a=0.01) and a default ``nn.Linear`` router
with bias (Eq. 17 omits the bias; the code keeps it).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MessagePrototypeFusion(nn.Module):
    """One layer's prototypes ``M [m, d]`` and router ``Linear(d, m)``."""

    def __init__(self, dim: int, num_prototypes: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(int(num_prototypes), int(dim)))
        self.router = nn.Linear(int(dim), int(num_prototypes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.prototypes, mode="fan_in", nonlinearity="leaky_relu", a=0.01)
        self.router.reset_parameters()

    def forward(self, h: Tensor) -> Tensor:
        """``h + softmax(router(h)) @ M`` for ``h`` of shape ``[..., d]``."""
        return h + torch.softmax(self.router(h), dim=-1) @ self.prototypes


class MTGPrompt(nn.Module):
    """One ``MessagePrototypeFusion`` per frozen layer; ``fuse[l]`` feeds layer ``l``."""

    def __init__(self, dims, num_prototypes: int):
        super().__init__()
        self.fuse = nn.ModuleList(MessagePrototypeFusion(d, num_prototypes) for d in dims)
