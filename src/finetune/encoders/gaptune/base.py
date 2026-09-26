"""Shared contract for the GapTune forward drivers.

A driver replays the VANILLA encoder built by
``src.model.build_encoder_from_cfg`` (the runner's strict checkpoint load is
unchanged) and exposes the two GapTune interventions (paper Eq. 13-16):

- a node prompt added once to the initial continuous state ``h^(0)``,
  after any frozen input embedding;
- per-layer message prompts added to the native, structurally weighted
  message, ``a_vu * v_u + p_{u->v}`` (never ``a_vu * (v_u + p)``, which is
  EdgePrompt's pre-weighting form), before the native aggregation/update.

``collect=True`` also returns the observations used for context pooling
and local descriptors.  Recorded messages are the native messages of the
pass (Eq. 14) without the prompt; the task collects them from an
unprompted pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn
from torch_geometric.utils import scatter


@dataclass
class LayerObs:
    """Observations of one message-passing layer ``l``."""

    inputs: torch.Tensor  # H^(l-1) [N, d_in]: the state fed to layer l
    messages: torch.Tensor  # native weighted messages [E_l, d_Ml]
    sender: torch.Tensor  # [E_l]
    receiver: torch.Tensor  # [E_l]


@dataclass
class GapTuneObservations:
    h0: torch.Tensor  # initial continuous state [N, d0] (incl. node prompt)
    h_final: torch.Tensor  # final node states [N, d_L] (= node_repr)
    layers: list[LayerObs]


@dataclass
class DriverOutput:
    node_repr: torch.Tensor
    graph_repr: Optional[torch.Tensor]
    obs: Optional[GapTuneObservations]


class GapTuneDriver(ABC):
    """Backbone-specific GapTune forward driver (classmethods only)."""

    @classmethod
    @abstractmethod
    def supports_model(cls, model: nn.Module) -> bool:
        """Whether *model* is the vanilla encoder this driver replays."""

    @classmethod
    @abstractmethod
    def _dims(cls, model: nn.Module) -> tuple[int, list[int], list[int], int]:
        """Return ``(d0, layer-input dims, message dims, final dim)``."""

    @classmethod
    def observation_dims(cls, model: nn.Module) -> dict:
        d0, _, message_dims, _ = cls._dims(model)
        return {"N": d0, "M": list(message_dims)}

    @classmethod
    def descriptor_dims(cls, model: nn.Module) -> dict:
        """``e_N = d0 + dim(h^(L))``; ``e_Ml = 2 * dim(h^(l-1)) + d_Ml`` (Eq. 10)."""
        d0, input_dims, message_dims, final_dim = cls._dims(model)
        return {
            "N": d0 + final_dim,
            "M": [2 * d_in + d_m for d_in, d_m in zip(input_dims, message_dims)],
        }

    @classmethod
    @abstractmethod
    def forward(
        cls,
        model: nn.Module,
        data,
        *,
        node_prompt: Optional[torch.Tensor] = None,
        message_prompts: Optional[Sequence[Optional[torch.Tensor]]] = None,
        collect: bool = False,
        edge_weight: Optional[torch.Tensor] = None,
        projections: Optional[Sequence[torch.Tensor]] = None,
    ) -> DriverOutput:
        """Replay *model* with the prompts injected.

        ``message_prompts[l]`` is ``[E_l, d_Ml]`` (or ``None``), with rows
        aligned to ``obs.layers[l].sender/receiver`` of the same graph.
        ``edge_weight`` ``[E]`` is a multiplicative structural weight on
        ``data.edge_index`` (gcn/gin only).  ``projections`` are the fixed
        per-layer NodeFormer projections (ignored by other backbones).
        """


def layer_prompts(message_prompts, num_layers: int) -> list[Optional[torch.Tensor]]:
    if message_prompts is None:
        return [None] * num_layers
    if len(message_prompts) != num_layers:
        raise ValueError(
            f"Expected {num_layers} message prompt entries (one per layer), "
            f"got {len(message_prompts)}."
        )
    return list(message_prompts)


def reject_edge_weight(edge_weight: Optional[torch.Tensor], model_name: str) -> None:
    if edge_weight is not None:
        raise NotImplementedError(
            f"GapTune '{model_name}' driver does not support edge_weight; "
            "only gcn and gin (source-free inversion checkpoints) do."
        )


def record(sink: Optional[list], inputs, messages, sender, receiver) -> None:
    """Append a ``LayerObs`` (messages flattened to ``[E, d]``) when collecting."""
    if sink is not None:
        sink.append(LayerObs(inputs, messages.reshape(messages.size(0), -1), sender, receiver))


def aggregate(
    messages: torch.Tensor,
    receiver: torch.Tensor,
    num_nodes: int,
    prompt: Optional[torch.Tensor],
) -> torch.Tensor:
    """Native sum aggregation of ``messages + prompt`` at ``receiver``."""
    if prompt is not None:
        messages = messages + prompt.reshape_as(messages)
    return scatter(messages, receiver, dim=0, dim_size=num_nodes, reduce="sum")


def finish(model: nn.Module, node_repr, batch, h0, layers, collect: bool) -> DriverOutput:
    graph_repr = model.pool(node_repr, batch) if batch is not None else None
    obs = GapTuneObservations(h0=h0, h_final=node_repr, layers=layers) if collect else None
    return DriverOutput(node_repr=node_repr, graph_repr=graph_repr, obs=obs)
