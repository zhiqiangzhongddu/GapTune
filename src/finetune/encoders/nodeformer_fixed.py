"""Fixed-projection replay of the vendored ``NodeFormerConv``.

``src/model/nodeformer.py`` reseeds each layer's random-feature projection
from ``sum(query)``, so any change to a layer input (a prompt) also changes
its kernel features.  Prompt methods that compare unprompted and prompted
passes (GapTune, MTG) therefore hold the projection fixed: paper App. B.10
calls the native generator once per layer ``l = 1..L`` with seed
``s + 1009 * l`` and reuses that projection for the source, target and
prompted passes.  Zero-prompt parity is defined against the native operator
supplied with the same projections.

``nodeformer_conv_forward`` reproduces ``NodeFormerConv.forward`` exactly
except that the caller supplies the projection.  The pretraining-only link
loss (``use_edge_loss``) is not returned.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.model.nodeformer import (
    add_conv_relational_bias,
    create_projection_matrix,
    kernelized_gumbel_softmax,
    kernelized_softmax,
)


def build_fixed_projections(nf_model: nn.Module, seed: int) -> list[torch.Tensor]:
    """Return one projection per conv layer ``l = 1..L`` seeded ``seed + 1009 * l``.

    ``nf_model`` is the ``NodeFormerEncoder`` from ``build_encoder_from_cfg``
    (or its inner ``NodeFormer``).  The global RNG state is left untouched.
    """
    inner = getattr(nf_model, "model", nf_model)
    projections = []
    for layer, conv in enumerate(inner.convs, start=1):
        with torch.random.fork_rng():
            projection = create_projection_matrix(
                conv.nb_random_features, conv.out_channels, seed=seed + 1009 * layer
            )
        projections.append(projection.to(conv.Wq.weight.device))
    return projections


def nodeformer_conv_aggregate(
    conv: nn.Module,
    z: torch.Tensor,
    adjs,
    tau: float,
    batch: Optional[torch.Tensor],
    projection: torch.Tensor,
    return_weight: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Native pre-``Wo`` aggregate of ``conv`` with a fixed projection.

    Returns ``(aggregate [B,N,H,C], value [B,N,H,C], weight)``: the kernel
    all-pair attention output plus any native relational bias, the projected
    values, and (``return_weight``) the per-edge coefficients ``[B,E,H]`` on
    ``adjs[0]`` computed from the same kernel features and per-graph
    denominators (``None`` otherwise).
    """
    N = z.size(1)
    query = conv.Wq(z).reshape(-1, N, conv.num_heads, conv.out_channels)
    key = conv.Wk(z).reshape(-1, N, conv.num_heads, conv.out_channels)
    value = conv.Wv(z).reshape(-1, N, conv.num_heads, conv.out_channels)

    if conv.use_gumbel and conv.training:
        attn_out = kernelized_gumbel_softmax(
            query, key, value, conv.kernel_transformation, projection, adjs[0],
            conv.nb_gumbel_sample, tau, return_weight, batch=batch,
        )
    else:
        attn_out = kernelized_softmax(
            query, key, value, conv.kernel_transformation, projection, adjs[0],
            tau, return_weight, batch=batch,
        )
    aggregate, weight = attn_out if return_weight else (attn_out, None)

    if len(adjs) < conv.rb_order:
        raise ValueError(
            f"NodeFormer rb_order={conv.rb_order} requires {conv.rb_order} "
            f"adjacency order(s), but only {len(adjs)} were provided."
        )
    for i in range(conv.rb_order):
        aggregate = aggregate + add_conv_relational_bias(value, adjs[i], conv.b[i], conv.rb_trans)
    return aggregate, value, weight


def nodeformer_conv_forward(
    conv: nn.Module,
    z: torch.Tensor,
    adjs,
    tau: float,
    batch: Optional[torch.Tensor],
    projection: torch.Tensor,
    return_weight: bool = False,
):
    """``NodeFormerConv.forward`` with an injected projection.

    Returns ``z_out`` or, with ``return_weight``, ``(z_out, weight [B,E,H])``.
    """
    aggregate, _, weight = nodeformer_conv_aggregate(
        conv, z, adjs, tau, batch, projection, return_weight
    )
    z_out = conv.Wo(aggregate.flatten(-2, -1))
    return (z_out, weight) if return_weight else z_out
