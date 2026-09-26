"""MTG forward driver: replay a VANILLA encoder with message prototypes fused
into the input of every frozen layer.

``fuse[l]`` maps the state that layer ``l`` would read (post act/BN/dropout
of the previous layer) to ``H_M``; the whole layer (attention, message,
update) then consumes ``H_M`` (MTG Eq. 15), the last layer included.
Per backbone (dims = ``resolve_edgeprompt_prompt_spec(cfg).dim_list``):

- gcn / gin / gat / transformer: ``conv(fuse[l](x), edge_index)`` with the
  native act -> [bn] -> dropout between layers.
- fagcn: fuse before each FAConv in hidden space; the ``eps * x0`` reference
  stays the unprompted ``lin_in(x)``; no prototype on raw ``x``.
- h2gcn: fuse before each hop aggregation (input then hidden space); the
  final concat uses the layer outputs ``[x1 || x2]``.
- nodeformer: fuse before each conv; the residual adds the prompted
  ``z_M``; each conv uses its FIXED projection (``build_fixed_projections``)
  so the prototypes cannot change the random features.  JK (if enabled)
  concatenates the layer outputs.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor

from src.model.encoder import GNNEncoder
from src.model.fagcn import FAGCNEncoder
from src.model.h2gcn import H2GCNEncoder
from src.model.nodeformer import NodeFormerEncoder
from src.model.transformer import TransformerEncoder

from .nodeformer_fixed import nodeformer_conv_forward


def _check_layers(fuse: Sequence, num_layers: int) -> None:
    if len(fuse) != num_layers:
        raise ValueError(f"MTG expects one fusion module per layer ({num_layers}), got {len(fuse)}.")


def _conv_stack(model, x, edge_index, fuse):
    """``GNNEncoder`` (gcn/gin/gat) and ``TransformerEncoder`` layer loop."""
    use_batchnorm = getattr(model, "use_batchnorm", False)
    last = len(model.convs) - 1
    for idx, conv in enumerate(model.convs):
        x = conv(fuse[idx](x), edge_index)
        if idx != last:
            x = model.act(x)
            if use_batchnorm:
                x = model.bns[idx](x)
            x = model.dropout(x)
        elif use_batchnorm:
            x = model.bns[idx](x)
    return x


def _fagcn(model, x, edge_index, fuse):
    x = model.lin_in(x)
    x0 = x
    last = len(model.convs) - 1
    for idx, conv in enumerate(model.convs):
        x = conv(fuse[idx](x), x0, edge_index)
        if model.use_batchnorm:
            x = model.bns[idx](x)
        if idx != last:
            x = model.act(x)
            x = model.dropout(x)
    return model.out_lin(x)


def _h2gcn_hop(x, row, col, deg):
    out = torch.zeros_like(x)
    out.index_add_(0, row, x[col])
    return out / deg.view(-1, 1)


def _h2gcn(model, x, edge_index, fuse):
    edge_with_self, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    row, col = edge_with_self
    deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1).to(x.device)

    x1 = model.lin1(_h2gcn_hop(fuse[0](x), row, col, deg))
    if model.bn1 is not None:
        x1 = model.bn1(x1)
    x1 = model.act(x1)
    x1 = model.dropout(x1)

    x2 = model.lin2(_h2gcn_hop(fuse[1](x1), row, col, deg))
    if model.bn2 is not None:
        x2 = model.bn2(x2)
    x2 = model.act(x2)
    x2 = model.dropout(x2)
    return model.out_lin(torch.cat([x1, x2], dim=-1))


def _nodeformer(model, x, edge_index, batch, fuse, projections):
    inner = model.model
    if projections is None or len(projections) != len(inner.convs):
        raise ValueError(
            f"MTG NodeFormer needs {len(inner.convs)} fixed per-layer projections; "
            "build them once with build_fixed_projections(model, seed)."
        )
    if isinstance(edge_index, SparseTensor):
        coo = edge_index.coo()
        edge_index = (coo[0], coo[1])
    adjs = [(edge_index[0].to(x.device), edge_index[1].to(x.device))]

    z = inner.fcs[0](x.unsqueeze(0))
    if inner.use_bn:
        z = inner.bns[0](z)
    z = inner.activation(z)
    z = F.dropout(z, p=inner.dropout, training=inner.training)
    layer_ = [z]
    for i, conv in enumerate(inner.convs):
        z_m = fuse[i](z)
        z = nodeformer_conv_forward(conv, z_m, adjs, model.tau, batch, projections[i])
        if inner.use_residual:
            z = z + z_m
        if inner.use_bn:
            z = inner.bns[i + 1](z)
        if inner.use_act:
            z = inner.activation(z)
        z = F.dropout(z, p=inner.dropout, training=inner.training)
        layer_.append(z)
    if inner.use_jk:
        z = torch.cat(layer_, dim=-1)
    return inner.fcs[-1](z).squeeze(0)


def forward_with_mtg(
    model: nn.Module,
    data,
    fuse: Sequence[nn.Module],
    projections: Optional[Sequence[torch.Tensor]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(node_repr, graph_repr)`` of *model* with ``fuse[l]`` before layer ``l``.

    ``projections`` (one per conv) is required for NodeFormer and ignored
    otherwise.  ``graph_repr`` is the encoder's native pooling when
    ``data.batch`` exists, else ``None``.
    """
    x, edge_index = data.x, data.edge_index
    batch = getattr(data, "batch", None)
    if isinstance(model, (GNNEncoder, TransformerEncoder)) and getattr(model, "model_type", "") != "mlp":
        _check_layers(fuse, len(model.convs))
        node_repr = _conv_stack(model, x, edge_index, fuse)
    elif isinstance(model, FAGCNEncoder):
        _check_layers(fuse, len(model.convs))
        node_repr = _fagcn(model, x, edge_index, fuse)
    elif isinstance(model, H2GCNEncoder):
        _check_layers(fuse, 2)
        node_repr = _h2gcn(model, x, edge_index, fuse)
    elif isinstance(model, NodeFormerEncoder):
        _check_layers(fuse, len(model.model.convs))
        node_repr = _nodeformer(model, x, edge_index, batch, fuse, projections)
    else:
        raise ValueError(f"MTG has no forward driver for encoder {type(model).__name__}.")
    graph_repr = model.pool(node_repr, batch) if batch is not None else None
    return node_repr, graph_repr
