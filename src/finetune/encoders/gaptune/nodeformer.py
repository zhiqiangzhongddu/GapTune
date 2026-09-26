"""GapTune driver for ``NodeFormerEncoder`` (paper App. B.10, Eq. 64-65).

``h0 = act(LN(fcs[0](x)))`` (after dropout, a no-op in eval); the node
prompt is added there and therefore also enters the first residual.  Each
conv runs with its FIXED projection (``build_fixed_projections``) so the
kernel features do not depend on the prompt.  Messages live on the input
edges only (no self-loops added): ``m = alpha_vu * v_u`` concatenated over
heads, with ``alpha_vu`` from the native kernel features and per-graph
denominators.  The prompted aggregate is ``a~_v = u~_v + r~_v + sum_u p``
(prompt reshaped into head channels), followed by the native Wo, residual,
LayerNorm and activation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_sparse import SparseTensor

from src.model.nodeformer import NodeFormerEncoder

from ..nodeformer_fixed import nodeformer_conv_aggregate
from .base import GapTuneDriver, finish, layer_prompts, record, reject_edge_weight


class NodeFormerDriver(GapTuneDriver):
    @classmethod
    def supports_model(cls, model) -> bool:
        return isinstance(model, NodeFormerEncoder)

    @classmethod
    def _dims(cls, model):
        inner = model.model
        return (
            inner.fcs[0].out_features,
            [conv.Wq.in_features for conv in inner.convs],
            [conv.num_heads * conv.out_channels for conv in inner.convs],
            inner.fcs[-1].out_features,
        )

    @classmethod
    def forward(
        cls,
        model,
        data,
        *,
        node_prompt=None,
        message_prompts=None,
        collect=False,
        edge_weight=None,
        projections=None,
    ):
        reject_edge_weight(edge_weight, "nodeformer")
        inner = model.model
        num_layers = len(inner.convs)
        if projections is None or len(projections) != num_layers:
            raise ValueError(
                f"NodeFormer GapTune driver needs {num_layers} fixed per-layer "
                "projections; build them once with build_fixed_projections(model, seed)."
            )
        prompts = layer_prompts(message_prompts, num_layers)
        edge_index = data.edge_index
        if isinstance(edge_index, SparseTensor):
            coo = edge_index.coo()
            edge_index = (coo[0], coo[1])
        sender = edge_index[0].to(data.x.device)
        receiver = edge_index[1].to(data.x.device)
        adjs = [(sender, receiver)]
        batch = getattr(data, "batch", None)
        layers = [] if collect else None

        z = inner.fcs[0](data.x.unsqueeze(0))
        if inner.use_bn:
            z = inner.bns[0](z)
        z = inner.activation(z)
        z = F.dropout(z, p=inner.dropout, training=inner.training)
        if node_prompt is not None:
            z = z + node_prompt.unsqueeze(0)
        layer_ = [z]

        for i, conv in enumerate(inner.convs):
            agg, value, weight = nodeformer_conv_aggregate(
                conv, z, adjs, model.tau, batch, projections[i], return_weight=collect
            )
            if collect:
                messages = weight.unsqueeze(-1) * value[:, sender]  # [1, E, H, C]
                record(layers, z.squeeze(0), messages.squeeze(0), sender, receiver)
            if prompts[i] is not None:
                prompt = prompts[i].reshape(-1, conv.num_heads, conv.out_channels)
                agg = agg + scatter(prompt, receiver, dim=0, dim_size=z.size(1), reduce="sum").unsqueeze(0)
            z = conv.Wo(agg.flatten(-2, -1))
            if inner.use_residual:
                z = z + layer_[i]
            if inner.use_bn:
                z = inner.bns[i + 1](z)
            if inner.use_act:
                z = inner.activation(z)
            z = F.dropout(z, p=inner.dropout, training=inner.training)
            layer_.append(z)

        if inner.use_jk:
            z = torch.cat(layer_, dim=-1)
        node_repr = inner.fcs[-1](z).squeeze(0)
        return finish(model, node_repr, batch, layer_[0].squeeze(0), layers, collect)
