"""GapTune driver for ``FAGCNEncoder`` (PyG 2.5 ``FAConv``).

``h0 = lin_in(x)``; the node prompt gives ``h~0 = h0 + p`` (Eq. 13), which
is both the first layer input and the ``eps * x0`` reference of every
FAConv.  Message ``m = tanh(att_l h_u + att_r h_v) * norm_vu * h_u``: the
signed coefficient is part of the native message (gcn_norm with the conv's
self-loops).  ``eps * h~0`` and the stack's bn/act/dropout stay in UPD.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.model.fagcn import FAGCNEncoder

from .base import GapTuneDriver, aggregate, finish, layer_prompts, record, reject_edge_weight


class FAGCNDriver(GapTuneDriver):
    @classmethod
    def supports_model(cls, model) -> bool:
        return isinstance(model, FAGCNEncoder)

    @classmethod
    def _dims(cls, model):
        hidden = model.lin_in.out_features
        num_layers = len(model.convs)
        return hidden, [hidden] * num_layers, [hidden] * num_layers, model.out_lin.out_features

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
        reject_edge_weight(edge_weight, "fagcn")
        edge_index = data.edge_index
        num_layers = len(model.convs)
        prompts = layer_prompts(message_prompts, num_layers)
        x = model.lin_in(data.x)
        if node_prompt is not None:
            x = x + node_prompt
        x0 = x
        layers = [] if collect else None

        for idx, conv in enumerate(model.convs):
            num_nodes = x.size(0)
            ei, norm = gcn_norm(
                edge_index, None, num_nodes, False, conv.add_self_loops, conv.flow, dtype=x.dtype
            )
            sender, receiver = ei
            alpha_l = conv.att_l(x)
            alpha_r = conv.att_r(x)
            gate = (alpha_l[sender] + alpha_r[receiver]).tanh().squeeze(-1)
            gate = F.dropout(gate, p=conv.dropout, training=conv.training)
            messages = x[sender] * (gate * norm).view(-1, 1)
            record(layers, x, messages, sender, receiver)
            out = aggregate(messages, receiver, num_nodes, prompts[idx])
            if conv.eps != 0.0:
                out = out + conv.eps * x0
            x = out
            if model.use_batchnorm:
                x = model.bns[idx](x)
            if idx != num_layers - 1:
                x = model.act(x)
                x = model.dropout(x)

        node_repr = model.out_lin(x)
        return finish(model, node_repr, getattr(data, "batch", None), x0, layers, collect)
