"""GapTune driver for ``TransformerEncoder`` (PyG 2.5 ``TransformerConv``).

Message ``m = alpha_vu * (W_V h_u + b_V)`` per head, flattened to
``heads * C``; attention is recomputed from the current (prompted) states.
The head mean and ``lin_skip(h_v)`` stay in UPD.  No self-loops are added.
"""

from __future__ import annotations

import math

import torch.nn.functional as F
from torch_geometric.utils import softmax

from src.model.transformer import TransformerEncoder

from .base import GapTuneDriver, aggregate, finish, layer_prompts, record, reject_edge_weight


class TransformerDriver(GapTuneDriver):
    @classmethod
    def supports_model(cls, model) -> bool:
        return isinstance(model, TransformerEncoder)

    @classmethod
    def _dims(cls, model):
        convs = model.convs
        return (
            convs[0].in_channels,
            [conv.in_channels for conv in convs],
            [conv.heads * conv.out_channels for conv in convs],
            convs[-1].out_channels,
        )

    @classmethod
    def _conv(cls, conv, x, edge_index, prompt, sink):
        H, C = conv.heads, conv.out_channels
        num_nodes = x.size(0)
        query = conv.lin_query(x).view(-1, H, C)
        key = conv.lin_key(x).view(-1, H, C)
        value = conv.lin_value(x).view(-1, H, C)
        sender, receiver = edge_index

        alpha = (query[receiver] * key[sender]).sum(dim=-1) / math.sqrt(C)
        alpha = softmax(alpha, receiver, num_nodes=num_nodes)
        alpha = F.dropout(alpha, p=conv.dropout, training=conv.training)
        messages = value[sender] * alpha.view(-1, H, 1)  # [E, H, C]
        record(sink, x, messages, sender, receiver)
        out = aggregate(messages, receiver, num_nodes, prompt).mean(dim=1)
        return out + conv.lin_skip(x)

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
        reject_edge_weight(edge_weight, "transformer")
        x, edge_index = data.x, data.edge_index
        num_layers = len(model.convs)
        prompts = layer_prompts(message_prompts, num_layers)
        if node_prompt is not None:
            x = x + node_prompt
        h0 = x
        layers = [] if collect else None

        for idx, conv in enumerate(model.convs):
            x = cls._conv(conv, x, edge_index, prompts[idx], layers)
            if idx != num_layers - 1:
                x = model.act(x)
                x = model.dropout(x)
        return finish(model, x, getattr(data, "batch", None), h0, layers, collect)
