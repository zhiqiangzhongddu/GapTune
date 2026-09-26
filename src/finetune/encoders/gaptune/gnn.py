"""GapTune drivers for ``GNNEncoder`` (gcn / gin / gat).

The stack replays ``GNNEncoder.forward`` (conv -> act -> [bn] -> dropout,
last layer conv -> [bn]); each subclass replays one PyG 2.5 conv with the
message materialised after structural weighting:

- gcn: ``m = norm_vu * W h_u`` (gcn_norm with the conv's self-loop fill);
  bias stays in UPD.
- gin: ``m = w_vu * h_u`` (``w = 1`` without ``edge_weight``; no message
  relu); ``(1 + eps) h_v`` and the MLP stay in UPD.
- gat: ``m = alpha_vu * W h_u`` per head, flattened to ``heads * C``;
  attention from the current (prompted) states; head mean and bias in UPD.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from src.model.encoder import GNNEncoder

from .base import GapTuneDriver, aggregate, finish, layer_prompts, record, reject_edge_weight


class _GNNStackDriver(GapTuneDriver):
    MODEL_TYPE = ""

    @classmethod
    def supports_model(cls, model: nn.Module) -> bool:
        return isinstance(model, GNNEncoder) and model.model_type == cls.MODEL_TYPE

    @classmethod
    def _conv_dims(cls, conv: nn.Module) -> tuple[int, int, int]:
        """``(input dim, message dim, output dim)`` of one conv."""
        raise NotImplementedError

    @classmethod
    def _conv(cls, conv, x, edge_index, edge_weight, prompt, sink) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def _dims(cls, model):
        conv_dims = [cls._conv_dims(conv) for conv in model.convs]
        return (
            conv_dims[0][0],
            [d[0] for d in conv_dims],
            [d[1] for d in conv_dims],
            conv_dims[-1][2],
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
        x, edge_index = data.x, data.edge_index
        num_layers = len(model.convs)
        prompts = layer_prompts(message_prompts, num_layers)
        if node_prompt is not None:
            x = x + node_prompt
        h0 = x
        layers = [] if collect else None

        for idx, conv in enumerate(model.convs):
            x = cls._conv(conv, x, edge_index, edge_weight, prompts[idx], layers)
            if idx != num_layers - 1:
                x = model.act(x)
                if model.use_batchnorm:
                    x = model.bns[idx](x)
                x = model.dropout(x)
            elif model.use_batchnorm:
                x = model.bns[idx](x)
        return finish(model, x, getattr(data, "batch", None), h0, layers, collect)


class GCNDriver(_GNNStackDriver):
    MODEL_TYPE = "gcn"

    @classmethod
    def _conv_dims(cls, conv):
        return conv.in_channels, conv.out_channels, conv.out_channels

    @classmethod
    def _conv(cls, conv, x, edge_index, edge_weight, prompt, sink):
        num_nodes = x.size(0)
        ei, norm = gcn_norm(
            edge_index, edge_weight, num_nodes, conv.improved, conv.add_self_loops, conv.flow, x.dtype
        )
        sender, receiver = ei
        messages = norm.view(-1, 1) * conv.lin(x)[sender]
        record(sink, x, messages, sender, receiver)
        out = aggregate(messages, receiver, num_nodes, prompt)
        if conv.bias is not None:
            out = out + conv.bias
        return out


class GINDriver(_GNNStackDriver):
    MODEL_TYPE = "gin"

    @classmethod
    def _conv_dims(cls, conv):
        return conv.nn[0].in_features, conv.nn[0].in_features, conv.nn[-1].out_features

    @classmethod
    def _conv(cls, conv, x, edge_index, edge_weight, prompt, sink):
        sender, receiver = edge_index
        messages = x[sender]
        if edge_weight is not None:
            messages = edge_weight.view(-1, 1) * messages
        record(sink, x, messages, sender, receiver)
        out = aggregate(messages, receiver, x.size(0), prompt)
        out = out + (1 + conv.eps) * x
        return conv.nn(out)


class GATDriver(_GNNStackDriver):
    MODEL_TYPE = "gat"

    @classmethod
    def _conv_dims(cls, conv):
        return conv.in_channels, conv.heads * conv.out_channels, conv.out_channels

    @classmethod
    def _conv(cls, conv, x, edge_index, edge_weight, prompt, sink):
        reject_edge_weight(edge_weight, "gat")
        H, C = conv.heads, conv.out_channels
        num_nodes = x.size(0)
        x_lin = conv.lin(x).view(-1, H, C)
        alpha_src = (x_lin * conv.att_src).sum(dim=-1)
        alpha_dst = (x_lin * conv.att_dst).sum(dim=-1)
        if conv.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        sender, receiver = edge_index

        alpha = F.leaky_relu(alpha_src[sender] + alpha_dst[receiver], conv.negative_slope)
        alpha = softmax(alpha, receiver, num_nodes=num_nodes)
        alpha = F.dropout(alpha, p=conv.dropout, training=conv.training)
        messages = alpha.unsqueeze(-1) * x_lin[sender]  # [E, H, C]
        record(sink, x, messages, sender, receiver)
        out = aggregate(messages, receiver, num_nodes, prompt).mean(dim=1)
        if conv.bias is not None:
            out = out + conv.bias
        return out
