"""GapTune driver for the repo's two-hop ``H2GCNEncoder`` (paper Eq. 63).

``h0 = x``.  Both hops use the row-normalised, self-loop-augmented
propagation ``S`` of the vanilla encoder (one loop appended per node,
existing loops kept).  The vanilla ``index_add_`` on ``row`` then divide is
restructured into per-edge messages ``S_vu h_u = h_u / deg_v`` with
``sender = col`` and ``receiver = row``; the prompt is added per edge and
the messages are summed (``S X~ + sum_u P``).  M1 lives in the input space,
M2 in the hidden space; lin/bn/act/dropout and the concat are unchanged.
"""

from __future__ import annotations

import torch
from torch_geometric.utils import add_self_loops

from src.model.h2gcn import H2GCNEncoder

from .base import GapTuneDriver, aggregate, finish, layer_prompts, record, reject_edge_weight


class H2GCNDriver(GapTuneDriver):
    @classmethod
    def supports_model(cls, model) -> bool:
        return isinstance(model, H2GCNEncoder)

    @classmethod
    def _dims(cls, model):
        dims = [model.lin1.in_features, model.lin2.in_features]
        return dims[0], dims, dims, model.out_lin.out_features

    @classmethod
    def _hop(cls, x, row, col, deg, prompt, sink):
        messages = x[col] / deg[row].view(-1, 1)
        record(sink, x, messages, col, row)
        return aggregate(messages, row, x.size(0), prompt)

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
        reject_edge_weight(edge_weight, "h2gcn")
        prompts = layer_prompts(message_prompts, 2)
        x = data.x
        if node_prompt is not None:
            x = x + node_prompt
        layers = [] if collect else None

        edge_with_self, _ = add_self_loops(data.edge_index, num_nodes=x.size(0))
        row, col = edge_with_self
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1).to(x.device)

        x1 = model.lin1(cls._hop(x, row, col, deg, prompts[0], layers))
        if model.bn1 is not None:
            x1 = model.bn1(x1)
        x1 = model.act(x1)
        x1 = model.dropout(x1)

        x2 = model.lin2(cls._hop(x1, row, col, deg, prompts[1], layers))
        if model.bn2 is not None:
            x2 = model.bn2(x2)
        x2 = model.act(x2)
        x2 = model.dropout(x2)

        node_repr = model.out_lin(torch.cat([x1, x2], dim=-1))
        return finish(model, node_repr, getattr(data, "batch", None), x, layers, collect)
