"""Task-native readouts on (induced) prediction graphs (GapTune paper Eq. 17).

- node: the focal node, ``h[data.target_node_index]``;
- edge: ``[h_u + h_v || h_u * h_v]`` for the candidate pair
  ``(u, v) = data.edge_label_index`` (local, batch-offset); symmetric in
  ``u`` and ``v``;
- graph: mean over the graph's nodes.

The level is the RAW task level (``finetune.dataset.task_level``): induced
node/edge tasks are promoted to graph batches, but keep their node/pair
readout.
"""

from __future__ import annotations

import torch

from src.utils.pool import get_batch_vector, pool_nodes, pool_target_nodes


def readout_input_dim(repr_dim: int, task_level_raw: str) -> int:
    """Readout width: ``2 * repr_dim`` for edges, ``repr_dim`` otherwise."""
    return 2 * int(repr_dim) if str(task_level_raw).lower() == "edge" else int(repr_dim)


def task_native_readout(node_repr: torch.Tensor, data, task_level_raw: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(representations, labels)`` with one row per prediction object."""
    level = str(task_level_raw).lower()
    labels = torch.as_tensor(data.y)
    if level == "node":
        return pool_target_nodes(node_repr, data), labels
    if level == "edge":
        u, v = torch.as_tensor(data.edge_label_index, device=node_repr.device).view(2, -1)
        h_u, h_v = node_repr[u], node_repr[v]
        return torch.cat([h_u + h_v, h_u * h_v], dim=-1), labels
    return pool_nodes(node_repr, get_batch_vector(data), mode="mean"), labels


__all__ = ["readout_input_dim", "task_native_readout"]
