"""Controlled target views of one graph (paper App. A.2; reused by App. C.7).

- feature (Eq. 22): ``X_a = X_s + a * Xi * Diag(sigma_X)`` with one ``N(0, 1)``
  draw ``Xi`` per repetition, reused across strengths and by the joint family;
  ``sigma_X`` is the source coordinate SD (population), exactly zero for
  constant coordinates, so these receive no noise; no clipping.
- structure: accepted double-edge swaps on the simple undirected edge set
  (self-loops and duplicates dropped); ``b_a = ceil(a |E| / 2)`` accepted
  swaps, strengths are prefixes of one accepted-swap sequence. Degrees and
  the edge count are preserved; ``eta`` (Eq. 23) is the realised changed-edge
  fraction.
- joint: the feature view's ``X`` with the structure view's adjacency.

Views never consult labels. The shared unperturbed control is keyed
``CONTROL`` and carries the simple source edge set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch_geometric.utils import to_undirected

FAMILIES = ("feature", "structure", "joint")
CONTROL = ("control", 0.0)


@dataclass
class View:
    x: torch.Tensor  # [N, d] node features
    edge_index: torch.Tensor  # [2, 2|E|] both orientations, no self-loops
    swaps: int  # accepted swaps applied; views with equal counts share the edge set
    eta: float  # realised changed-edge fraction (Eq. 23)


def simple_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Sorted unique undirected edges ``[|E|, 2]`` with ``u < v`` (self-loops dropped)."""
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    pairs = torch.stack([edge_index.min(dim=0).values, edge_index.max(dim=0).values], dim=1)
    return torch.unique(pairs, dim=0)


def feature_sigma(x: torch.Tensor) -> torch.Tensor:
    """Source coordinate SDs; exactly zero for constant coordinates."""
    sigma = x.std(dim=0, unbiased=False)
    return sigma.masked_fill(x.amax(dim=0) == x.amin(dim=0), 0.0)


def feature_view(x: torch.Tensor, noise: torch.Tensor, alpha: float, sigma: torch.Tensor) -> torch.Tensor:
    """Eq. 22."""
    return x + alpha * noise * sigma


def swap_budget(alpha: float, num_edges: int) -> int:
    """``ceil(alpha |E| / 2)`` accepted swaps (rounded first so 0.3 * 20 / 2 stays 3)."""
    return math.ceil(round(alpha * num_edges / 2, 6))


def _swap_proposals(num_edges: int, generator: torch.Generator):
    """Endless ``((i, j), flip)`` proposal stream, drawn in chunks."""
    while True:
        pairs = torch.randint(num_edges, (4096, 2), generator=generator).tolist()
        flips = torch.randint(2, (4096,), generator=generator).tolist()
        yield from zip(pairs, flips)


def double_edge_swaps(edges: torch.Tensor, budgets: list[int], generator: torch.Generator) -> list[torch.Tensor]:
    """Snapshots of ``edges`` ``[|E|, 2]`` after each budget (ascending) of accepted swaps.

    A proposal picks edges ``{u, v}``, ``{a, b}``, randomly orients ``{a, b}``
    (orienting one edge gives both rewirings) and rewires to ``{u, b}``,
    ``{a, v}``; it is accepted only when the four endpoints are distinct and
    both new edges are absent. Rejected proposals do not count; the accepted
    sequence depends only on ``generator``, so snapshots are its prefixes.
    """
    current = [tuple(e) for e in edges.tolist()]
    present = set(current)
    pending = list(budgets)
    snapshots, accepted = [], 0
    max_proposals = 100 * max(budgets) + 1000
    for step, ((i, j), flip) in enumerate(_swap_proposals(len(current), generator)):
        while pending and pending[0] == accepted:
            snapshots.append(torch.tensor(current, dtype=edges.dtype).reshape(-1, 2))
            pending.pop(0)
        if not pending:
            return snapshots
        if step == max_proposals:
            raise RuntimeError(
                f"Only {accepted} of {pending[-1]} double-edge swaps accepted after {max_proposals} proposals."
            )
        (u, v), (a, b) = current[i], current[j]
        if flip:
            a, b = b, a
        if len({u, v, a, b}) < 4:
            continue
        new_i, new_j = (min(u, b), max(u, b)), (min(a, v), max(a, v))
        if new_i in present or new_j in present:
            continue
        present.difference_update((current[i], current[j]))
        present.update((new_i, new_j))
        current[i], current[j] = new_i, new_j
        accepted += 1


def changed_edge_fraction(source_edges: torch.Tensor, target_edges: torch.Tensor, num_nodes: int) -> float:
    """Eq. 23: ``|E_s sym-diff E_r| / (2 |E_s|)`` on ``[|E|, 2]`` undirected edge lists."""
    source_keys = source_edges[:, 0] * num_nodes + source_edges[:, 1]
    target_keys = target_edges[:, 0] * num_nodes + target_edges[:, 1]
    shared = int(torch.isin(target_keys, source_keys).sum())
    return (source_keys.numel() + target_keys.numel() - 2 * shared) / (2 * source_keys.numel())


def controlled_views(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    strengths: list[float],
    *,
    generator: torch.Generator,
) -> dict[tuple[str, float], View]:
    """``CONTROL`` plus ``(family, alpha)`` views for every nonzero strength.

    ``generator`` (CPU) draws the feature noise first, then the swap sequence.
    """
    num_nodes = x.size(0)
    edges = simple_undirected_edges(edge_index.cpu())
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype).to(x.device)
    sigma = feature_sigma(x)
    alphas = sorted(a for a in strengths if a > 0)
    budgets = [swap_budget(a, edges.size(0)) for a in alphas]
    snapshots = double_edge_swaps(edges, budgets, generator) if alphas else []

    def undirected(e: torch.Tensor) -> torch.Tensor:
        return to_undirected(e.t(), num_nodes=num_nodes).to(edge_index.device)

    source_index = undirected(edges)
    views = {CONTROL: View(x, source_index, 0, 0.0)}
    for alpha, swaps, rewired in zip(alphas, budgets, snapshots):
        x_alpha = feature_view(x, noise, alpha, sigma)
        rewired_index = undirected(rewired)
        eta = changed_edge_fraction(edges, rewired, num_nodes)
        views[("feature", alpha)] = View(x_alpha, source_index, 0, 0.0)
        views[("structure", alpha)] = View(x, rewired_index, swaps, eta)
        views[("joint", alpha)] = View(x_alpha, rewired_index, swaps, eta)
    return views
