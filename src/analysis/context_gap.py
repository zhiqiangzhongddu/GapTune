"""Independent context-gap measurements (paper App. A.4, Eq. 24-27).

Unprompted frozen encoder, layers ``l = 1..L``:
- ``H^(l)``: node states after the layer update and activation (the state fed
  to layer ``l + 1``; the encoder output for ``l = L``);
- ``M^(l)``: directed neighbour messages after structural (GCN degree)
  weighting, before summation; implementation self-loop messages excluded.
Both come from the GapTune driver's ``collect=True`` observations.

Per ``(q, l)`` the source sample fixes a coordinate standardisation (Eq. 24)
and a Gaussian bandwidth (median nonzero squared pairwise distance), both
reused for every target of the repetition; ``delta_q`` is the layer mean of
the biased squared MMD with diagonal terms (Eq. 25-27). Kernel sums use
float64 and exact pairwise differences, so identical samples give 0.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.finetune.encoders.gaptune import get_gaptune_driver

from .perturbations import CONTROL, View

STD_EPS = 1e-6  # Eq. 24 floor on source coordinate SDs
BANDWIDTH_FLOOR = 1e-6  # positive floor on tau^2 for a degenerate source sample


def sample_ids(total: int, budget: int, generator: torch.Generator) -> torch.Tensor:
    """``min(budget, total)`` indices drawn uniformly without replacement."""
    return torch.randperm(total, generator=generator)[: min(budget, total)]


@torch.no_grad()
def unprompted_observations(model, data) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """``([H^(1..L)], [M^(1..L)])``; message rows follow ``data.edge_index`` order minus self-loops."""
    obs = get_gaptune_driver(model).forward(model, data, collect=True).obs
    states = [layer.inputs for layer in obs.layers[1:]] + [obs.h_final]
    messages = [layer.messages[layer.sender != layer.receiver] for layer in obs.layers]
    return states, messages


def squared_distances(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exact pairwise squared Euclidean distances (no matmul expansion)."""
    return torch.cdist(u, v, compute_mode="donot_use_mm_for_euclid_dist").pow(2)


def median_bandwidth(u: torch.Tensor, floor: float = BANDWIDTH_FLOOR) -> float:
    """``tau^2`` = median nonzero squared pairwise distance of ``u``, floored."""
    d2 = squared_distances(u, u)
    d2 = d2[d2 > 0].sort().values
    if d2.numel() == 0:
        return floor
    k = d2.numel()
    return max(float(d2[(k - 1) // 2] + d2[k // 2]) / 2, floor)


def mmd2(u: torch.Tensor, v: torch.Tensor, tau2: float) -> float:
    """Biased empirical MMD^2 with diagonal terms (Eq. 25), Gaussian kernel."""

    def kernel_mean(a, b):
        return torch.exp(-squared_distances(a, b) / (2 * tau2)).mean()

    return float(kernel_mean(u, u) + kernel_mean(v, v) - 2 * kernel_mean(u, v))


class SourceReference:
    """Source-fitted standardisation and bandwidth per ``(q, l)``, fixed within a repetition."""

    def __init__(self, samples: dict[str, list[torch.Tensor]], eps: float = STD_EPS, floor: float = BANDWIDTH_FLOOR):
        self.fits = {}
        for q, layers in samples.items():
            self.fits[q] = []
            for z in layers:
                z = z.double()
                mean, scale = z.mean(dim=0), z.std(dim=0, unbiased=False).clamp_min(eps)
                u = (z - mean) / scale
                self.fits[q].append((mean, scale, u, median_bandwidth(u, floor)))

    def gaps(self, samples: dict[str, list[torch.Tensor]]) -> dict[str, float]:
        """``delta_q`` (Eq. 26-27) of a target sample with the same layout as the source."""
        return {
            q: sum(
                mmd2(u, (z.double() - mean) / scale, tau2)
                for (mean, scale, u, tau2), z in zip(fits, samples[q])
            ) / len(fits)
            for q, fits in self.fits.items()
        }


def view_context_gaps(
    model,
    views: dict[tuple[str, float], View],
    *,
    node_budget: int,
    message_budget: int,
    generator: torch.Generator,
) -> dict[tuple[str, float], dict[str, float]]:
    """``{"H": delta_H, "M": delta_M}`` of every view against ``views[CONTROL]``.

    Node identities are sampled once and shared by all views; message rows
    are sampled from each edge set, and views with the same accepted-swap
    count (the same edge set) reuse them. ``model`` must be in eval mode.
    """
    node_ids = sample_ids(views[CONTROL].x.size(0), node_budget, generator)
    message_ids: dict[int, torch.Tensor] = {}

    def sample(view: View) -> dict[str, list[torch.Tensor]]:
        states, messages = unprompted_observations(model, Data(x=view.x, edge_index=view.edge_index))
        if view.swaps not in message_ids:
            message_ids[view.swaps] = sample_ids(messages[0].size(0), message_budget, generator)
        ids = message_ids[view.swaps].to(messages[0].device)
        return {
            "H": [h[node_ids.to(h.device)] for h in states],
            "M": [m[ids] for m in messages],
        }

    source = sample(views[CONTROL])
    reference = SourceReference(source)
    return {
        key: reference.gaps(source if key == CONTROL else sample(view))
        for key, view in views.items()
    }
