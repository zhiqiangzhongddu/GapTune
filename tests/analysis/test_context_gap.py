"""App. A.4 context-gap tests.

- biased MMD^2 (Eq. 25): identical samples give exactly 0, symmetric,
  nonnegative, equal to a direct evaluation of Eq. 25;
- bandwidth: median of the nonzero squared pairwise distances (average of
  the two middle values), positive floor for a degenerate sample;
- source-fitted standardisation floors zero-variance coordinates;
- GCN observations: layer states and self-loop-free messages;
- view measurement: the control has zero gaps; unchanged edge sets reuse
  message identities, other edge sets draw their own.
"""

from __future__ import annotations

import math

import torch
from torch_geometric.data import Data
from yacs.config import CfgNode as CN

from src.analysis.context_gap import (
    BANDWIDTH_FLOOR,
    SourceReference,
    median_bandwidth,
    mmd2,
    unprompted_observations,
    view_context_gaps,
)
from src.analysis.perturbations import CONTROL, View, controlled_views
from src.config import set_cfg
from src.model.encoder import build_encoder_from_cfg

IN_DIM, NUM_NODES = 4, 30


def _gcn(num_layers: int = 2):
    cfg = CN()
    set_cfg(cfg)
    cfg.model.name = "gcn"
    cfg.model.num_layers = num_layers
    cfg.model.in_dim = IN_DIM
    cfg.model.hidden_dim = 8
    cfg.model.out_dim = 5
    torch.manual_seed(0)
    return build_encoder_from_cfg(cfg, IN_DIM).eval()


def _graph(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(NUM_NODES, (2, 60), generator=g)
    return torch.randn(NUM_NODES, IN_DIM, generator=g), torch.cat([ei, ei.flip(0)], dim=1)


def test_mmd_identity_symmetry_and_eq25():
    g = torch.Generator().manual_seed(0)
    u = torch.randn(12, 3, generator=g, dtype=torch.float64)
    v = torch.randn(9, 3, generator=g, dtype=torch.float64) + 0.5
    tau2 = 1.7
    assert mmd2(u, u, tau2) == 0.0
    assert math.isclose(mmd2(u, v, tau2), mmd2(v, u, tau2), rel_tol=1e-12)
    assert mmd2(u, v, tau2) > 0

    def k(a, b):
        return math.exp(-float(((a - b) ** 2).sum()) / (2 * tau2))

    n, m = len(u), len(v)
    direct = (
        sum(k(a, b) for a in u for b in u) / n**2
        + sum(k(a, b) for a in v for b in v) / m**2
        - 2 * sum(k(a, b) for a in u for b in v) / (n * m)
    )
    assert math.isclose(mmd2(u, v, tau2), direct, rel_tol=1e-10)


def test_bandwidth_is_the_median_nonzero_squared_distance():
    as_points = lambda values: torch.tensor(values, dtype=torch.float64).view(-1, 1)  # noqa: E731
    assert median_bandwidth(as_points([0, 1, 3])) == 4.0  # {1, 9, 4}
    assert median_bandwidth(as_points([0, 1, 3, 7])) == 12.5  # {1, 4, 9, 16, 36, 49}
    assert median_bandwidth(as_points([0, 0, 0, 2])) == 4.0  # off-diagonal zeros excluded (else 2.0)
    assert median_bandwidth(as_points([5, 5, 5])) == BANDWIDTH_FLOOR


def test_source_standardisation_floors_zero_variance_coordinates():
    g = torch.Generator().manual_seed(1)
    source = torch.randn(20, 3, generator=g)
    source[:, 1] = 2.0
    reference = SourceReference({"H": [source]})
    assert reference.gaps({"H": [source.clone()]}) == {"H": 0.0}
    target = source.clone()
    target[:, 1] = 2.5  # shift only the constant coordinate
    gap = reference.gaps({"H": [target]})["H"]
    assert math.isfinite(gap) and gap > 0


def test_gcn_observations_are_layer_states_and_loop_free_messages():
    model = _gcn(num_layers=3)
    x, ei = _graph()
    ei = torch.cat([ei, torch.tensor([[3], [3]])], dim=1)  # an existing self-loop
    states, messages = unprompted_observations(model, Data(x=x, edge_index=ei))
    assert [h.shape for h in states] == [(NUM_NODES, 8), (NUM_NODES, 8), (NUM_NODES, 5)]
    model(Data(x=x, edge_index=ei))  # vanilla per-layer states after update + activation
    for h, reference in zip(states, model.get_layer_node_reprs(), strict=True):
        torch.testing.assert_close(h, reference)
    non_loop = int((ei[0] != ei[1]).sum())
    assert [m.size(0) for m in messages] == [non_loop] * 3


def test_view_gaps_control_zero_and_message_identity_reuse():
    model = _gcn()
    x, ei = _graph()
    views = controlled_views(x, ei, [0.2], generator=torch.Generator().manual_seed(2))
    control = views[CONTROL]
    num_messages = control.edge_index.size(1)
    kwargs = dict(node_budget=10, message_budget=num_messages // 2)

    gaps = view_context_gaps(model, views, generator=torch.Generator().manual_seed(3), **kwargs)
    assert gaps[CONTROL] == {"H": 0.0, "M": 0.0}
    assert all(gaps[key]["H"] > 0 and gaps[key]["M"] > 0 for key in views if key != CONTROL)

    same_edges = {CONTROL: control, ("feature", 0.1): View(control.x, control.edge_index, 0, 0.0)}
    gaps = view_context_gaps(model, same_edges, generator=torch.Generator().manual_seed(3), **kwargs)
    assert gaps[("feature", 0.1)] == {"H": 0.0, "M": 0.0}  # same message identities

    relabelled = {CONTROL: control, ("structure", 0.1): View(control.x, control.edge_index, 1, 0.0)}
    gaps = view_context_gaps(model, relabelled, generator=torch.Generator().manual_seed(3), **kwargs)
    assert gaps[("structure", 0.1)]["H"] == 0.0 and gaps[("structure", 0.1)]["M"] > 0  # fresh draw
