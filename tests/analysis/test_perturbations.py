"""App. A.2 controlled-view tests (pure tensors).

- double-edge swaps preserve every degree and the edge count, stay simple and
  loop-free; strengths are prefixes of one accepted-swap sequence;
- eta (Eq. 23) against a hand-computed symmetric difference;
- ``ceil(alpha |E| / 2)`` budgets without float drift;
- one feature-noise draw reused across strengths and by the joint family;
  zero-variance coordinates receive no noise.
"""

from __future__ import annotations

import torch

from src.analysis.perturbations import (
    CONTROL,
    FAMILIES,
    changed_edge_fraction,
    controlled_views,
    double_edge_swaps,
    feature_sigma,
    simple_undirected_edges,
    swap_budget,
)

NUM_NODES = 40


def _edge_index(seed: int = 0, num_edges: int = 120) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(NUM_NODES, (2, num_edges), generator=g)
    return torch.cat([ei, ei.flip(0)], dim=1)  # may hold self-loops and duplicates


def _degrees(edges: torch.Tensor) -> torch.Tensor:
    return torch.bincount(edges.flatten(), minlength=NUM_NODES)


def test_simple_undirected_edges_drop_loops_and_duplicates():
    ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 1, 3, 3, 2]])
    assert simple_undirected_edges(ei).tolist() == [[0, 1], [2, 3]]


def test_swaps_preserve_degrees_edge_count_and_simplicity():
    edges = simple_undirected_edges(_edge_index())
    snapshots = double_edge_swaps(edges, [0, 5, 30], torch.Generator().manual_seed(1))
    assert torch.equal(snapshots[0], edges)
    for snap in snapshots:
        assert snap.shape == edges.shape
        assert bool((snap[:, 0] < snap[:, 1]).all())  # canonical, loop-free
        assert torch.unique(snap, dim=0).size(0) == snap.size(0)  # no duplicates
        assert torch.equal(_degrees(snap), _degrees(edges))
    assert changed_edge_fraction(edges, snapshots[2], NUM_NODES) > 0


def test_strengths_are_prefixes_of_one_accepted_sequence():
    edges = simple_undirected_edges(_edge_index())
    many = double_edge_swaps(edges, [3, 5, 20], torch.Generator().manual_seed(2))
    for budget, snap in zip([3, 5, 20], many):
        alone = double_edge_swaps(edges, [budget], torch.Generator().manual_seed(2))[0]
        assert torch.equal(alone, snap)


def test_single_swap_changes_two_edges():
    edges = simple_undirected_edges(_edge_index())
    swapped = double_edge_swaps(edges, [1], torch.Generator().manual_seed(3))[0]
    assert changed_edge_fraction(edges, swapped, NUM_NODES) == 4 / (2 * edges.size(0))


def test_eta_matches_hand_computed_symmetric_difference():
    source = torch.tensor([[0, 1], [2, 3], [4, 5]])
    target = torch.tensor([[0, 3], [1, 2], [4, 5]])
    assert changed_edge_fraction(source, target, 6) == 4 / 6


def test_swap_budget_is_the_ceiling_without_float_drift():
    assert swap_budget(0.3, 20) == 3  # 0.3 * 20 / 2 = 3.0000000000000004 in floats
    assert swap_budget(0.1, 10) == 1
    assert swap_budget(0.05, 119081) == 2978


def test_feature_noise_is_shared_and_skips_constant_coordinates():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(NUM_NODES, 5, generator=g)
    x[:, 2] = 0.37  # constant coordinate
    column = x[:, 2:3].contiguous()  # a contiguous reduction leaves a float residue in the raw SD
    assert column.std(dim=0, unbiased=False) != 0
    assert feature_sigma(column) == 0 and feature_sigma(x)[2] == 0
    views = controlled_views(x, _edge_index(), [0.0, 0.1, 0.2], generator=torch.Generator().manual_seed(5))

    assert set(views) == {CONTROL} | {(f, a) for f in FAMILIES for a in (0.1, 0.2)}
    control = views[CONTROL]
    assert torch.equal(control.x, x)
    for alpha in (0.1, 0.2):
        feature, structure, joint = (views[(f, alpha)] for f in FAMILIES)
        assert torch.equal(feature.x[:, 2], x[:, 2])
        assert not torch.equal(feature.x, x)
        assert torch.equal(feature.edge_index, control.edge_index) and feature.eta == 0 and feature.swaps == 0
        assert torch.equal(structure.x, x) and structure.eta > 0
        assert torch.equal(joint.x, feature.x) and torch.equal(joint.edge_index, structure.edge_index)
        assert joint.swaps == structure.swaps == swap_budget(alpha, control.edge_index.size(1) // 2)
    # One noise draw: the 0.2 displacement is twice the 0.1 displacement.
    torch.testing.assert_close(views[("feature", 0.2)].x - x, 2 * (views[("feature", 0.1)].x - x))


def test_views_are_symmetric_loop_free_and_deterministic():
    x = torch.randn(NUM_NODES, 3)
    first = controlled_views(x, _edge_index(), [0.3], generator=torch.Generator().manual_seed(6))
    again = controlled_views(x, _edge_index(), [0.3], generator=torch.Generator().manual_seed(6))
    for key, view in first.items():
        ei = view.edge_index
        assert bool((ei[0] != ei[1]).all())
        assert set(map(tuple, ei.t().tolist())) == set(map(tuple, ei.flip(0).t().tolist()))
        assert 2 * simple_undirected_edges(ei).size(0) == ei.size(1)
        assert torch.equal(view.x, again[key].x) and torch.equal(ei, again[key].edge_index)
