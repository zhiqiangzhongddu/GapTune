"""Repetition-level statistics tests (paper Eq. 30, Eq. 74, Tables 10-11).

- paired t interval and p value against scipy.stats, and a Table 10 row;
- W/T/L counts and the exact sign test (5/0/0 -> 0.0625);
- Holm against a hand-computed example and the Table 11 family;
- Spearman with average ranks for ties, undefined for constant vectors;
- bootstrap resamples complete repetitions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from src.analysis.stats import (
    holm,
    mean_spearman,
    paired_t,
    repetition_bootstrap,
    sign_test,
    spearman,
    win_tie_loss,
)


@pytest.mark.parametrize("n", [5, 10])
def test_paired_t_matches_scipy(n):
    d = np.random.default_rng(n).normal(1.0, 2.0, size=n)
    summary = paired_t(d)
    low, high = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d))
    assert summary["n"] == n
    assert math.isclose(summary["low"], low) and math.isclose(summary["high"], high)
    assert math.isclose(summary["sd"], d.std(ddof=1))
    assert math.isclose(summary["p"], stats.ttest_1samp(d, 0.0).pvalue)


def test_paired_t_reproduces_table_10_and_11_photo_free_row():
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    d = 3.23 + 2.15 * z / z.std(ddof=1)  # mean 3.23, paired SD 2.15, n = 5
    summary = paired_t(d)
    assert (round(summary["low"], 2), round(summary["high"], 2)) == (0.56, 5.90)
    assert round(summary["p"], 4) == 0.0283


def test_paired_t_degenerate_differences():
    assert paired_t([1.0] * 5)["p"] == 0.0
    summary = paired_t([0.0] * 5)
    assert (summary["low"], summary["high"]) == (0.0, 0.0) and math.isnan(summary["p"])


def test_win_tie_loss_and_exact_sign_test():
    assert win_tie_loss([1.0, 0.0, -2.0, 3.0]) == (2, 1, 1)
    assert sign_test([0.5] * 5) == 2 / 2**5
    assert math.isclose(sign_test([1, 1, 1, 1, -1]), 2 * (1 + 5) / 2**5)
    assert sign_test([0.0, 0.0]) == 1.0


def test_holm_hand_computed_example():
    # sorted: 0.005*4 = 0.02, 0.01*3 = 0.03, 0.03*2 = 0.06, max(0.04*1, 0.06) = 0.06
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(holm([0.2, 0.9]), [0.4, 0.9])


def test_holm_reproduces_table_11():
    p = [0.0283, 0.0104, 0.0240, 0.1371, 0.0240, 0.0960, 0.5625, 0.0024, 0.0141, 0.9096, 0.2058, 0.0452]
    p_holm = [0.2157, 0.1141, 0.2157, 0.5485, 0.2157, 0.4800, 1.0, 0.0285, 0.1413, 1.0, 0.6174, 0.2715]
    np.testing.assert_allclose(holm(p), p_holm, atol=1e-3)  # table p values are rounded


def test_spearman_average_ranks_and_undefined_constant():
    x, y = [1, 2, 2, 3, 5], [2, 1, 4, 4, 6]
    assert math.isclose(spearman(x, y), stats.spearmanr(x, y).statistic)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert math.isnan(spearman([1, 2, 3], [4, 4, 4]))
    rho = spearman([1, 2, 3], [1, 3, 2])
    assert mean_spearman([[1, 2, 3], [1, 2, 3]], [[1, 3, 2], [7, 7, 7]]) == rho  # undefined rep excluded
    assert math.isnan(mean_spearman([[1, 2]], [[3, 3]]))


def test_bootstrap_resamples_complete_repetitions():
    reps = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
    seen = []

    def curve_means(sample):
        seen.extend(sample)
        return np.mean(np.array(sample), axis=0)

    low, high = repetition_bootstrap(reps, curve_means, num_samples=200, seed=1)
    assert set(seen) <= set(reps)  # repetitions are kept intact
    assert low.shape == high.shape == (2,)
    np.testing.assert_allclose(high - low, (high - low)[0])  # paired columns move together
    assert bool((low <= [2.0, 3.0]).all() and ([2.0, 3.0] <= high).all())
    again = repetition_bootstrap(reps, lambda s: np.mean(np.array(s), axis=0), num_samples=200, seed=1)
    np.testing.assert_array_equal(again[0], low)
    constant = repetition_bootstrap([1.5] * 4, np.mean, num_samples=50)
    assert constant == (1.5, 1.5)
