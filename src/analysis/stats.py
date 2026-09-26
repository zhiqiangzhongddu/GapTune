"""Repetition-level summaries (paper App. A.5, C.7, C.8).

The sampling unit is a complete repetition. Paired differences ``D_s`` get
the pointwise t interval of Eq. 74 with a two-sided t-test p value, W/T/L
counts and the exact sign test; p values across a contrast family get the
Holm adjustment. App. A uses the mean within-repetition Spearman correlation
(Eq. 30) and a percentile bootstrap over complete repetitions.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy import stats


def paired_t(diffs: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    """Mean, sample SD (ddof 1), Eq. 74 interval and two-sided t-test p of paired differences."""
    d = np.asarray(diffs, dtype=float)
    n = d.size
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    half = stats.t.ppf(0.5 + confidence / 2, n - 1) * sd / math.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.float64(mean) / (sd / math.sqrt(n))
    p = float(2 * stats.t.sf(abs(t_stat), n - 1))
    return {"n": n, "mean": mean, "sd": sd, "low": mean - half, "high": mean + half, "p": p}


def win_tie_loss(diffs: Sequence[float]) -> tuple[int, int, int]:
    """Counts of positive, zero and negative differences."""
    d = np.asarray(diffs, dtype=float)
    return int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())


def sign_test(diffs: Sequence[float]) -> float:
    """Exact two-sided sign test (ties dropped); 5 wins of 5 gives 2 / 2^5."""
    wins, _, losses = win_tie_loss(diffs)
    if wins + losses == 0:
        return 1.0
    return float(stats.binomtest(wins, wins + losses, 0.5).pvalue)


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    steps = (p.size - np.arange(p.size)) * p[order]
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(steps))
    return adjusted


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman correlation with average ranks for ties; NaN (undefined) if either vector is constant."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1])


def mean_spearman(xs: Sequence[Sequence[float]], ys: Sequence[Sequence[float]]) -> float:
    """Eq. 30: mean of within-repetition correlations; undefined repetitions are excluded."""
    rhos = np.array([spearman(x, y) for x, y in zip(xs, ys)])
    rhos = rhos[~np.isnan(rhos)]
    return float(rhos.mean()) if rhos.size else float("nan")


def repetition_bootstrap(
    repetitions: Sequence,
    statistic: Callable[[list], float | np.ndarray],
    *,
    num_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval of ``statistic`` over resamples of complete repetitions (with replacement).

    Resamples where the statistic is undefined (NaN) are left out of the interval.
    """
    rng = np.random.default_rng(seed)
    n = len(repetitions)
    draws = np.array([
        statistic([repetitions[i] for i in rng.integers(n, size=n)]) for _ in range(num_samples)
    ])
    tail = 100 * (1 - confidence) / 2
    return np.nanpercentile(draws, tail, axis=0), np.nanpercentile(draws, 100 - tail, axis=0)
