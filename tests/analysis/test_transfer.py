"""App. A transfer study on a tiny synthetic graph (random GCN weights, CPU).

- one repetition runs end to end: a row per view with accuracies in [0, 1],
  ``R = 100 (A_match - A_transfer)`` and ``E = R - R_0``; the control's ``E``
  and context gaps are exactly 0;
- a repetition is reproducible for a fixed seed;
- the App. A.3 protocol: the donor prompt stays frozen, every target prompt is
  fresh, both fresh heads share one initialisation above frozen prompts, and
  changing the evaluation labels changes no fit and no context gap;
- views, context-gap samples and initialisation use separate random streams;
- the summary: curve means of ``E``, mean within-repetition Spearman (Eq. 30)
  over the defined repetitions, pooled within-family correlations, bootstrap
  intervals around the point;
- the study is registered in the analysis CLI.
"""

from __future__ import annotations

import copy
import math

import torch
from scipy.stats import spearmanr
from torch_geometric.data import Data
from yacs.config import CfgNode as CN

from src.analysis import run as analysis_run
from src.analysis import transfer
from src.analysis.perturbations import CONTROL, FAMILIES
from src.analysis.transfer import _fit, _frozen_encoders, run_repetition, run_transfer, summarize
from src.config import set_cfg
from src.model import build_encoder_from_cfg

IN_DIM, NUM_NODES, NUM_CLASSES = 6, 40, 3
STRENGTHS = [0.1, 0.3]


def _cfg():
    cfg = CN()
    set_cfg(cfg)
    cfg.model.in_dim, cfg.model.hidden_dim, cfg.model.out_dim, cfg.model.num_layers = IN_DIM, 8, 8, 2
    ds = cfg.finetune.dataset
    ds.task_level, ds.induced, ds.task_type, ds.num_classes = "node", False, "classification", NUM_CLASSES
    cfg.finetune.epochs = 3
    cfg.analysis.transfer.strengths = STRENGTHS
    cfg.analysis.node_budget, cfg.analysis.message_budget = 16, 32
    return cfg


def _setup():
    cfg = _cfg()
    torch.manual_seed(0)
    model, encoder = _frozen_encoders(cfg, build_encoder_from_cfg(cfg, IN_DIM).state_dict(), torch.device("cpu"))
    g = torch.Generator().manual_seed(0)
    ei = torch.randint(NUM_NODES, (2, 80), generator=g)
    y = torch.arange(NUM_NODES) % NUM_CLASSES
    train = torch.zeros(NUM_NODES, dtype=torch.bool)
    train[: 2 * NUM_CLASSES] = True  # two support nodes per class, no validation
    data = Data(
        x=torch.randn(NUM_NODES, IN_DIM, generator=g),
        edge_index=torch.cat([ei, ei.flip(0)], dim=1),
        y=y,
        train_mask=train,
        val_mask=torch.zeros_like(train),
        test_mask=~train,
    )
    return cfg, model, encoder, data


def test_repetition_runs_end_to_end_with_a_zero_control():
    cfg, model, encoder, data = _setup()
    rows = run_repetition(cfg, model, encoder, data, seed=0)
    assert [(r["family"], r["alpha"]) for r in rows] == [CONTROL] + [
        (family, alpha) for alpha in STRENGTHS for family in FAMILIES
    ]
    control = rows[0]
    assert control["E"] == 0.0 and control["delta_H"] == 0.0 and control["delta_M"] == 0.0
    for r in rows:
        assert 0.0 <= r["acc_match"] <= 1.0 and 0.0 <= r["acc_transfer"] <= 1.0
        assert r["R"] == 100 * (r["acc_match"] - r["acc_transfer"])
        assert r["E"] == r["R"] - control["R"]
        assert (r["swaps"] > 0) == (r["family"] != "feature" and r != control)
        assert math.isfinite(r["delta_H"]) and math.isfinite(r["delta_M"])
    assert all(r["delta_M"] > 0 for r in rows if r["family"] in ("structure", "joint"))


def test_repetition_is_reproducible_for_a_fixed_seed():
    cfg, model, encoder, data = _setup()
    assert run_repetition(cfg, model, encoder, data, seed=3) == run_repetition(cfg, model, encoder, data, seed=3)


def _logged_fits(monkeypatch, data):
    """One repetition's rows and ``(task, start state, end state, prompt trainable)`` of every fit."""
    fits = []

    def logged(task, model, graph, monitor):
        start = copy.deepcopy(task.state_dict())
        trainable = all(p.requires_grad for p in task.prompt.parameters())
        _fit(task, model, graph, monitor)
        fits.append((task, start, copy.deepcopy(task.state_dict()), trainable))

    monkeypatch.setattr(transfer, "_fit", logged)
    cfg, model, encoder, _ = _setup()
    return run_repetition(cfg, model, encoder, data, seed=0), fits


def _part(state, prefix):
    return {k: v for k, v in state.items() if k.startswith(prefix)}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_repetition_follows_the_app_a3_protocol(monkeypatch):
    data = _setup()[3]
    rows, fits = _logged_fits(monkeypatch, data)
    assert len(fits) == 1 + 3 * len(rows)  # donor, then per view: target prompt, matched head, transfer head
    donor, _, donor_end, donor_trainable = fits[0]
    trained = [_part(donor_end, "prompt.")]
    assert donor_trainable
    for v in range(len(rows)):
        (target, start, end, trainable), match, other = fits[1 + 3 * v : 4 + 3 * v]
        assert trainable and target is not donor and match[0] is target and other[0] is donor
        assert not any(_same(_part(start, "prompt."), p) for p in trained)  # a fresh prompt, never a trained one
        trained.append(_part(end, "prompt."))
        # both heads fit above frozen prompts (the donor's never changes) from one shared initialisation
        assert not match[3] and not other[3]
        assert _same(_part(match[1], "prompt."), trained[-1]) and _same(_part(match[2], "prompt."), trained[-1])
        assert _same(_part(other[1], "prompt."), trained[0]) and _same(_part(other[2], "prompt."), trained[0])
        assert _same(_part(match[1], "classifier."), _part(other[1], "classifier."))

    # evaluation labels never reach a fit or the context gaps
    relabelled = data.clone()
    relabelled.y = torch.where(data.test_mask, (data.y + 1) % NUM_CLASSES, data.y)
    rows_b, fits_b = _logged_fits(monkeypatch, relabelled)
    assert all(_same(a[1], b[1]) and _same(a[2], b[2]) for a, b in zip(fits, fits_b, strict=True))
    scores = ("acc_match", "acc_transfer", "R", "E")
    assert [{k: r[k] for k in r if k not in scores} for r in rows] == [
        {k: r[k] for k in r if k not in scores} for r in rows_b
    ]


def test_views_gaps_and_initialisation_use_separate_streams(monkeypatch):
    cfg, model, encoder, data = _setup()
    seeds = []

    def recorded(fn):
        def wrapper(*args, generator, **kwargs):
            seeds.append(generator.initial_seed())
            return fn(*args, generator=generator, **kwargs)

        return wrapper

    monkeypatch.setattr(transfer, "controlled_views", recorded(transfer.controlled_views))
    monkeypatch.setattr(transfer, "view_context_gaps", recorded(transfer.view_context_gaps))
    monkeypatch.setattr(transfer, "set_seed", seeds.append)
    monkeypatch.setattr(transfer, "_fit", lambda *args: None)  # the streams only; no training
    repetitions = [int(s) for s in cfg.analysis.repetitions]
    for seed in repetitions:
        run_repetition(cfg, model, encoder, data, seed)
    # no two roles or repetitions share a stream, and none is the stream of a split made from a repetition seed
    assert len(set(seeds)) == 3 * len(repetitions) and not set(seeds) & set(repetitions)


def _synthetic_rows():
    rows = []
    for seed, offset in ((0, 0.0), (1, 1.0), (2, -2.0)):
        rows.append({"seed": seed, "family": CONTROL[0], "alpha": 0.0, "eta": 0.0,
                     "delta_H": 0.0, "delta_M": 0.0, "E": 0.0})
        for i, (family, alpha) in enumerate((f, a) for a in STRENGTHS for f in FAMILIES):
            delta = (i + 1) * (1 + seed)
            rows.append({"seed": seed, "family": family, "alpha": alpha, "eta": alpha,
                         "delta_H": delta, "delta_M": -delta, "E": 2.0 * i + offset})
    return rows


def test_summary_curve_correlations_and_bootstrap():
    rows = _synthetic_rows()
    curve, correlations = summarize(rows, num_samples=200)
    assert [(c["family"], c["alpha"]) for c in curve] == [(f, a) for a in STRENGTHS for f in FAMILIES]
    for i, c in enumerate(curve):
        assert math.isclose(c["E_mean"], 2.0 * i - 1.0 / 3.0) and c["n"] == 3
        assert c["E_low"] <= c["E_mean"] <= c["E_high"]
    by_key = {(c["scope"], c["q"]): c for c in correlations}
    within_h, within_m = by_key[("within_repetition", "H")], by_key[("within_repetition", "M")]
    assert math.isclose(within_h["rho"], 1.0) and math.isclose(within_m["rho"], -1.0)  # monotone in every repetition
    assert math.isclose(within_h["low"], 1.0) and math.isclose(within_h["high"], 1.0)
    assert within_h["n"] == within_m["n"] == 3
    assert {scope for scope, _ in by_key} == {"within_repetition"} | {f"pooled_{f}" for f in FAMILIES}
    for family in FAMILIES:
        pooled = [r for r in rows if r["family"] == family]
        for q in ("H", "M"):
            assert by_key[(f"pooled_{family}", q)]["n"] == 3 * len(STRENGTHS)
            expected = spearmanr([r[f"delta_{q}"] for r in pooled], [r["E"] for r in pooled])[0]
            assert math.isclose(by_key[(f"pooled_{family}", q)]["rho"], expected)


def test_summary_counts_only_defined_repetitions():
    rows = _synthetic_rows()
    for r in rows:
        if r["seed"] == 2:
            r["E"] = 0.0  # constant E: this repetition's Spearman is undefined (Eq. 30 excludes it)
    _, correlations = summarize(rows, num_samples=200)
    within = [c for c in correlations if c["scope"] == "within_repetition"]
    assert [c["q"] for c in within] == ["H", "M"]
    for c, rho in zip(within, (1.0, -1.0)):
        assert c["n"] == 2
        # resamples that draw only the undefined repetition are left out of the interval
        assert all(math.isclose(c[k], rho) for k in ("rho", "low", "high"))


def test_transfer_study_is_registered():
    assert analysis_run.STUDIES["transfer"] is run_transfer
