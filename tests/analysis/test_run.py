"""Analysis CLI wiring: cfg.analysis defaults and the study registry."""

from __future__ import annotations

from src.analysis import run as analysis_run
from src.config import cfg as base_cfg
from src.utils.save_results import get_explicit_cfg_keys


def test_analysis_cfg_defaults():
    assert base_cfg.analysis.study == ""
    assert base_cfg.analysis.output_dir == "outputs/analysis"
    assert list(base_cfg.analysis.repetitions) == [42, 0, 100, 123, 2024, 1, 2, 3, 4, 5]


def test_unknown_or_missing_study_exits_with_a_clear_error(capsys):
    assert analysis_run.run_analysis_from_cli(["analysis.study", "nope"]) == 1
    assert "analysis.study must be one of" in capsys.readouterr().err
    assert analysis_run.run_analysis_from_cli([]) == 1
    assert "got ''" in capsys.readouterr().err


def test_unimplemented_study_exits_with_a_clear_error(monkeypatch, capsys):
    monkeypatch.setitem(analysis_run.STUDIES, "planned", None)
    assert analysis_run.run_analysis_from_cli(["analysis.study", "planned"]) == 1
    assert "not implemented" in capsys.readouterr().err


def test_registered_study_receives_the_merged_cfg(monkeypatch):
    received = []

    def study(cfg):
        received.append(cfg)
        return 7

    monkeypatch.setitem(analysis_run.STUDIES, "dummy", study)
    status = analysis_run.run_analysis_from_cli(
        ["analysis.study", "dummy", "analysis.repetitions", "[1, 2]", "analysis.node_budget", "16"]
    )
    assert status == 7
    (cfg,) = received
    assert list(cfg.analysis.repetitions) == [1, 2] and cfg.analysis.node_budget == 16
    assert "analysis.node_budget" in get_explicit_cfg_keys(cfg)
    assert base_cfg.analysis.study == ""  # the shared default cfg is untouched
