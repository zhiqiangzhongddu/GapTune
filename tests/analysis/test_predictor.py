"""Checkpoint resolution and output layout of the fixed-predictor studies (App. C.5, C.6)."""

from __future__ import annotations

import os

import pytest
import torch

from src.analysis.predictor import finetuned_checkpoints, study_dir, typed_prompts
from src.config import cfg as base_cfg


def test_finetuned_checkpoints_follow_the_standard_seeds(tmp_path):
    cfg = base_cfg.clone()
    cfg.seeds = [42, 0, 100]
    cfg.finetune.num_runs = 2
    for seed in (42, 0):
        (tmp_path / f"ft_seed{seed}.pt").touch()
    cfg.analysis.finetuned_checkpoint = str(tmp_path / "ft_seed{seed}.pt")
    assert finetuned_checkpoints(cfg) == [(42, str(tmp_path / "ft_seed42.pt")), (0, str(tmp_path / "ft_seed0.pt"))]
    cfg.finetune.num_runs = 3
    with pytest.raises(FileNotFoundError, match="ft_seed100.pt"):
        finetuned_checkpoints(cfg)
    cfg.analysis.finetuned_checkpoint = str(tmp_path / "ft_seed42.pt")
    with pytest.raises(ValueError, match="placeholder"):
        finetuned_checkpoints(cfg)
    cfg.finetune.num_runs = 1
    # A single explicit file is labelled with its own training seed, not cfg.seeds[0].
    torch.save({"cfg": {"seed": 0}}, tmp_path / "ft_seed0.pt")
    cfg.analysis.finetuned_checkpoint = str(tmp_path / "ft_seed0.pt")
    assert finetuned_checkpoints(cfg) == [(0, str(tmp_path / "ft_seed0.pt"))]


def test_study_dir_drops_the_seed_from_the_run_tag(tmp_path):
    cfg = base_cfg.clone()
    cfg.analysis.output_dir = str(tmp_path)
    path = study_dir(cfg, "rotation", "/ckpt/photo/ft_gaptune_plus1_to_photo_e20_bs32_seed42.pt")
    assert path == os.path.join(str(tmp_path), "rotation", "ft_gaptune_plus1_to_photo_e20_bs32")
    assert os.path.isdir(path)


def test_typed_prompts_key_active_types():
    node, message = torch.zeros(3, 2), torch.ones(4, 2)
    assert list(typed_prompts(node, [message, None, message])) == ["N", "M1", "M3"]
    assert list(typed_prompts(None, [message])) == ["M1"]
    assert list(typed_prompts(node, None)) == ["N"]
