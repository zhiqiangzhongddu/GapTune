"""FinetuneRunner pads featureless graph targets only for wider checkpoints.

Cross-dataset transfer onto QM7b (e.g. ZINC -> QM7b) restores in_dim=100 from
the checkpoint, so the 1-dim degree feature must be zero-padded. Same-dataset
QM7b checkpoints (in_dim=1) and node/edge targets must not request padding.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.finetuner import FinetuneRunner
from src.utils.checkpoint import cfg_to_dict


class _StopSetup(Exception):
    pass


def _pad_flag(target: str, task_level: str, ckpt_in_dim: int):
    captured = {}

    def fake_create_dataset(**kwargs):
        captured.update(kwargs)
        raise _StopSetup()

    with tempfile.TemporaryDirectory() as tmp:
        pre = set_cfg(CN())
        pre.model.name = "gcn"
        pre.model.in_dim = ckpt_in_dim
        path = os.path.join(tmp, "pre.pt")
        torch.save({"model_state": {}, "cfg": cfg_to_dict(pre), "dataset": {}}, path)

        cfg = set_cfg(CN())
        cfg.seed = 42
        cfg.model.in_dim = 0  # restored from the checkpoint cfg
        cfg.finetune.dataset.name = target
        cfg.finetune.dataset.task_level = task_level
        cfg.finetune.dataset.induced = task_level != "graph"
        cfg.finetune.checkpoint_dir = os.path.join(tmp, "ft")
        cfg.finetune.log_dir = os.path.join(tmp, "logs")
        runner = FinetuneRunner(cfg, path, "pre")
        with patch("src.finetune.finetuner.create_dataset", side_effect=fake_create_dataset):
            try:
                runner._setup()
            except _StopSetup:
                pass
    # None (not False) if _setup never reached create_dataset.
    return captured.get("pad_featureless_features")


class FeaturelessPaddingRuleTest(unittest.TestCase):
    def test_cross_dataset_graph_target_is_padded(self):
        self.assertIs(_pad_flag("qm7b", "graph", 100), True)

    def test_same_dataset_qm7b_checkpoint_is_not_padded(self):
        self.assertIs(_pad_flag("qm7b", "graph", 1), False)

    def test_node_target_is_not_padded(self):
        self.assertIs(_pad_flag("photo", "node", 100), False)

    def test_edge_target_is_not_padded(self):
        self.assertIs(_pad_flag("cornell", "edge", 100), False)


if __name__ == "__main__":
    unittest.main()
