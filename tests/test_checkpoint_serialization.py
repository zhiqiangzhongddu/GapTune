from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from yacs.config import CfgNode as CN

from src.finetune.finetuner import FinetuneRunner
from src.utils.checkpoint import (
    canonical_config_sha256,
    cfg_to_dict,
    save_checkpoint,
    save_training_log,
)
from src.utils.provenance import CampaignProvenanceError


class CheckpointConfigSerializationTest(unittest.TestCase):
    def assert_plain_config(self, value) -> None:
        if type(value) is dict:
            for key, item in value.items():
                self.assertIs(type(key), str)
                self.assert_plain_config(item)
            return
        if type(value) is list:
            for item in value:
                self.assert_plain_config(item)
            return
        self.assertIn(type(value), {str, int, float, bool, type(None)})

    def test_yacs_checkpoint_and_log_configs_are_recursively_plain_and_safe(self):
        cfg = CN()
        cfg.model = CN()
        cfg.model.name = "gcn"
        cfg.model.widths = (16, 32)
        cfg.options = CN({"enabled": True, "dropout": 0.1})

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            log_path = Path(tmp) / "training.json"
            save_checkpoint(
                path=str(checkpoint_path),
                model=model,
                optimizer=optimizer,
                epoch=3,
                cfg=cfg,
                dataset_meta={"name": "toy"},
                metrics={"test_acc": 0.5},
                extra={"task_state": model.state_dict()},
            )
            save_training_log(
                path=str(log_path),
                cfg=cfg,
                dataset_meta={"name": "toy"},
                history=[],
                best_info={"epoch": 3, "metrics": {"test_acc": 0.5}},
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            log = json.loads(log_path.read_text(encoding="utf-8"))

        self.assert_plain_config(checkpoint["cfg"])
        self.assert_plain_config(log["config"])
        self.assertEqual(checkpoint["cfg"], log["config"])
        self.assertEqual(checkpoint["cfg"]["model"]["widths"], [16, 32])

    def test_plain_dict_is_copied_and_unsupported_values_fail_with_path(self):
        cfg = {"model": {"name": "gcn"}, "seeds": [7, 17]}
        copied = cfg_to_dict(cfg)
        self.assertEqual(copied, cfg)
        self.assertIsNot(copied, cfg)
        self.assertIsNot(copied["model"], cfg["model"])

        with self.assertRaisesRegex(
            TypeError,
            r"Unsupported configuration value at cfg\.model\.factory",
        ):
            cfg_to_dict({"model": {"factory": object()}})

    def test_canonical_config_digest_and_publication_override_binding(self):
        saved = {
            "model": {"name": "gcn", "hidden_dim": 64, "graph_pooling": "mean"},
            "pretrain": {
                "method": "attr_masking",
                "dataset": {"name": "photo", "induced": True},
            },
        }
        self.assertEqual(
            canonical_config_sha256(saved),
            canonical_config_sha256(CN(saved)),
        )

        cfg = CN()
        cfg.model = CN({"name": "gcn", "hidden_dim": 64, "graph_pooling": "max"})
        cfg.pretrain = CN({
            "method": "attr_masking",
            "dataset": CN({"name": "photo", "induced": True}),
        })
        cfg.provenance = CN({
            "pretrained_checkpoint_config_sha256": canonical_config_sha256(saved)
        })
        cfg.save_results = CN({
            "explicit_keys": [
                "model.name",
                "model.hidden_dim",
                "model.graph_pooling",
                "pretrain.method",
                "pretrain.dataset.name",
                "pretrain.dataset.induced",
            ]
        })
        runner = FinetuneRunner.__new__(FinetuneRunner)
        runner.cfg = cfg
        runner.pretrain_cfg = saved
        runner._validate_loaded_pretrain_config()

        cfg.model.hidden_dim = 32
        with self.assertRaisesRegex(CampaignProvenanceError, "model.hidden_dim"):
            runner._validate_loaded_pretrain_config()
        cfg.model.hidden_dim = 64

        del saved["pretrain"]["dataset"]["induced"]
        cfg.provenance.pretrained_checkpoint_config_sha256 = canonical_config_sha256(saved)
        with self.assertRaisesRegex(CampaignProvenanceError, "missing.*induced"):
            runner._validate_loaded_pretrain_config()

        cfg.provenance.pretrained_checkpoint_config_sha256 = "0" * 64
        with self.assertRaisesRegex(CampaignProvenanceError, "config digest mismatch"):
            runner._validate_loaded_pretrain_config()

    def test_finetune_checkpoint_loader_is_safe_only_for_publication(self):
        runner = FinetuneRunner.__new__(FinetuneRunner)
        runner.cfg = SimpleNamespace(
            provenance=SimpleNamespace(publication=True),
        )
        with patch(
            "src.finetune.finetuner.torch.load",
            return_value={},
        ) as load:
            runner._load_finetune_checkpoint("publication.pt", map_location="cpu")
        load.assert_called_once_with(
            "publication.pt",
            map_location="cpu",
            weights_only=True,
        )

        runner.cfg.provenance.publication = False
        with patch(
            "src.finetune.finetuner.torch.load",
            return_value={},
        ) as load:
            runner._load_finetune_checkpoint("development.pt", map_location="cpu")
        load.assert_called_once_with("development.pt", map_location="cpu")


if __name__ == "__main__":
    unittest.main()
