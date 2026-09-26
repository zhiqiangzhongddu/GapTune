from __future__ import annotations

import csv
import json
import os
import pickle
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from yacs.config import CfgNode as CN

from src.config import set_cfg, update_cfg
from src.data_loader.induced_graphs import InducedGraphDataset, SingleGraphDataLoader
from src.data_loader.split_provenance import (
    build_materialized_split_manifest,
    validate_materialized_split_manifest,
)
from src.finetune.finetuner import FinetuneRunner
from src.utils.provenance import (
    CampaignProvenanceError,
    aggregate_run_provenance,
    bind_pretrained_checkpoint_provenance,
    campaign_provenance_payload,
    compute_file_sha256,
    compute_source_tree_snapshot,
    prepare_campaign_provenance,
    saved_provenance_matches,
)
from src.utils.save_results import append_workflow_result


def _cfg():
    return set_cfg(CN())


def _graph(value: float, label: int) -> Data:
    return Data(
        x=torch.tensor([[value], [value + 1.0]], dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        y=torch.tensor([label], dtype=torch.long),
    )


def _write_unsafe_checkpoint_marker(path: str):
    Path(path).write_text("executed\n", encoding="utf-8")
    return {"executed": True}


class _UnsafeCheckpointValue:
    def __init__(self, marker_path: Path):
        self.marker_path = str(marker_path)

    def __reduce__(self):
        return _write_unsafe_checkpoint_marker, (self.marker_path,)


class SourceTreeProvenanceTest(unittest.TestCase):
    def test_source_digest_covers_tracked_and_untracked_code_but_not_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "outputs").mkdir()
            (root / "data").mkdir()
            (root / "slurm" / "output").mkdir(parents=True)
            (root / "src" / "tracked.py").write_text("TRACKED = 1\n", encoding="utf-8")
            (root / "tests" / "untracked.py").write_text("def test_x(): pass\n", encoding="utf-8")
            (root / "outputs" / "result.tsv").write_text("metric\n", encoding="utf-8")
            (root / "data" / "dataset.pt").write_bytes(b"data")
            (root / "slurm" / "output" / "job.out").write_text("running\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "src/tracked.py"], cwd=root, check=True)

            first = compute_source_tree_snapshot(root)
            second = compute_source_tree_snapshot(root)
            self.assertEqual(first, second)
            self.assertEqual(first.paths, ("src/tracked.py", "tests/untracked.py"))

            (root / "outputs" / "result.tsv").write_text("changed\n", encoding="utf-8")
            (root / "data" / "dataset.pt").write_bytes(b"changed")
            (root / "slurm" / "output" / "job.out").write_text("finished\n", encoding="utf-8")
            self.assertEqual(first.digest, compute_source_tree_snapshot(root).digest)

            (root / "slurm" / "new_launcher.slurm").write_text("#!/bin/bash\n", encoding="utf-8")
            self.assertNotEqual(first.digest, compute_source_tree_snapshot(root).digest)
            (root / "slurm" / "new_launcher.slurm").unlink()

            (root / "tests" / "untracked.py").write_text("def test_y(): pass\n", encoding="utf-8")
            self.assertNotEqual(first.digest, compute_source_tree_snapshot(root).digest)

            before_txt = compute_source_tree_snapshot(root)
            (root / "src" / "frozen_args.txt").write_text("epochs=200\n", encoding="utf-8")
            after_txt = compute_source_tree_snapshot(root)
            self.assertIn("src/frozen_args.txt", after_txt.paths)
            self.assertNotEqual(before_txt.digest, after_txt.digest)

    def test_development_defaults_are_inert(self):
        cfg = _cfg()
        with patch(
            "src.utils.provenance.compute_source_tree_snapshot",
            side_effect=AssertionError("must not hash development checkout"),
        ), patch(
            "src.utils.provenance.runtime_version_payload",
            side_effect=AssertionError("must not query development versions"),
        ), patch(
            "src.utils.provenance.compute_file_sha256",
            side_effect=AssertionError("must not hash development files"),
        ):
            self.assertEqual(prepare_campaign_provenance(cfg), {})
            self.assertEqual(campaign_provenance_payload(cfg), {})

    def test_publication_requires_and_validates_all_identity_fields(self):
        cfg = _cfg()
        cfg.provenance.publication = True
        with self.assertRaisesRegex(CampaignProvenanceError, "campaign_id"):
            prepare_campaign_provenance(cfg)

        cfg.provenance.campaign_id = "fresh-f5-f100-v1"
        cfg.provenance.source_tree_digest = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "outputs").mkdir()
            (root / "src" / "method.py").write_text("VALUE = 1\n", encoding="utf-8")
            manifest = root / "outputs" / "campaign.md"
            manifest.write_text("frozen design\n", encoding="utf-8")
            cfg.provenance.design_manifest_path = str(manifest)
            cfg.provenance.design_manifest_digest = compute_file_sha256(manifest)
            with self.assertRaisesRegex(CampaignProvenanceError, "mismatch"):
                prepare_campaign_provenance(cfg, root=root)

            snapshot = compute_source_tree_snapshot(root)
            cfg.provenance.source_tree_digest = snapshot.digest
            payload = prepare_campaign_provenance(cfg, root=root)

        self.assertTrue(payload["publication"])
        self.assertEqual(payload["design_manifest_path"], str(manifest.resolve()))
        self.assertEqual(payload["source_tree_digest"], snapshot.digest)
        self.assertEqual(payload["source_tree_path_count"], 1)
        self.assertEqual(payload["source_tree_paths"], ["src/method.py"])
        self.assertEqual(set(payload["runtime_versions"]), {"python", "pytorch", "pyg", "cuda"})

    def test_design_manifest_must_exist_match_and_live_outside_source_roots(self):
        cfg = _cfg()
        cfg.provenance.publication = True
        cfg.provenance.campaign_id = "campaign-a"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "outputs").mkdir()
            source_file = root / "src" / "method.py"
            source_file.write_text("VALUE = 1\n", encoding="utf-8")
            snapshot = compute_source_tree_snapshot(root)
            cfg.provenance.source_tree_digest = snapshot.digest

            cfg.provenance.design_manifest_path = str(root / "missing.md")
            cfg.provenance.design_manifest_digest = "0" * 64
            with self.assertRaisesRegex(CampaignProvenanceError, "existing file"):
                prepare_campaign_provenance(cfg, root=root)

            inside = root / "src" / "design.txt"
            inside.write_text("design\n", encoding="utf-8")
            cfg.provenance.design_manifest_path = str(inside)
            cfg.provenance.design_manifest_digest = compute_file_sha256(inside)
            cfg.provenance.source_tree_digest = compute_source_tree_snapshot(root).digest
            with self.assertRaisesRegex(CampaignProvenanceError, "outside"):
                prepare_campaign_provenance(cfg, root=root)

            outside = root / "outputs" / "design.md"
            outside.write_text("design\n", encoding="utf-8")
            cfg.provenance.design_manifest_path = os.path.join(
                str(root), "outputs", "..", "outputs", "design.md"
            )
            cfg.provenance.design_manifest_digest = "f" * 64
            with self.assertRaisesRegex(CampaignProvenanceError, "design-manifest digest"):
                prepare_campaign_provenance(cfg, root=root)

            cfg.provenance.design_manifest_digest = compute_file_sha256(outside)
            payload = prepare_campaign_provenance(cfg, root=root)
            self.assertEqual(payload["design_manifest_path"], str(outside.resolve()))

    def test_pretrained_checkpoint_binding_is_exact_and_development_is_inert(self):
        cfg = _cfg()
        with patch(
            "src.utils.provenance.compute_file_sha256",
            side_effect=AssertionError("development must not hash"),
        ):
            self.assertEqual(
                bind_pretrained_checkpoint_provenance(cfg, "relative.pt"),
                "relative.pt",
            )

        cfg.provenance.publication = True
        cfg.provenance.pretrained_checkpoint_config_sha256 = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "pretrained.pt"
            checkpoint.write_bytes(b"checkpoint-v1")
            normalized = bind_pretrained_checkpoint_provenance(cfg, checkpoint)
            self.assertEqual(normalized, str(checkpoint.resolve()))
            self.assertEqual(
                cfg.provenance.pretrained_checkpoint_sha256,
                compute_file_sha256(checkpoint),
            )
            checkpoint.write_bytes(b"checkpoint-v2")
            with self.assertRaisesRegex(CampaignProvenanceError, "digest mismatch"):
                bind_pretrained_checkpoint_provenance(cfg, checkpoint)

    def test_publication_bool_rejects_unknown_text(self):
        cfg = _cfg()
        cfg.provenance.publication = "definitely"
        with self.assertRaisesRegex(CampaignProvenanceError, "must be a boolean"):
            prepare_campaign_provenance(cfg)

    def test_cli_accepts_campaign_fields(self):
        resolved = update_cfg(
            _cfg(),
            [
                "provenance.publication", "True",
                "provenance.campaign_id", "campaign-a",
                "provenance.design_manifest_digest", "a" * 64,
                "provenance.source_tree_digest", "b" * 64,
                "provenance.pretrained_checkpoint_config_sha256", "c" * 64,
            ],
        )
        self.assertTrue(resolved.provenance.publication)
        self.assertEqual(resolved.provenance.campaign_id, "campaign-a")


class MaterializedSplitDigestTest(unittest.TestCase):
    def _loaders(self, graphs, *, batch_size: int, shuffle: bool):
        train = DataLoader(Subset(graphs, [2, 0]), batch_size=batch_size, shuffle=shuffle)
        val = DataLoader(Subset(graphs, [1]), batch_size=batch_size, shuffle=False)
        test = DataLoader(Subset(graphs, []), batch_size=batch_size, shuffle=False)
        return train, val, test

    def test_graph_split_digest_is_batch_worker_and_shuffle_independent(self):
        graphs = [_graph(1.0, 0), _graph(3.0, 1), _graph(5.0, 0)]
        first = build_materialized_split_manifest(
            *self._loaders(graphs, batch_size=1, shuffle=True)
        )
        second = build_materialized_split_manifest(
            *self._loaders(graphs, batch_size=3, shuffle=False)
        )
        self.assertEqual(first, second)
        self.assertEqual(first["train"]["num_samples"], 2)
        self.assertEqual(first["test"]["num_samples"], 0)

    def test_graph_digest_changes_with_order_content_or_label(self):
        graphs = [_graph(1.0, 0), _graph(3.0, 1), _graph(5.0, 0)]
        baseline = build_materialized_split_manifest(
            *self._loaders(graphs, batch_size=2, shuffle=False)
        )

        changed = [graph.clone() for graph in graphs]
        changed[0].y = torch.tensor([1])
        changed_manifest = build_materialized_split_manifest(
            *self._loaders(changed, batch_size=2, shuffle=False)
        )
        self.assertNotEqual(
            baseline["train"]["sha256"], changed_manifest["train"]["sha256"]
        )
        self.assertEqual(
            baseline["val"]["sha256"], changed_manifest["val"]["sha256"]
        )

        reverse_train = DataLoader(Subset(graphs, [0, 2]), batch_size=2, shuffle=False)
        ordered = build_materialized_split_manifest(
            reverse_train,
            *self._loaders(graphs, batch_size=2, shuffle=False)[1:],
        )
        self.assertNotEqual(baseline["train"]["sha256"], ordered["train"]["sha256"])

    def test_node_digest_uses_only_the_relevant_split_labels_and_mask(self):
        data = Data(
            x=torch.arange(8, dtype=torch.float32).view(4, 2),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            y=torch.tensor([0, 1, 0, 1]),
            train_mask=torch.tensor([True, False, False, False]),
            val_mask=torch.tensor([False, True, False, False]),
            test_mask=torch.tensor([False, False, True, True]),
        )
        loaders = tuple(SingleGraphDataLoader(data) for _ in range(3))
        baseline = build_materialized_split_manifest(*loaders)

        changed = data.clone()
        changed.y[3] = 0
        changed_loaders = tuple(SingleGraphDataLoader(changed) for _ in range(3))
        altered = build_materialized_split_manifest(*changed_loaders)
        self.assertEqual(baseline["train"], altered["train"])
        self.assertEqual(baseline["val"], altered["val"])
        self.assertNotEqual(baseline["test"], altered["test"])

    def test_edge_materialization_hashes_queries_messages_and_labels(self):
        def edge_data(label: float):
            return Data(
                x=torch.arange(6, dtype=torch.float32).view(3, 2),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_label_index=torch.tensor([[0, 2], [2, 0]], dtype=torch.long),
                edge_label=torch.tensor([label, 0.0]),
            )

        baseline = build_materialized_split_manifest(
            SingleGraphDataLoader(edge_data(1.0)),
            SingleGraphDataLoader(edge_data(1.0)),
            SingleGraphDataLoader(edge_data(1.0)),
        )
        changed = build_materialized_split_manifest(
            SingleGraphDataLoader(edge_data(0.0)),
            SingleGraphDataLoader(edge_data(1.0)),
            SingleGraphDataLoader(edge_data(1.0)),
        )
        self.assertNotEqual(baseline["train"], changed["train"])
        self.assertEqual(baseline["val"], changed["val"])

    def test_induced_graph_dataset_and_absent_zero_val_are_supported(self):
        graphs = [_graph(1.0, 0), _graph(3.0, 1), _graph(5.0, 0)]
        induced = InducedGraphDataset(
            graphs,
            base_num_nodes=3,
            split_tags=["train", "val", "test"],
        )
        train = DataLoader(Subset(induced, [0]), batch_size=1, shuffle=False)
        val = DataLoader(Subset(induced, []), batch_size=1, shuffle=False)
        test = DataLoader(Subset(induced, [2]), batch_size=1, shuffle=False)
        manifest = build_materialized_split_manifest(train, val, test)
        self.assertEqual(manifest["val"]["num_samples"], 0)

        absent = build_materialized_split_manifest(train, None, None)
        self.assertEqual(absent["val"]["num_samples"], 0)
        self.assertEqual(absent["test"]["num_samples"], 0)

    def test_tensor_hashing_handles_bfloat_bool_empty_and_noncontiguous(self):
        noncontiguous = torch.arange(12, dtype=torch.float32).view(3, 4).t()
        self.assertFalse(noncontiguous.is_contiguous())
        graph = Data(
            x=noncontiguous,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=torch.tensor([1]),
            bf=torch.tensor([1.5, -2.0], dtype=torch.bfloat16),
            flags=torch.tensor([True, False], dtype=torch.bool),
            empty=torch.empty((0, 2), dtype=torch.float32),
        )
        loader = DataLoader([graph], batch_size=1, shuffle=False)
        first = build_materialized_split_manifest(loader, None, None)
        second = build_materialized_split_manifest(loader, None, None)
        self.assertEqual(first, second)

    def test_manifest_count_requires_a_non_boolean_nonnegative_integer(self):
        baseline = {
            name: {"sha256": digit * 64, "num_samples": 0}
            for name, digit in zip(("train", "val", "test"), ("1", "2", "3"))
        }
        validate_materialized_split_manifest(baseline)
        for invalid in (True, False, "0", 1.0, -1):
            with self.subTest(count=invalid):
                manifest = {name: dict(entry) for name, entry in baseline.items()}
                manifest["train"]["num_samples"] = invalid
                with self.assertRaises(ValueError):
                    validate_materialized_split_manifest(manifest)


class ProvenancePersistenceTest(unittest.TestCase):
    def _prepared_cfg(self):
        cfg = _cfg()
        cfg.provenance.publication = True
        cfg.provenance.campaign_id = "campaign-a"
        cfg.provenance.design_manifest_path = "/audit/design.md"
        cfg.provenance.design_manifest_digest = "1" * 64
        cfg.provenance.source_tree_digest = "2" * 64
        cfg.provenance.source_tree_path_count = 2
        cfg.provenance.source_tree_paths = ["src/a.py", "slurm/a.slurm"]
        cfg.provenance.pretrained_checkpoint_path = "/audit/pretrained.pt"
        cfg.provenance.pretrained_checkpoint_sha256 = "3" * 64
        cfg.provenance.pretrained_checkpoint_config_sha256 = "4" * 64
        return cfg

    def _runner(self, cfg, run_dir: str, *, run_name: str = "provenance-test"):
        runner = FinetuneRunner.__new__(FinetuneRunner)
        runner.cfg = cfg
        runner.split_content_digests = {
            name: {"sha256": digit * 64, "num_samples": 1}
            for name, digit in zip(("train", "val", "test"), ("4", "5", "6"))
        }
        runner.run_name = run_name
        runner.pretrained_run_name = "pretrain"
        runner.pretrained_checkpoint = str(cfg.provenance.pretrained_checkpoint_path)
        runner.pretrain_dataset_name = "source"
        runner.pretrain_task_level = "graph"
        runner.pretrain_method = "attr_masking"
        runner.pretrain_dataset_meta = {}
        runner.task = None
        runner.multilabel_target_stats = None
        runner.regression_target_stats = None
        runner.train_history = []
        runner.best_epoch = None
        runner.best_metric = float("nan")
        runner.monitor_name = "val_acc"
        runner.best_metrics = {}
        runner.dataset_meta = {"name": "target"}
        runner.run_group = "target"
        runner.run_dir = run_dir
        cfg.finetune.log_dir = ""
        return runner

    def test_checkpoint_extra_and_log_persist_same_provenance(self):
        cfg = self._prepared_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(cfg, tmp)
            checkpoint_extra = runner._checkpoint_extra()
            self.assertEqual(
                checkpoint_extra["provenance"]["split_digests"],
                runner.split_content_digests,
            )
            self.assertFalse(checkpoint_extra["provenance_complete"])
            self.assertFalse(checkpoint_extra["provenance"]["provenance_complete"])
            cfg.finetune.log_dir = tmp
            runner._save_training_log()
            log_path = Path(runner._log_path())
            payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["provenance"], checkpoint_extra["provenance"])
        self.assertFalse(payload["provenance_complete"])

    def test_development_artifacts_and_tsv_have_no_provenance(self):
        cfg = _cfg()
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(cfg, tmp)
            checkpoint_extra = runner._checkpoint_extra()
            self.assertNotIn("provenance", checkpoint_extra)
            self.assertNotIn("provenance_complete", checkpoint_extra)
            runner._save_training_log()
            log_payload = json.loads(Path(runner._log_path()).read_text(encoding="utf-8"))
            self.assertNotIn("provenance", log_payload)
            self.assertNotIn("provenance_complete", log_payload)

            cfg.save_results.output_dir = tmp
            now = datetime.now(timezone.utc)
            append_workflow_result(
                cfg=cfg,
                workflow="finetune",
                started_at=now,
                ended_at=now,
                checkpoint_save_paths=[],
                seeds=[42],
                best_epochs=[1],
                metric_summary={"test_acc": {"mean": 0.5, "std": 0.0}},
                provenance=None,
            )
            with (Path(tmp) / "finetune.tsv").open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle, delimiter="\t"))
            self.assertFalse(any(column.startswith("provenance.") for column in header))

    def test_result_tsv_persists_ordered_seed_split_digests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "design.md"
            manifest.write_text("frozen campaign\n", encoding="utf-8")
            pretrained = root / "pretrained.pt"
            pretrained.write_bytes(b"pretrained")
            cfg = _cfg()
            cfg.provenance.publication = True
            cfg.provenance.campaign_id = "campaign-a"
            cfg.provenance.design_manifest_path = str(manifest)
            cfg.provenance.design_manifest_digest = compute_file_sha256(manifest)
            cfg.provenance.source_tree_digest = compute_source_tree_snapshot().digest
            prepare_campaign_provenance(cfg)
            cfg.provenance.pretrained_checkpoint_config_sha256 = "c" * 64
            bind_pretrained_checkpoint_provenance(cfg, pretrained)
            runners = []
            for seed, digit in ((7, "7"), (17, "8")):
                run_cfg = cfg.clone()
                run_cfg.seed = seed
                runners.append(SimpleNamespace(
                    cfg=run_cfg,
                    provenance_complete=True,
                    split_content_digests={
                        "train": {"sha256": digit * 64, "num_samples": 1},
                        "val": {"sha256": "a" * 64, "num_samples": 1},
                        "test": {"sha256": "b" * 64, "num_samples": 1},
                    },
                ))
            provenance = aggregate_run_provenance(cfg, runners, [7, 17])
            cfg.save_results.output_dir = tmp
            now = datetime.now(timezone.utc)
            for field in (
                "campaign_id",
                "runtime_versions",
                "pretrained_checkpoint_sha256",
                "pretrained_checkpoint_config_sha256",
            ):
                with self.subTest(rejected_result_provenance=field):
                    mismatched = dict(provenance)
                    mismatched[field] = "different"
                    with self.assertRaisesRegex(
                        CampaignProvenanceError,
                        "matching the live campaign",
                    ):
                        append_workflow_result(
                            cfg=cfg,
                            workflow="finetune",
                            started_at=now,
                            ended_at=now,
                            checkpoint_save_paths=[],
                            seeds=[7, 17],
                            best_epochs=[1, 2],
                            metric_summary={
                                "test_acc": {"mean": 0.5, "std": 0.1}
                            },
                            provenance=mismatched,
                        )
                    self.assertFalse((Path(tmp) / "finetune.tsv").exists())
            append_workflow_result(
                cfg=cfg,
                workflow="finetune",
                started_at=now,
                ended_at=now,
                checkpoint_save_paths=[],
                seeds=[7, 17],
                best_epochs=[1, 2],
                metric_summary={"test_acc": {"mean": 0.5, "std": 0.1}},
                provenance=provenance,
            )
            with (Path(tmp) / "finetune.tsv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(row["provenance.campaign_id"], "campaign-a")
        self.assertEqual(row["provenance.provenance_complete"], "True")
        self.assertEqual(json.loads(row["provenance.split_digests"])[1]["seed"], 17)
        self.assertNotIn("provenance.source_tree_paths", row)
        self.assertEqual(
            row["provenance.source_tree_path_count"],
            str(cfg.provenance.source_tree_path_count),
        )
        self.assertIn("pytorch", json.loads(row["provenance.runtime_versions"]))

    def test_saved_artifact_mismatch_fails_closed(self):
        cfg = self._prepared_cfg()
        current = campaign_provenance_payload(
            cfg,
            split_digests={"train": {"sha256": "3" * 64}},
            provenance_complete=True,
        )
        self.assertTrue(
            saved_provenance_matches(current, dict(current), require_split_digests=True)
        )
        for key in (
            "campaign_id",
            "design_manifest_path",
            "design_manifest_digest",
            "source_tree_digest",
            "source_tree_path_count",
            "source_tree_paths",
            "pretrained_checkpoint_path",
            "pretrained_checkpoint_sha256",
            "pretrained_checkpoint_config_sha256",
            "runtime_versions",
            "split_digests",
        ):
            with self.subTest(key=key):
                saved = dict(current)
                saved[key] = "different"
                self.assertFalse(
                    saved_provenance_matches(
                        current,
                        saved,
                        require_split_digests=True,
                    )
                )
        incomplete = dict(current)
        incomplete["provenance_complete"] = False
        self.assertFalse(
            saved_provenance_matches(
                current,
                incomplete,
                require_split_digests=True,
            )
        )

    def test_runner_never_reuses_mismatched_publication_checkpoint(self):
        cfg = self._prepared_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(cfg, tmp, run_name="publication-reuse")
            checkpoint_path = Path(runner._checkpoint_path())
            log_path = Path(runner._log_path())
            provenance = runner._provenance_payload(provenance_complete=True)
            torch.save({
                "provenance_complete": True,
                "extra": {
                    "provenance": provenance,
                    "provenance_complete": True,
                },
            }, checkpoint_path)
            log_path.write_text(json.dumps({
                "provenance": provenance,
                "provenance_complete": True,
            }), encoding="utf-8")
            self.assertEqual(runner._existing_checkpoint_path(), str(checkpoint_path))
            original_checkpoint = checkpoint_path.read_bytes()

            cfg.provenance.campaign_id = "different-campaign"
            with self.assertRaisesRegex(CampaignProvenanceError, "Refusing to overwrite"):
                runner._existing_checkpoint_path()
            self.assertEqual(checkpoint_path.read_bytes(), original_checkpoint)

            cfg.provenance.campaign_id = "campaign-a"
            log_path.unlink()
            with self.assertRaisesRegex(CampaignProvenanceError, "companion log"):
                runner._existing_checkpoint_path()
            self.assertEqual(checkpoint_path.read_bytes(), original_checkpoint)

            log_path.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(CampaignProvenanceError, "Unreadable"):
                runner._existing_checkpoint_path()
            self.assertEqual(checkpoint_path.read_bytes(), original_checkpoint)

    def test_publication_reuse_rejects_unsafe_checkpoint_without_execution(self):
        cfg = self._prepared_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(cfg, tmp, run_name="unsafe-reuse")
            checkpoint_path = Path(runner._checkpoint_path())
            log_path = Path(runner._log_path())
            marker = Path(tmp) / "executed.txt"
            torch.save(
                {"unsafe": _UnsafeCheckpointValue(marker)},
                checkpoint_path,
            )
            log_path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(
                CampaignProvenanceError,
                "Unreadable publication checkpoint",
            ):
                runner._existing_checkpoint_path()

            self.assertFalse(marker.exists())

    def test_selected_publication_checkpoint_rejects_unsafe_payload_before_use(self):
        cfg = self._prepared_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(cfg, tmp, run_name="unsafe-selected")
            checkpoint_path = Path(runner._checkpoint_path())
            marker = Path(tmp) / "executed.txt"
            torch.save(
                {"unsafe": _UnsafeCheckpointValue(marker)},
                checkpoint_path,
            )
            runner.test_loader = object()
            runner._checkpoint_written_this_run = True
            runner.device = torch.device("cpu")
            runner.model = Mock()
            runner.task = Mock()
            runner.optimizer = Mock()
            runner.task_supports_epoch = True

            with self.assertRaises(pickle.UnpicklingError):
                runner._evaluate_selected_test_checkpoint()

            self.assertFalse(marker.exists())
            runner.model.load_state_dict.assert_not_called()

    def test_publication_finalization_rejects_unsafe_checkpoint_without_execution(self):
        cfg = self._prepared_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(cfg, tmp, run_name="unsafe-finalize")
            checkpoint_path = Path(runner._checkpoint_path())
            log_path = Path(runner._log_path())
            marker = Path(tmp) / "executed.txt"
            torch.save(
                {"unsafe": _UnsafeCheckpointValue(marker)},
                checkpoint_path,
            )
            log_path.write_text("{}", encoding="utf-8")

            with patch(
                "src.finetune.finetuner.prepare_campaign_provenance",
                return_value={},
            ), patch(
                "src.finetune.finetuner.bind_pretrained_checkpoint_provenance",
                return_value=runner.pretrained_checkpoint,
            ):
                with self.assertRaisesRegex(
                    CampaignProvenanceError,
                    "unreadable publication checkpoint",
                ):
                    runner._finalize_publication_artifacts()

            self.assertFalse(marker.exists())

    def test_aggregation_rejects_seed_mismatch_and_incomplete_runner(self):
        cfg = self._prepared_cfg()
        run_cfg = cfg.clone()
        run_cfg.seed = 7
        runner = SimpleNamespace(
            cfg=run_cfg,
            provenance_complete=True,
            split_content_digests={
                name: {"sha256": digit * 64, "num_samples": 1}
                for name, digit in zip(("train", "val", "test"), ("4", "5", "6"))
            },
        )
        with self.assertRaisesRegex(CampaignProvenanceError, "runner.cfg.seed"):
            aggregate_run_provenance(cfg, [runner], [17])
        runner.provenance_complete = False
        with self.assertRaisesRegex(CampaignProvenanceError, "incomplete"):
            aggregate_run_provenance(cfg, [runner], [7])

    def test_finalization_revalidates_before_atomically_marking_pair(self):
        cfg = self._prepared_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(cfg, tmp, run_name="publication-finalize")
            checkpoint_path = Path(runner._checkpoint_path())
            log_path = Path(runner._log_path())
            provenance = runner._provenance_payload(provenance_complete=False)
            torch.save({
                "provenance_complete": False,
                "extra": {
                    "provenance": provenance,
                    "provenance_complete": False,
                },
            }, checkpoint_path)
            log_path.write_text(json.dumps({
                "provenance": provenance,
                "provenance_complete": False,
            }), encoding="utf-8")

            with patch(
                "src.finetune.finetuner.prepare_campaign_provenance",
                side_effect=CampaignProvenanceError("source changed"),
            ):
                with self.assertRaisesRegex(CampaignProvenanceError, "source changed"):
                    runner._finalize_publication_artifacts()
            self.assertFalse(torch.load(checkpoint_path)["provenance_complete"])
            self.assertFalse(json.loads(log_path.read_text())["provenance_complete"])

            with patch(
                "src.finetune.finetuner.prepare_campaign_provenance",
                return_value={},
            ) as revalidate, patch(
                "src.finetune.finetuner.bind_pretrained_checkpoint_provenance",
                return_value=runner.pretrained_checkpoint,
            ):
                runner._finalize_publication_artifacts()
            revalidate.assert_called_once_with(cfg)
            checkpoint = torch.load(checkpoint_path)
            log = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertTrue(checkpoint["provenance_complete"])
            self.assertTrue(checkpoint["extra"]["provenance_complete"])
            self.assertTrue(log["provenance_complete"])
            self.assertTrue(log["provenance"]["provenance_complete"])
            self.assertTrue(runner.provenance_complete)

    def test_publication_append_revalidates_and_never_writes_on_failure(self):
        cfg = self._prepared_cfg()
        provenance = campaign_provenance_payload(cfg, provenance_complete=True)
        with tempfile.TemporaryDirectory() as tmp:
            cfg.save_results.output_dir = tmp
            now = datetime.now(timezone.utc)
            with patch(
                "src.utils.provenance.prepare_campaign_provenance",
                side_effect=CampaignProvenanceError("source changed"),
            ):
                with self.assertRaisesRegex(CampaignProvenanceError, "source changed"):
                    append_workflow_result(
                        cfg=cfg,
                        workflow="finetune",
                        started_at=now,
                        ended_at=now,
                        checkpoint_save_paths=[],
                        seeds=[7],
                        best_epochs=[1],
                        metric_summary={},
                        provenance=provenance,
                    )
            self.assertFalse((Path(tmp) / "finetune.tsv").exists())


if __name__ == "__main__":
    unittest.main()
