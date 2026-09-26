import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from src.data_loader.dataset_loader import make_loaders
from src.data_loader.datasets import _induced_feature_identity, _induced_feature_tag, create_dataset
from src.data_loader.summary import DatasetSummaryRow, _load_existing_summary_rows, _rows_to_tsv


def _summary_row(name):
    return DatasetSummaryRow(name, "node", 1, 3, 3.0, 2, 2.0, 4, "classification", 2)


class SummaryPersistenceTest(unittest.TestCase):
    def test_summary_merges_under_lock_and_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.tsv"
            self.assertEqual(_rows_to_tsv([_summary_row("cora")], path), 1)
            self.assertEqual(_rows_to_tsv([_summary_row("pubmed")], path), 2)
            self.assertEqual([row.name for row in _load_existing_summary_rows(path)], ["cora", "pubmed"])
            self.assertTrue((Path(tmp) / "summary.tsv.lock").is_file())
            self.assertEqual(list(Path(tmp).glob(".summary.tsv.*.tmp")), [])

    def test_failed_replace_preserves_previous_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.tsv"
            _rows_to_tsv([_summary_row("cora")], path)
            before = path.read_bytes()
            with patch("src.data_loader.summary.os.replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    _rows_to_tsv([_summary_row("pubmed")], path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(tmp).glob(".summary.tsv.*.tmp")), [])


class EdgeMessageContextTest(unittest.TestCase):
    def test_eval_context_adds_train_positives_but_not_heldout_targets(self):
        data = Data(
            x=torch.randn(5, 3),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            num_nodes=5,
        )

        class Dataset:
            def __getitem__(self, index):
                self.assert_index = index
                return data

        payload = {
            "train_pos_idx": [0],
            "val_pos_idx": [1],
            "test_pos_idx": [2],
            "message_pos_idx": [3],
            "context_pos_idx": [0, 3],
            "train_neg_edge_index": torch.empty((2, 0), dtype=torch.long),
            "val_neg_edge_index": torch.empty((2, 0), dtype=torch.long),
            "test_neg_edge_index": torch.empty((2, 0), dtype=torch.long),
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.data_loader.dataset_loader._get_or_create_edge_split_payload",
            return_value=payload,
        ):
            train, val, test = make_loaders(
                Dataset(), "toy", "edge", 4, 0, (0.25, 0.25, 0.25), 1,
                induced=False, split_root=tmp,
            )
        train_data = next(iter(train))
        val_data = next(iter(val))
        test_data = next(iter(test))
        self.assertTrue(torch.equal(train_data.edge_index, data.edge_index[:, [3]]))
        expected_eval = data.edge_index[:, [0, 3]]
        self.assertTrue(torch.equal(val_data.edge_index, expected_eval))
        self.assertTrue(torch.equal(test_data.edge_index, expected_eval))


class InducedCacheIdentityTest(unittest.TestCase):
    def test_feature_source_and_reduction_change_cache_identity(self):
        dataset = SimpleNamespace(root="data/datasets/toy")
        raw = _induced_feature_identity(
            base_dataset=dataset, requested_root="data/datasets", feat_reduction=False,
            feat_reduction_dim=100, persist_feature_svd=True, feature_svd_dir="data/feature_svd",
        )
        svd100 = _induced_feature_identity(
            base_dataset=dataset, requested_root="data/datasets", feat_reduction=True,
            feat_reduction_dim=100, persist_feature_svd=True, feature_svd_dir="data/feature_svd",
        )
        svd64 = dict(svd100, feat_reduction_dim=64)
        other_source = dict(svd100, feature_source="persisted_svd:/other")
        tags = {_induced_feature_tag(value) for value in (raw, svd100, svd64, other_source)}
        self.assertEqual(len(tags), 4)


class EdgeInducedCacheMaskingRuleTest(unittest.TestCase):
    """Edge caches stamped with a non-both-direction masking rule are misses."""

    def _create(self, tmp, **kwargs):
        generator = torch.Generator().manual_seed(0)
        src = torch.randint(0, 40, (160,), generator=generator)
        dst = torch.randint(0, 40, (160,), generator=generator)
        keep = src != dst
        data = Data(x=torch.randn(40, 4, generator=generator),
                    edge_index=torch.stack([src[keep], dst[keep]]), num_nodes=40)

        class ToyDataset:
            root = str(Path(tmp) / "raw")

            def __getitem__(self, index):
                return data

        with patch("src.data_loader.datasets._load_node_dataset", return_value=ToyDataset()):
            return create_dataset(
                "toy", root=tmp, task_level="edge", feat_reduction=False, induced=True,
                induced_max_hops=1, split=(0.2, 0.1, 0.1), seed=0,
                split_root=str(Path(tmp) / "splits"), induced_root=str(Path(tmp) / "induced"),
                **kwargs,
            )

    def _cache_path(self, tmp):
        (path,) = (Path(tmp) / "induced").rglob("*_induced_edge_*.pt")
        return path

    def _stamp(self, path, masking):
        payload = torch.load(path)
        payload["graphs"], payload["split_tags"] = payload["graphs"][:1], payload["split_tags"][:1]
        if masking is None:
            payload["meta"].pop("queried_edge_masking")
        else:
            payload["meta"]["queried_edge_masking"] = masking
        torch.save(payload, path)

    def test_fresh_cache_records_both_direction_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = self._create(tmp)
            self.assertGreater(len(full), 1)
            meta = torch.load(self._cache_path(tmp))["meta"]
            self.assertEqual(meta["queried_edge_masking"], "both_directions")

    def test_legacy_cache_without_rule_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._create(tmp)
            self._stamp(self._cache_path(tmp), None)
            self.assertEqual(len(self._create(tmp, require_induced_cache_hit=True)), 1)

    def test_directed_only_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = len(self._create(tmp))
            path = self._cache_path(tmp)
            self._stamp(path, "queried_direction_only")
            with self.assertRaisesRegex(RuntimeError, "Required induced edge cache miss"):
                self._create(tmp, require_induced_cache_hit=True)
            self.assertEqual(len(self._create(tmp)), full)
            self.assertEqual(torch.load(path)["meta"]["queried_edge_masking"], "both_directions")


if __name__ == "__main__":
    unittest.main()
