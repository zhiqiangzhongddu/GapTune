"""A saved graph-filter manifest with no dropped graphs must be reused.

A stored drop_count of 0 used to be read as -1, so datasets without empty
graphs (qm7b, zinc, mnist) re-scanned and re-saved their manifest on every load.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from src.data_loader.filter_empty_graph import GRAPH_FILTER_REASON, _load_graph_filter_manifest


def _manifest(keep, drop):
    return {
        "keep_indices": list(keep),
        "drop_indices": list(drop),
        "meta": {
            "dataset_name": "toy",
            "task_level": "graph",
            "reason": GRAPH_FILTER_REASON,
            "raw_total": len(keep) + len(drop),
            "filtered_total": len(keep),
            "drop_count": len(drop),
        },
    }


class GraphFilterManifestTest(unittest.TestCase):
    def _load(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy_graph_filter.pt"
            torch.save(payload, path)
            return _load_graph_filter_manifest(path, "toy", raw_total=payload["meta"]["raw_total"])

    def test_zero_drop_manifest_is_reused(self):
        loaded = self._load(_manifest(keep=[0, 1, 2], drop=[]))
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["meta"]["drop_count"], 0)

    def test_manifest_with_drops_is_reused(self):
        self.assertIsNotNone(self._load(_manifest(keep=[0, 2], drop=[1])))

    def test_count_mismatch_is_rejected(self):
        payload = _manifest(keep=[0, 1, 2], drop=[])
        payload["meta"]["drop_count"] = 1
        self.assertIsNone(self._load(payload))


if __name__ == "__main__":
    unittest.main()
