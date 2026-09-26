"""Featureless graph datasets keep their published feature protocol.

qm7b has no native node features; every published pretrained/finetuned
artifact for it was built on the 1-dim EnsureFeatureTransform degree
feature. Silently padding such datasets to feat_reduction_dim would
make every existing qm7b encoder checkpoint unloadable (meta/loader
in_dim mismatch, runs die before training). Padding is now opt-in via
create_dataset(pad_featureless_features=True), used only for cross-dataset
fine-tuning whose checkpoint in_dim exceeds the native width (e.g. ZINC/PubMed
-> QM7b transfer).
"""

import unittest

from src.data_loader.datasets import create_dataset

COMMON = dict(
    root="data/datasets",
    task_level="graph",
    feat_reduction=True,
    feat_reduction_dim=100,
    feature_svd_dir="data/feature_svd",
    graph_filter_dir="data/filters",
)


class FeaturelessFeatureProtocolTest(unittest.TestCase):
    def test_qm7b_default_keeps_published_1dim_protocol(self):
        ds = create_dataset(name="qm7b", **COMMON)
        self.assertEqual(ds[0].num_features, 1)

    def test_qm7b_pad_opt_in_serves_100dim(self):
        ds = create_dataset(name="qm7b", pad_featureless_features=True, **COMMON)
        self.assertEqual(ds[0].num_features, 100)

    def test_native_x_dataset_unaffected_by_default(self):
        ds = create_dataset(name="toxcast", **COMMON)
        self.assertEqual(ds[0].num_features, 100)


if __name__ == "__main__":
    unittest.main()
