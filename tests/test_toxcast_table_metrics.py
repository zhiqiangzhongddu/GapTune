import unittest

from src.results import train_tables


class ToxCastTableMetricsTest(unittest.TestCase):
    def test_train_table_uses_auc_instead_of_accuracy(self):
        spec = next(item for item in train_tables.DATASETS if item.name == "toxcast")
        self.assertEqual(spec.metric, "test_auc")

        row = {
            "test_acc_mean": "0.91",
            "test_acc_std": "0.01",
            "test_auc_mean": "0.61",
            "test_auc_std": "0.02",
        }
        latest = {
            train_tables.CellKey("gcn", "toxcast", spec.f5_split): row,
        }
        rendered = train_tables._render_table("f5", latest)

        self.assertIn("Toxcast multilabel graph classification", rendered)
        self.assertIn("61.00$_{\\pm2.00}$", rendered)
        self.assertNotIn("91.00$_{\\pm1.00}$", rendered)


if __name__ == "__main__":
    unittest.main()
