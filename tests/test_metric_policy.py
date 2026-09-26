import unittest

from src.results import finetune_tables, train_tables
from src.results.metric_policy import eval_metric


class MetricPolicyTest(unittest.TestCase):
    def test_policy_rules(self):
        self.assertEqual(eval_metric("qm7b", "graph", "regression"), "test_mae")
        self.assertEqual(eval_metric("cornell", "edge", "classification"), "test_auc")
        self.assertEqual(eval_metric("toxcast", "graph", "classification"), "test_auc")
        self.assertEqual(eval_metric("Tox21", "graph", "classification"), "test_auc")
        self.assertEqual(eval_metric("bace", "graph", "classification"), "test_acc")
        self.assertEqual(eval_metric("mnist", "graph", "classification"), "test_acc")
        self.assertEqual(eval_metric("photo", "node", "classification"), "test_acc")

    def test_every_table_reports_the_same_metric_per_dataset(self):
        expected = {
            "photo": "test_acc",
            "ogbn-arxiv": "test_acc",
            "dblp": "test_auc",
            "airports": "test_acc",
            "chameleon": "test_acc",
            "cornell": "test_auc",
            "qm7b": "test_mae",
            "toxcast": "test_auc",
            "mnist": "test_acc",
        }
        for module in (train_tables, finetune_tables):
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    {spec.name: spec.metric for spec in module.DATASETS}, expected
                )


if __name__ == "__main__":
    unittest.main()
