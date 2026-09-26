import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from yacs.config import CfgNode as CN

from src.config import cfg
from src.finetune.finetuner import FinetuneRunner
from src.utils.provenance import CampaignProvenanceError
from src.utils.run_helpers import (
    collect_run_metrics,
    load_checkpoint_metrics,
    should_save_result,
)


class SaveSkippedResultsTest(unittest.TestCase):
    def _deferred_reuse_runner(self, root: Path, name: str):
        local_cfg = SimpleNamespace(
            finetune=SimpleNamespace(
                checkpoint_dir=str(root / name / "checkpoints"),
                skip_if_exists=True,
            ),
            provenance=SimpleNamespace(publication=True),
            save_results=SimpleNamespace(save_skipped=False),
        )
        runner = FinetuneRunner.__new__(FinetuneRunner)
        runner.cfg = local_cfg
        runner.run_dir = str(root / name / "run")
        runner._skip_due_to_existing_checkpoint = False
        runner._setup = Mock()
        checkpoint = str(root / name / "existing.pt")
        runner._existing_checkpoint_path = Mock(return_value=checkpoint)

        runner.fit()

        runner._setup.assert_called_once_with()
        self.assertTrue(runner._skip_due_to_existing_checkpoint)
        self.assertEqual(runner._reused_checkpoint_path, checkpoint)
        return runner, local_cfg

    def test_default_does_not_save_skipped_runs(self):
        runner = SimpleNamespace(_skip_due_to_existing_checkpoint=True)

        self.assertFalse(cfg.save_results.save_skipped)
        self.assertFalse(should_save_result([runner], cfg))

    def test_explicit_opt_in_saves_skipped_runs(self):
        runner = SimpleNamespace(_skip_due_to_existing_checkpoint=True)
        local_cfg = SimpleNamespace(save_results=SimpleNamespace(save_skipped=True))

        self.assertTrue(should_save_result([runner], local_cfg))

    def test_deferred_all_reuse_skips_aggregate_but_partial_resume_saves_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, local_cfg = self._deferred_reuse_runner(root, "first")
            second, _ = self._deferred_reuse_runner(root, "second")

        append_result = Mock()
        if should_save_result([first, second], local_cfg):
            append_result()
        append_result.assert_not_called()

        newly_completed = SimpleNamespace(_skip_due_to_existing_checkpoint=False)
        if should_save_result([first, newly_completed], local_cfg):
            append_result()
        append_result.assert_called_once_with()

    def test_publication_metric_fallback_is_safe_and_legacy_development_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "legacy.pt"
            torch.save(
                {
                    "cfg": CN({"legacy": True}),
                    "metrics": {"test_acc": 0.75},
                },
                checkpoint,
            )

            with patch(
                "src.utils.run_helpers.torch.load",
                wraps=torch.load,
            ) as load:
                metrics = load_checkpoint_metrics(
                    str(checkpoint),
                    log_prefix="[test]",
                )
            self.assertEqual(metrics, {"test_acc": 0.75})
            self.assertNotIn("weights_only", load.call_args.kwargs)

            publication_runner = SimpleNamespace(
                best_metrics={},
                cfg=SimpleNamespace(
                    provenance=SimpleNamespace(publication=True),
                ),
                get_checkpoint_path_for_metrics=lambda: str(checkpoint),
            )
            with patch(
                "src.utils.run_helpers.torch.load",
                wraps=torch.load,
            ) as load:
                with self.assertRaisesRegex(
                    CampaignProvenanceError,
                    "safely load publication checkpoint metrics",
                ):
                    collect_run_metrics(
                        publication_runner,
                        log_prefix="[test]",
                    )
            self.assertEqual(load.call_count, 1)
            self.assertIs(load.call_args.kwargs["weights_only"], True)


if __name__ == "__main__":
    unittest.main()
