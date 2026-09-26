"""Finetune runner — main training orchestrator for all finetune methods.

The runner supports two execution paths, selected by the task's base class:

* **Step-based** (``StepFinetuneTask``): the runner owns the training loop
  via ``run_step_epoch()``. The task provides ``step()`` and ``evaluate()``
  per batch. Used by supervised finetuning.

* **Epoch-based** (``FinetuneTask``): the task owns the training loop via
  ``train_epoch()`` and ``evaluate_split()``. The runner calls these once
  per epoch and handles monitoring, checkpointing, and early stopping.
  Used by prompt methods (GraphPrompt, EdgePrompt, GPF, GPPT, All-in-One).

Invariants maintained by both paths:
  - Frozen encoder mode is applied by the runner before each epoch.
  - Monitoring, checkpoint saving, and early stopping are runner-managed.
  - Evaluation follows the same metric naming convention.
  - Gradient clipping is applied only in the step-based path (via
    ``run_step_epoch``). Epoch-based methods handle grad clipping internally
    or via ``run_epoch_loop``.
"""

from __future__ import annotations

import json
import os
import time
from numbers import Integral
from collections.abc import Mapping

from typing import Any

import torch
from torch import optim

from src.data_loader import create_dataset, dataset_info, log_split_instance_counts
from src.data_loader.split_provenance import validate_materialized_split_manifest
from src.finetune.monitoring import resolve_finetune_monitor_spec
from src.model import build_encoder_from_cfg
from src.utils.checkpoint import (
    canonical_config_sha256,
    cfg_to_dict,
    save_checkpoint,
    save_json_atomic,
    save_torch_atomic,
    save_training_log,
)
from src.utils.monitoring import is_metric_improved, merge_epoch_metrics, monitor_uses_train_split, resolve_monitor_value, should_print_metric
from src.utils.naming import (
    build_finetune_run_name_from_cfg,
    compact_artifact_stem,
    format_split_for_name,
)
from src.utils.paths import ensure_dir
from src.utils.provenance import (
    CampaignProvenanceError,
    bind_pretrained_checkpoint_provenance,
    campaign_provenance_payload,
    compute_file_sha256,
    is_publication_campaign,
    prepare_campaign_provenance,
    saved_provenance_matches,
)
from src.utils.supervised_eval import runner_evaluate_split
from src.utils.dataset_helpers import (
    checkpoint_dataset_dir_name,
    is_few_shot_split,
    make_workflow_loaders,
    populate_dataset_cfg_from_meta,
    resolve_effective_task_level,
    resolve_loader_task_level,
    shared_induced_root,
    shared_split_root,
)
from src.utils.parsing import resolve_task_type, resolve_workflow_split, to_bool
from src.utils.random import set_seed
from src.utils.save_results import get_explicit_cfg_keys
from src.utils.training import run_step_epoch
from .dataset_cfg import resolve_target_dataset_cfg
from .encoders.edgeprompt import build_prompt_encoder
from .frozen_load import check_frozen_encoder_load
from .registry import build_finetune_task, get_finetune_task_class
from .task_base import _VALID_FROZEN_ENCODER_MODES, FinetuneTask

_BEST_TEST_METRICS_TO_PRINT = ("test_acc", "test_micro_f1", "test_macro_f1", "test_auc", "test_mae", "test_mse")
# Width of the synthesized degree feature (EnsureFeatureTransform) that
# featureless graph datasets such as QM7b serve.
_FEATURELESS_NATIVE_FEATURE_DIM = 1


class FinetuneRunner:
    """Fine-tune a pretrained encoder on a target dataset."""

    def __init__(self, cfg, pretrained_checkpoint: str, pretrained_run_name: str = None):
        # Direct runner construction must honor the same publication boundary
        # as the CLI runtime: validate the live source tree before loading data,
        # a checkpoint, or performing any optimization.
        prepare_campaign_provenance(cfg)
        if not os.path.isfile(pretrained_checkpoint):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_checkpoint}")
        pretrained_checkpoint = bind_pretrained_checkpoint_provenance(
            cfg,
            pretrained_checkpoint,
        )

        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(seed=self.cfg.seed)
        # Define summary fields up front so skip paths are safe for run-level aggregation.
        self.best_metrics: dict[str, float] = {}
        self.best_epoch = None
        self.best_metric = float("nan")
        self.monitor_name = "train_loss"
        self.monitor_mode = "min"
        self.train_history = []
        self.split_content_digests: dict[str, dict[str, Any]] = {}
        self.provenance_complete = False
        self._checkpoint_written_this_run = False
        self._reused_checkpoint_path: str | None = None

        self.pretrained_checkpoint = pretrained_checkpoint
        self.pretrained_run_name = pretrained_run_name or os.path.splitext(os.path.basename(pretrained_checkpoint))[0]
        # Pretrained artifacts are a separate, explicit trusted-input boundary:
        # publication mode pins their exact path and SHA-256, but historical
        # files may still contain a YACS CfgNode. Finetune artifacts reopened
        # later in this runner always use weights_only=True for publication.
        self._loaded_checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
        if not isinstance(self._loaded_checkpoint, Mapping):
            raise CampaignProvenanceError(
                "Pretrained checkpoint must contain a mapping payload."
            )
        if is_publication_campaign(cfg):
            # Close the hash/load race: the bytes must still match immediately
            # after deserialization, not merely before it.
            expected_digest = str(cfg.provenance.pretrained_checkpoint_sha256)
            loaded_digest = compute_file_sha256(pretrained_checkpoint)
            if loaded_digest != expected_digest:
                raise CampaignProvenanceError(
                    "Pretrained checkpoint changed while it was being loaded: "
                    f"expected {expected_digest}, found {loaded_digest}."
                )
        raw_pretrain_cfg = self._loaded_checkpoint.get("cfg") or {}
        try:
            self.pretrain_cfg = cfg_to_dict(raw_pretrain_cfg)
        except TypeError as exc:
            raise CampaignProvenanceError(
                "Pretrained checkpoint has no canonicalizable config mapping."
            ) from exc
        if is_publication_campaign(cfg):
            self._validate_loaded_pretrain_config()
        self.pretrain_dataset_meta = self._loaded_checkpoint.get("dataset", {}) or {}
        pretrain_block = self.pretrain_cfg.get("pretrain", {}) if isinstance(self.pretrain_cfg, dict) else {}
        pretrain_ds_block = pretrain_block.get("dataset", {}) if isinstance(pretrain_block, dict) else {}

        self.pretrain_dataset_name = (
            pretrain_ds_block.get("name")
            or ((self.pretrain_cfg.get("dataset") or {}).get("name") if isinstance(self.pretrain_cfg, dict) else None)
        )
        self.pretrain_task_level = (
            pretrain_ds_block.get("task_level")
            or ((self.pretrain_cfg.get("dataset") or {}).get("task_level") if isinstance(self.pretrain_cfg, dict) else None)
        )
        self.pretrain_method = (
            pretrain_block.get("method") if isinstance(pretrain_block, dict) else None
        )
        model_block = self.pretrain_cfg.get("model", {})
        self.pretrain_model_name = (
            model_block.get("name") if isinstance(model_block, dict) else None
        )
        self.pretrain_induced = (
            pretrain_ds_block.get("induced")
            if isinstance(pretrain_ds_block, dict)
            else None
        )

        self._apply_pretrained_model_cfg()

        self.finetune_method = (
            getattr(getattr(cfg, "finetune", None), "method", "supervised") or "supervised"
        ).lower().replace("-", "_")

        # Resolve the task class early so we can validate config and use
        # class-level hooks (variant_tag, adjust_dataset_cfg, encoder_builder)
        # *before* the skip-if-exists check.  This prevents invalid configs
        # from being silently skipped just because a same-name checkpoint exists.
        self.task_cls = get_finetune_task_class(self.finetune_method)
        if self.task_cls is None:
            raise ValueError(f"Unknown finetune method: {self.finetune_method}")

        target_ds = self._get_target_dataset_cfg()
        raw_task_level = target_ds["task_level"]
        self.task_level_raw = raw_task_level

        # Write ALL resolved dataset params (including method adjustments
        # like GPF induced override and EdgePrompt subgraph sizing) back
        # to cfg so that _build_run_name(), _setup(), and saved logs all
        # see the effective values.
        ds_cfg = self.cfg.finetune.dataset
        ds_cfg.name = target_ds["name"]
        ds_cfg.task_level = raw_task_level
        ds_cfg.induced = target_ds["induced"]
        ds_cfg.task_level_raw = raw_task_level
        ds_cfg.task_level_effective = resolve_effective_task_level(
            raw_task_level,
            target_ds["induced"],
        )
        self.effective_task_level = ds_cfg.task_level_effective
        for _key in ("induced_min_size", "induced_max_size", "induced_max_hops"):
            if _key in target_ds and hasattr(ds_cfg, _key):
                setattr(ds_cfg, _key, target_ds[_key])

        # Validate *after* dataset resolution so method-level checks can read
        # the resolved task_level/induced/effective level from cfg.finetune.dataset.
        self.task_cls.validate_cfg(cfg)

        self.split = self._resolve_split()
        self.freeze_pretrained_effective = self._effective_freeze_pretrained()
        self.run_name = self._build_run_name()
        self.run_group = checkpoint_dataset_dir_name(target_ds["name"])
        self.run_dir = os.path.join(self.cfg.finetune.checkpoint_dir, self.run_group)
        self._skip_due_to_existing_checkpoint = False
        ckpt_path = self._existing_checkpoint_path()
        if self.cfg.finetune.skip_if_exists and ckpt_path is not None:
            if is_publication_campaign(self.cfg):
                # Publication identity also includes the exact materialized
                # split, which is only available after loader construction.
                print(
                    "[Finetune] Deferring checkpoint skip check until effective "
                    f"context/split provenance is built: {ckpt_path}"
                )
            else:
                self._skip_due_to_existing_checkpoint = True
                self._reused_checkpoint_path = ckpt_path
                del self._loaded_checkpoint  # free pretrained weights on skip path
                print(f"[Finetune] Checkpoint already exists, skipping: {ckpt_path}")
                return

        self._target_ds = target_ds
        self._is_setup = False

    # ------------------------------------------------------------------ #
    # Lazy heavy setup
    # ------------------------------------------------------------------ #
    def _setup(self) -> None:
        """Load dataset, build model/task/optimizer/loaders. Called once by fit()."""
        if self._is_setup:
            return
        self._is_setup = True

        cfg = self.cfg
        target_ds = self._target_ds
        raw_task_level = self.task_level_raw
        induced = target_ds["induced"]
        # Cross-dataset transfer onto a featureless graph dataset (QM7b serves
        # 1-dim degree features): when the checkpoint-restored model.in_dim is
        # wider than that, zero-pad the degree feature to feat_reduction_dim,
        # the source feature width (paper D.1). create_dataset pads only graph
        # datasets without native x, so native-feature targets and
        # same-dataset QM7b checkpoints (in_dim=1) are unaffected.
        pad_featureless_features = (
            raw_task_level == "graph"
            and int(cfg.model.in_dim or 0) > _FEATURELESS_NATIVE_FEATURE_DIM
        )
        self.dataset = create_dataset(
            name=target_ds["name"],
            root=target_ds["root"],
            task_level=raw_task_level,
            feat_reduction=target_ds["feat_reduction"],
            feat_reduction_dim=target_ds["feat_reduction_dim"],
            feature_svd_dir=target_ds["feature_svd_dir"],
            induced=induced,
            induced_min_size=target_ds["induced_min_size"],
            induced_max_size=target_ds["induced_max_size"],
            induced_max_hops=target_ds["induced_max_hops"],
            induced_root=shared_induced_root(self.cfg, target_ds.get("induced_root", "")),
            split=self.split,
            seed=self.cfg.seed,
            split_root=shared_split_root(self.cfg),
            pad_featureless_features=pad_featureless_features,
        )
        effective_task_level = self.effective_task_level
        target_cfg = self.cfg.finetune.dataset
        # Reset cfg to the runner-resolved dataset view after dataset
        # construction. This is intentionally idempotent with __init__ and
        # protects setup from any external cfg mutation between validation
        # and lazy initialization.
        target_cfg.task_level = raw_task_level
        target_cfg.task_level_raw = raw_task_level
        target_cfg.task_level_effective = effective_task_level
        target_cfg.name = target_ds["name"]
        target_cfg.induced = induced

        self.dataset_meta = dataset_info(
            dataset=self.dataset,
            task_level=raw_task_level,
            name=target_ds["name"],
            induced=induced,
        )
        populate_dataset_cfg_from_meta(cfg.model, target_cfg, self.dataset_meta)

        # Use the method-declared encoder builder (default: standard encoder;
        # EdgePrompt declares "prompt" for PromptGNNEncoder).
        encoder_kind = self.task_cls.resolve_encoder_builder(cfg)
        if encoder_kind == "prompt":
            self.model = build_prompt_encoder(
                cfg=cfg,
                in_dim=cfg.model.in_dim,
            ).to(self.device)
        else:
            self.model = build_encoder_from_cfg(
                cfg=cfg,
                in_dim=cfg.model.in_dim,
            ).to(self.device)
        # Non-default encoder classes may have different state_dict keys, so
        # strict loading is only used when the standard encoder is expected.
        pretrain_strict = encoder_kind == "default"
        missing, unexpected = self.model.load_state_dict(
            self._loaded_checkpoint.get("model_state", {}),
            strict=False,
        )
        del self._loaded_checkpoint  # free pretrained weights from CPU memory
        if missing:
            if pretrain_strict:
                raise RuntimeError(
                    f"[Finetune] Architecture mismatch: {len(missing)} missing keys when loading "
                    f"pretrained weights (model config may not match checkpoint): {missing}"
                )
            print(f"[Finetune] Expected missing keys for {self.finetune_method}: {missing}")
        if unexpected:
            print(f"[Finetune] Unexpected keys when loading pretrained weights: {unexpected}")
        # Frozen-encoder safety: a partial load into a frozen backbone leaves
        # random weights in place of pretrained weights and silently
        # invalidates downstream metrics.  Fail loud before training starts.
        require_frozen = bool(getattr(self.task_cls, "requires_frozen_encoder", False))
        min_ratio = float(getattr(cfg.finetune, "frozen_load_min_match_ratio", 1.0))
        check_frozen_encoder_load(
            encoder=self.model,
            missing_keys=missing,
            unexpected_keys=unexpected,
            min_match_ratio=min_ratio,
            require_frozen=require_frozen,
        )
        model_name = getattr(cfg.model, "name", "model")
        print(f"[Finetune] Encoder architecture ({model_name}):\n{self.model}")

        self.task = build_finetune_task(self.finetune_method, cfg).to(self.device)
        # Gated mechanisms that only a constructed task can decide (dataset
        # meta is unavailable to variant_tag at cfg time) declare engaged-only
        # run tokens here, so checkpoint identity, skip_if_exists, and the
        # results key all read the final name.
        for engaged_token in getattr(self.task, "engaged_run_tokens", ()) or ():
            self.run_name = f"{self.run_name}-{engaged_token}"
        if hasattr(self.task, "validate_encoder"):
            self.task.validate_encoder(self.model)
        self._maybe_freeze_pretrained_encoder()
        print(f"[Finetune] Task head:\n{self.task}")
        self.task_supports_epoch = isinstance(self.task, FinetuneTask)
        self.optimizers = None
        if self.task_supports_epoch:
            optimizers = self.task.build_optimizers(self.model)
            if optimizers is None:
                params = [p for p in self.model.parameters() if p.requires_grad] + list(
                    self.task.parameters_to_optimize()
                )
                self.optimizer = optim.Adam(
                    params=params,
                    lr=cfg.finetune.lr,
                    weight_decay=cfg.finetune.weight_decay,
                )
                self.optimizers = {"primary": self.optimizer}
            elif isinstance(optimizers, dict):
                self.optimizers = optimizers
                primary = optimizers.get("primary")
                if primary is None:
                    primary = next(iter(optimizers.values()))
                self.optimizer = primary
            else:
                self.optimizer = optimizers
                self.optimizers = {"primary": optimizers}
        else:
            params = [p for p in self.model.parameters() if p.requires_grad] + list(
                self.task.parameters_to_optimize()
            )
            self.optimizer = optim.Adam(
                params=params,
                lr=cfg.finetune.lr,
                weight_decay=cfg.finetune.weight_decay,
            )
            self.optimizers = {"primary": self.optimizer}

        self._init_monitoring()
        self.train_history = []
        ensure_dir(self.run_dir)
        loader_result = self._make_loaders(induced=induced)
        if is_publication_campaign(self.cfg):
            (
                self.train_loader,
                self.val_loader,
                self.test_loader,
                split_meta,
            ) = loader_result
            self.split_content_digests = dict(split_meta.get("content_digests") or {})
            validate_materialized_split_manifest(self.split_content_digests)
        else:
            self.train_loader, self.val_loader, self.test_loader = loader_result
        log_split_instance_counts(
            self.train_loader,
            self.val_loader,
            self.test_loader,
            task_level=resolve_loader_task_level(self.task_level_raw, self.effective_task_level, induced),
            split=self.split,
            induced=induced,
            prefix="[Finetune][Split]",
        )
        # Fit target statistics exactly once from the training loader. Every
        # finetune method routes regression and multilabel losses through
        # registered shared modules; validation/test labels are never
        # inspected here.
        self.regression_target_stats = self.task.fit_regression_target_stats(
            self.train_loader
        )
        self.multilabel_target_stats = self.task.fit_multilabel_target_stats(
            self.train_loader
        )
        self._log_training_setup()

    def _log_training_setup(self) -> None:
        """Log parameter counts and optimizer LR groups for debugging."""
        # Parameter counts.
        encoder_total = sum(p.numel() for p in self.model.parameters())
        encoder_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        task_total = sum(p.numel() for p in self.task.parameters())
        task_trainable = sum(p.numel() for p in self.task.parameters() if p.requires_grad)
        print(
            f"[Finetune][Setup] Encoder params: {encoder_trainable:,} trainable / {encoder_total:,} total | "
            f"Task params: {task_trainable:,} trainable / {task_total:,} total"
        )
        # Optimizer LR groups.
        if isinstance(self.optimizers, dict):
            for opt_name, opt in self.optimizers.items():
                for i, group in enumerate(opt.param_groups):
                    n_params = sum(p.numel() for p in group["params"])
                    print(
                        f"[Finetune][Setup] Optimizer '{opt_name}' group {i}: "
                        f"lr={group.get('lr', '?')}, wd={group.get('weight_decay', '?')}, "
                        f"params={n_params:,}"
                    )
        # Monitor info.
        print(
            f"[Finetune][Setup] Monitor: {self.monitor_name} (mode={self.monitor_mode}) | "
            f"Method: {self.finetune_method} | Freeze: {self.freeze_pretrained_effective}"
        )

    def _apply_pretrained_model_cfg(self) -> None:
        """Restore the full model config subtree from the pretrained checkpoint.

        This ensures architecture-shaping parameters (nested blocks like
        ``model.fagcn``, ``model.nodeformer``, etc.) are
        aligned before the encoder is constructed, preventing silent partial
        loads via ``strict=False``. Graph pooling has no checkpoint state and
        is the one downstream readout override allowed to survive when the
        caller explicitly requested it.
        """
        if not isinstance(self.pretrain_cfg, dict):
            return
        model_cfg = self.pretrain_cfg.get("model", {}) or {}
        if not model_cfg:
            return
        explicit_keys = set(get_explicit_cfg_keys(self.cfg))
        downstream_overrides = {
            "graph_pooling": self.cfg.model.graph_pooling,
        } if "model.graph_pooling" in explicit_keys else {}
        self._merge_dict_into_cfg(self.cfg.model, model_cfg)
        for key, value in downstream_overrides.items():
            setattr(self.cfg.model, key, value)

    def _validate_explicit_pretrain_config_overrides(self) -> None:
        """Reject publication model/pretrain claims absent from the checkpoint."""
        requested = cfg_to_dict(self.cfg)
        for key in get_explicit_cfg_keys(self.cfg):
            if key == "model.graph_pooling":
                continue
            if not key.startswith(("model.", "pretrain.")):
                continue
            requested_found, requested_value = self._lookup_plain_config(
                requested, key
            )
            saved_found, saved_value = self._lookup_plain_config(
                self.pretrain_cfg, key
            )
            if not requested_found or not saved_found:
                raise CampaignProvenanceError(
                    "Publication scientific override is missing from the pretrained "
                    f"checkpoint config: {key}."
                )
            if requested_value != saved_value:
                raise CampaignProvenanceError(
                    "Publication scientific override disagrees with the pretrained "
                    f"checkpoint config at {key}: requested {requested_value!r}, "
                    f"saved {saved_value!r}."
                )

    def _validate_loaded_pretrain_config(self) -> None:
        expected = str(self.cfg.provenance.pretrained_checkpoint_config_sha256)
        actual = canonical_config_sha256(self.pretrain_cfg)
        if actual != expected:
            raise CampaignProvenanceError(
                "Pretrained checkpoint config digest mismatch: "
                f"expected {expected}, found {actual}."
            )
        self._validate_explicit_pretrain_config_overrides()

    @staticmethod
    def _lookup_plain_config(config: Mapping[str, Any], dotted: str) -> tuple[bool, Any]:
        current: Any = config
        for part in dotted.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return False, None
            current = current[part]
        return True, current

    @staticmethod
    def _merge_dict_into_cfg(cfg_node, source_dict: dict) -> None:
        """Recursively merge *source_dict* values into a YACS CfgNode.

        YACS ``CfgNode`` is a ``dict`` subclass, so ``isinstance(..., dict)``
        correctly distinguishes nested config blocks from scalar values.
        """
        for key, value in source_dict.items():
            if not hasattr(cfg_node, key):
                continue  # skip keys unknown to the current config schema
            if isinstance(value, dict):
                sub_node = getattr(cfg_node, key, None)
                if sub_node is not None and isinstance(sub_node, dict):
                    FinetuneRunner._merge_dict_into_cfg(sub_node, value)
            else:
                setattr(cfg_node, key, value)

    def _get_target_dataset_cfg(self) -> dict[str, Any]:
        """Resolve target dataset settings, allowing finetune overrides."""
        return resolve_target_dataset_cfg(self.cfg, self.task_cls)

    def _init_monitoring(self) -> None:
        """Resolve checkpoint monitoring for finetuning and prompt-based methods."""
        label_dim = int(getattr(self.cfg.finetune.dataset, "label_dim", 1) or 1)
        few_shot_no_val = self._few_shot_without_validation()
        method_name = str(getattr(self, "finetune_method", getattr(self.cfg.finetune, "method", "")) or "").lower()
        method_name = method_name.replace("-", "_")
        spec = resolve_finetune_monitor_spec(
            self.cfg,
            task_level=str(getattr(self, "task_level_raw", self.cfg.finetune.dataset.task_level) or "").lower(),
            label_dim=label_dim,
            few_shot_without_validation=few_shot_no_val,
            task_cls=getattr(self, "task_cls", None),
            method_name=method_name,
        )
        self.monitor_name = spec.name
        self.monitor_mode = spec.mode
        self.best_metric = spec.best_metric
        self.best_epoch = None
        self.best_metrics: dict[str, float] = {}

    def _is_few_shot_split(self) -> bool:
        """Return True when split uses few-shot form (shots_per_class, val_weight, test_weight)."""
        return is_few_shot_split(getattr(self, "split", None))

    def _few_shot_without_validation(self) -> bool:
        split = getattr(self, "split", None)
        if not self._is_few_shot_split() or split is None or len(split) != 3:
            return False
        try:
            val = float(split[1])
            test = float(split[2])
        except Exception:
            return False
        return val <= 1e-12 and test > 0.0

    @staticmethod
    def _is_valid_shot_count(value: object) -> bool:
        """Validate the 'shots' field used by few-shot split shorthand."""
        try:
            numeric = float(value)
            return numeric.is_integer() and numeric >= 1.0
        except Exception:
            return isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 1

    def _make_loaders(self, induced: bool):
        return make_workflow_loaders(
            dataset=self.dataset,
            dataset_name=self.cfg.finetune.dataset.name,
            task_level_raw=self.task_level_raw,
            effective_task_level=self.effective_task_level,
            batch_size=self.cfg.finetune.batch_size,
            num_workers=self.cfg.finetune.num_workers,
            split=self.split,
            seed=self.cfg.seed,
            induced=induced,
            split_root=shared_split_root(self.cfg),
            return_split_provenance=is_publication_campaign(self.cfg),
        )

    def _effective_freeze_pretrained(self) -> bool:
        """Resolve whether to freeze the pretrained encoder.

        Resolution order:
        1. Method-specific ``freeze_encoder`` (e.g. ``finetune.supervised.freeze_encoder``).
        2. Default: ``True`` (freeze). Prompt methods enforce this via
           ``requires_frozen_encoder``; supervised overrides to ``False``.
        """
        finetune_cfg = getattr(self.cfg, "finetune", None)
        method = str(getattr(finetune_cfg, "method", "supervised") or "supervised").lower().replace("-", "_")

        # Check method-specific freeze overrides.
        method_cfg = getattr(finetune_cfg, method, None)
        if method_cfg is not None:
            raw_freeze = getattr(method_cfg, "freeze_encoder", None)
            if raw_freeze is not None:
                return to_bool(raw_freeze)

        # Default: freeze (safe for prompt methods that require it).
        return True

    def _maybe_freeze_pretrained_encoder(self) -> None:
        freeze = bool(getattr(self, "freeze_pretrained_effective", self._effective_freeze_pretrained()))
        # Validate: methods that require a frozen encoder must not run with freeze=False.
        if not freeze and getattr(self.task, "requires_frozen_encoder", False):
            method = str(getattr(self.cfg.finetune, "method", "supervised") or "supervised")
            raise ValueError(
                f"Finetune method '{method}' requires a frozen encoder "
                f"(set finetune.{method}.freeze_encoder=True)."
            )
        if not freeze:
            return
        frozen_params = 0
        for param in self.model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_params += param.numel()
        print(f"[Finetune] Frozen encoder parameters (count={frozen_params})")

    def _apply_frozen_encoder_mode(self, model: torch.nn.Module) -> None:
        """Apply the task-declared frozen encoder mode policy.

        The effective mode is resolved through
        :meth:`_FinetuneBase.resolve_frozen_encoder_mode`, which methods
        with cfg-dependent behaviour override as a classmethod.  This
        keeps the class-attribute contract validated by
        ``__init_subclass__`` authoritative and avoids instance-level
        mutation of ``self.frozen_encoder_mode``.  The returned value is
        re-validated against :data:`_VALID_FROZEN_ENCODER_MODES` because
        ``__init_subclass__`` only checks the class attribute, not
        override return values.
        """
        task_cls = type(self.task)
        mode = task_cls.resolve_frozen_encoder_mode(self.cfg)
        if mode not in _VALID_FROZEN_ENCODER_MODES:
            raise ValueError(
                f"{task_cls.__name__}.resolve_frozen_encoder_mode returned "
                f"'{mode}'; expected one of {sorted(_VALID_FROZEN_ENCODER_MODES)}."
            )
        if mode == "eval":
            model.eval()
        elif mode == "train_bn_eval":
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        else:  # mode == "train"
            model.train()

    def _resolve_split(self) -> tuple:
        ds_cfg = self.cfg.finetune.dataset
        split = resolve_workflow_split(
            getattr(ds_cfg, "fixed_split", None),
            default=(0.1, 0.1, 0.8),
        )
        # Finetune-specific: validate few-shot split constraints.
        if self._is_valid_shot_count(split[0]):
            try:
                val_ratio = float(split[1])
                test_ratio = float(split[2])
            except Exception as exc:
                raise ValueError("Few-shot split must be numeric: (shots_per_class, val_weight, test_weight).") from exc
            if val_ratio < 0.0 or test_ratio < 0.0:
                raise ValueError("Few-shot split requires non-negative val/test weights.")
            if (val_ratio + test_ratio) <= 0.0:
                raise ValueError("Few-shot split requires val_weight + test_weight > 0.")
        return split

    def _build_run_name(self) -> str:
        dataset_cfg = self.cfg.finetune.dataset
        raw_task_level = getattr(self, "task_level_raw", getattr(dataset_cfg, "task_level", ""))
        return build_finetune_run_name_from_cfg(
            self.cfg,
            split=self.split,
            task_level_raw=raw_task_level,
            task_cls=self.task_cls,
            finetune_method=self.finetune_method,
            pretrained_run_name=self.pretrained_run_name,
            freeze_pretrained_effective=self.freeze_pretrained_effective,
        )

    def _artifact_stem(self) -> str:
        return compact_artifact_stem(self.run_name)

    def _checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, f"{self._artifact_stem()}.pt")

    def _load_finetune_checkpoint(self, path: str, *, map_location):
        """Load task checkpoints safely at the publication trust boundary."""
        if is_publication_campaign(self.cfg):
            return torch.load(
                path,
                map_location=map_location,
                weights_only=True,
            )
        # Historical development artifacts may contain legacy Python objects.
        # Keep that explicitly non-publication compatibility path unchanged.
        return torch.load(path, map_location=map_location)

    def _legacy_checkpoint_path(self) -> str | None:
        """Return the pre-compaction path when its basename is filesystem-safe."""
        run_name = getattr(self, "run_name", None)
        if not run_name:
            return None
        basename = f"{run_name}.pt"
        if len(os.fsencode(basename)) > 255:
            return None
        path = os.path.join(self.run_dir, basename)
        return None if path == self._checkpoint_path() else path

    def _existing_checkpoint_path(self) -> str | None:
        candidates = [self._checkpoint_path()]
        legacy_path = self._legacy_checkpoint_path()
        if legacy_path is not None:
            candidates.append(legacy_path)
        publication = is_publication_campaign(self.cfg)
        existing_candidates = [path for path in candidates if os.path.isfile(path)]
        if publication:
            log_candidates = [self._log_path()]
            legacy_log = self._legacy_log_path()
            if legacy_log is not None:
                log_candidates.append(legacy_log)
            existing_logs = [path for path in log_candidates if os.path.isfile(path)]
            if not existing_candidates and not existing_logs:
                return None
            if not existing_candidates:
                raise CampaignProvenanceError(
                    "Publication artifact is incomplete: a companion log exists "
                    "without its checkpoint. Refusing to overwrite it."
                )
            if len(existing_candidates) != 1:
                raise CampaignProvenanceError(
                    "Conflicting publication checkpoints exist for the same "
                    f"logical run: {existing_candidates}."
                )
            ckpt_path = existing_candidates[0]
            expected_log = (
                self._log_path()
                if ckpt_path == self._checkpoint_path()
                else self._legacy_log_path()
            )
            if expected_log is None or not os.path.isfile(expected_log):
                raise CampaignProvenanceError(
                    "Publication checkpoint has no readable companion log: "
                    f"{ckpt_path}. Refusing to overwrite it."
                )
            unexpected_logs = [path for path in existing_logs if path != expected_log]
            if unexpected_logs:
                raise CampaignProvenanceError(
                    "Conflicting publication logs exist for the same logical "
                    f"run: {[expected_log, *unexpected_logs]}."
                )
        else:
            ckpt_path = next(iter(existing_candidates), None)
            if ckpt_path is None:
                return None

        if not publication:
            return ckpt_path

        try:
            payload = self._load_finetune_checkpoint(
                ckpt_path,
                map_location="cpu",
            )
        except Exception as exc:
            raise CampaignProvenanceError(
                f"Unreadable publication checkpoint {ckpt_path!r}."
            ) from exc
        if not isinstance(payload, dict):
            raise CampaignProvenanceError(
                f"Invalid publication checkpoint payload: {ckpt_path}."
            )
        extra = payload.get("extra") or {}
        try:
            with open(expected_log, "r", encoding="utf-8") as handle:
                log_payload = json.load(handle)
        except Exception as exc:
            raise CampaignProvenanceError(
                f"Unreadable publication companion log {expected_log!r}."
            ) from exc
        if not isinstance(log_payload, dict):
            raise CampaignProvenanceError(
                f"Invalid publication companion log payload: {expected_log}."
            )
        current_provenance = self._provenance_payload(
            provenance_complete=True,
        )
        require_split_digests = bool(self.split_content_digests)
        checkpoint_provenance = extra.get("provenance")
        log_provenance = log_payload.get("provenance")
        checkpoint_matches = saved_provenance_matches(
            current_provenance,
            checkpoint_provenance,
            require_split_digests=require_split_digests,
        )
        log_matches = saved_provenance_matches(
            current_provenance,
            log_provenance,
            require_split_digests=require_split_digests,
        )
        if (
            not checkpoint_matches
            or not log_matches
            or checkpoint_provenance != log_provenance
            or payload.get("provenance_complete") is not True
            or extra.get("provenance_complete") is not True
            or log_payload.get("provenance_complete") is not True
        ):
            raise CampaignProvenanceError(
                "Publication checkpoint/log provenance is missing, incomplete, "
                f"or mismatched for {ckpt_path}. Refusing to overwrite it."
            )
        return ckpt_path

    def get_checkpoint_path_for_metrics(self) -> str:
        if self._reused_checkpoint_path is not None:
            return self._reused_checkpoint_path
        return self._checkpoint_path()

    def _log_path(self) -> str:
        log_dir = getattr(self.cfg.finetune, "log_dir", "")
        if log_dir:
            return os.path.join(log_dir, self.run_group, f"{self._artifact_stem()}_log.json")
        return os.path.join(self.run_dir, f"{self._artifact_stem()}_log.json")

    def _legacy_log_path(self) -> str | None:
        run_name = getattr(self, "run_name", None)
        if not run_name:
            return None
        basename = f"{run_name}_log.json"
        if len(os.fsencode(basename)) > 255:
            return None
        log_dir = getattr(self.cfg.finetune, "log_dir", "")
        directory = os.path.join(log_dir, self.run_group) if log_dir else self.run_dir
        path = os.path.join(directory, basename)
        return None if path == self._log_path() else path

    def _provenance_payload(
        self,
        *,
        provenance_complete: bool = False,
    ) -> dict[str, Any]:
        return campaign_provenance_payload(
            self.cfg,
            split_digests=self.split_content_digests,
            provenance_complete=provenance_complete,
        )

    def _pretrained_from_payload(self) -> dict[str, Any]:
        """Build the one canonical pretrained-source receipt for both artifacts."""
        return {
            "run_name": self.pretrained_run_name,
            "checkpoint": self.pretrained_checkpoint,
            "checkpoint_sha256": str(
                getattr(
                    getattr(self.cfg, "provenance", None),
                    "pretrained_checkpoint_sha256",
                    "",
                )
                or ""
            ),
            "checkpoint_config_sha256": str(
                getattr(
                    getattr(self.cfg, "provenance", None),
                    "pretrained_checkpoint_config_sha256",
                    "",
                )
                or ""
            ),
            "dataset": self.pretrain_dataset_name,
            "task_level": self.pretrain_task_level,
            "method": self.pretrain_method,
            "model": getattr(self, "pretrain_model_name", None),
            "induced": getattr(self, "pretrain_induced", None),
            "freeze_pretrained_effective": bool(
                getattr(self, "freeze_pretrained_effective", False)
            ),
            "dataset_meta": self.pretrain_dataset_meta,
        }

    def _save_training_log(self) -> None:
        extra = {
            "artifact_identity": {
                "logical_run_name": self.run_name,
                "artifact_stem": self._artifact_stem(),
            },
            "pretrained_from": self._pretrained_from_payload(),
        }
        multilabel_stats = getattr(self, "multilabel_target_stats", None)
        if multilabel_stats:
            extra["multilabel_target_stats"] = multilabel_stats
        regression_stats = getattr(self, "regression_target_stats", None)
        if regression_stats:
            extra["regression_target_stats"] = regression_stats
        initialization = getattr(
            getattr(self, "task", None), "initialization_metadata", None
        )
        if initialization is not None:
            extra["task_initialization"] = dict(initialization)
        if is_publication_campaign(self.cfg):
            extra["provenance"] = self._provenance_payload()
            extra["provenance_complete"] = False
        save_training_log(
            path=self._log_path(),
            cfg=self.cfg,
            dataset_meta=self.dataset_meta,
            history=self.train_history,
            best_info={
                "epoch": self.best_epoch,
                "metric": self.best_metric,
                "monitor": self.monitor_name,
                "metrics": self.best_metrics,
            },
            extra=extra,
        )

    def _is_improved(self, metric: float) -> bool:
        return is_metric_improved(metric, self.best_metric, self.monitor_mode)

    def _resolve_monitor_value(
        self,
        train_loss: float,
        train_logs: dict[str, float],
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
    ) -> float:
        """
        Resolve the scalar used for checkpoint selection and early stopping.

        Returns NaN when the requested metric is absent (NaN never counts as
        an improvement); ``fit`` writes a final-epoch fallback checkpoint if
        the monitor never resolves over the whole run.
        """
        return resolve_monitor_value(
            self.monitor_name,
            train_loss=train_loss,
            train_logs=train_logs,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )

    def _monitor_uses_train_split(self) -> bool:
        """Train-only monitor metrics do not require validation passes."""
        return monitor_uses_train_split(self.monitor_name)

    def _checkpoint_extra(self) -> dict:
        """Checkpoint payload beyond the encoder: provenance + task state.

        With a frozen encoder, the learned parameters (prompts, GPPT tokens,
        prototype banks, classifier heads) live on the task module, not on
        ``self.model`` — without ``task_state`` the saved artifact cannot
        reproduce the reported model.
        """
        extra = {
            "artifact_identity": {
                "logical_run_name": self.run_name,
                "artifact_stem": self._artifact_stem(),
            },
            "pretrained_from": self._pretrained_from_payload(),
        }
        task = getattr(self, "task", None)
        if task is not None and hasattr(task, "state_dict"):
            extra["task_state"] = task.state_dict()
        multilabel_stats = getattr(self, "multilabel_target_stats", None)
        if multilabel_stats:
            extra["multilabel_target_stats"] = multilabel_stats
        regression_stats = getattr(self, "regression_target_stats", None)
        if regression_stats:
            extra["regression_target_stats"] = regression_stats
        initialization = getattr(task, "initialization_metadata", None)
        if initialization is not None:
            extra["task_initialization"] = dict(initialization)
        if is_publication_campaign(self.cfg):
            extra["provenance"] = self._provenance_payload()
            extra["provenance_complete"] = False
        return extra

    def _save_best_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        train_logs: dict[str, float],
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
        monitor_value: float,
    ) -> None:
        if self.monitor_name is None:
            # Keep checkpoint/log artifacts aligned with the latest epoch when
            # explicit monitoring is disabled.
            self.best_metric = float(monitor_value)
            self.best_epoch = epoch
        else:
            if not self._is_improved(monitor_value):
                return
            self.best_metric = float(monitor_value)
            self.best_epoch = epoch
        metrics = {
            "train_loss": train_loss,
            "best_epoch": epoch,
            **train_logs,
            **val_metrics,
            **test_metrics,
        }
        if self.monitor_name is not None:
            metrics[self.monitor_name] = float(monitor_value)
        save_checkpoint(
            path=self._checkpoint_path(),
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            cfg=self.cfg,
            dataset_meta=self.dataset_meta,
            metrics=metrics,
            extra=self._checkpoint_extra(),
        )
        self._checkpoint_written_this_run = True
        self.best_metrics = metrics
        self._save_training_log()
        if self.monitor_name is None:
            print(f"[Finetune] Checkpoint updated at epoch={epoch} (monitor disabled).")
        else:
            print(f"[Finetune] Best epoch updated: epoch={epoch} {self.monitor_name}={monitor_value:.4f}")

    def train_epoch(self):
        result = run_step_epoch(
            model=self.model,
            task=self.task,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            grad_clip=float(getattr(self.cfg.finetune, "grad_clip", 0.0) or 0.0),
            skip_model_train=self.freeze_pretrained_effective,
        )
        return result

    def _evaluate_split(self, loader, prefix: str, mask_attr: str) -> dict[str, float]:
        return runner_evaluate_split(
            model=self.model,
            task=self.task,
            loader=loader,
            device=self.device,
            prefix=prefix,
            mask_attr=mask_attr,
            task_type=resolve_task_type(getattr(self.cfg.finetune.dataset, "task_type", None)),
        )

    def _evaluate_selected_test_checkpoint(self) -> dict[str, float]:
        """Restore the selected checkpoint and evaluate the held-out test once."""
        if self.test_loader is None or not self._checkpoint_written_this_run:
            return {}

        payload = self._load_finetune_checkpoint(
            self._checkpoint_path(),
            map_location=self.device,
        )
        self.model.load_state_dict(payload["model_state"])
        task_state = (payload.get("extra") or {}).get("task_state")
        if task_state is not None:
            self.task.load_state_dict(task_state)
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)

        if self.task_supports_epoch:
            test_metrics = self.task.evaluate_split(
                model=self.model,
                loader=self.test_loader,
                device=self.device,
                prefix="test",
                mask_attr="test_mask",
            )
        else:
            test_metrics = self._evaluate_split(
                self.test_loader, prefix="test", mask_attr="test_mask"
            )

        metrics = dict(payload.get("metrics") or {})
        metrics.update(test_metrics)
        extra = dict(payload.get("extra") or {})
        extra["task_state"] = self.task.state_dict()
        initialization = getattr(self.task, "initialization_metadata", None)
        if initialization is not None:
            extra["task_initialization"] = dict(initialization)
        save_checkpoint(
            path=self._checkpoint_path(),
            model=self.model,
            optimizer=self.optimizer,
            epoch=int(payload.get("epoch", self.best_epoch or 0)),
            cfg=payload.get("cfg", self.cfg),
            dataset_meta=payload.get("dataset", self.dataset_meta),
            metrics=metrics,
            extra=extra,
        )
        self.best_metrics = metrics
        return test_metrics

    def _maybe_refit_selected_task(self) -> None:
        """Let an epoch-based task refit on all training data after selection."""
        hook = getattr(self.task, "refit_after_selection", None)
        if hook is None or self.best_epoch is None or not self._checkpoint_written_this_run:
            return
        if self.freeze_pretrained_effective:
            self._apply_frozen_encoder_mode(self.model)
        result = hook(
            model=self.model,
            loader=self.train_loader,
            device=self.device,
            selected_epochs=int(self.best_epoch),
        )
        if not result:
            return

        self.optimizers = result["optimizers"]
        self.optimizer = self.optimizers["primary"]
        metrics = dict(self.best_metrics)
        metrics["selection_refit_epochs"] = int(result["epochs"])
        metrics["selection_refit_train_loss"] = float(result["train_loss"])
        extra = self._checkpoint_extra()
        extra["selection_refit"] = {
            "epochs": int(result["epochs"]),
            "uses_full_train_split": True,
            "selection_monitor": self.monitor_name,
        }
        save_checkpoint(
            path=self._checkpoint_path(),
            model=self.model,
            optimizer=self.optimizer,
            epoch=int(self.best_epoch),
            cfg=self.cfg,
            dataset_meta=self.dataset_meta,
            metrics=metrics,
            extra=extra,
        )
        self.best_metrics = metrics
        self._save_training_log()

    def _finalize_publication_artifacts(self) -> None:
        """Atomically mark the selected checkpoint/log pair publishable.

        Every checkpoint and log emitted during optimization carries an
        explicit false marker. Only this end-of-run path revalidates the live
        source/design/pretrained inputs and replaces each complete artifact
        atomically with a true marker. A crash between the two replacements
        leaves a mismatched pair which reuse rejects fail-closed.
        """
        if not is_publication_campaign(self.cfg):
            return

        # Recompute all live hashes after optimization, before either artifact
        # can be marked complete.
        prepare_campaign_provenance(self.cfg)
        self.pretrained_checkpoint = bind_pretrained_checkpoint_provenance(
            self.cfg,
            self.pretrained_checkpoint,
        )

        checkpoint_path = self._checkpoint_path()
        log_path = self._log_path()
        if not os.path.isfile(checkpoint_path) or not os.path.isfile(log_path):
            raise CampaignProvenanceError(
                "Cannot finalize publication provenance without both the selected "
                f"checkpoint and companion log: {checkpoint_path}, {log_path}."
            )
        try:
            checkpoint_payload = self._load_finetune_checkpoint(
                checkpoint_path,
                map_location="cpu",
            )
        except Exception as exc:
            raise CampaignProvenanceError(
                f"Cannot finalize unreadable publication checkpoint {checkpoint_path!r}."
            ) from exc
        try:
            with open(log_path, "r", encoding="utf-8") as handle:
                log_payload = json.load(handle)
        except Exception as exc:
            raise CampaignProvenanceError(
                f"Cannot finalize unreadable publication log {log_path!r}."
            ) from exc
        if not isinstance(checkpoint_payload, dict) or not isinstance(log_payload, dict):
            raise CampaignProvenanceError(
                "Cannot finalize malformed publication checkpoint/log payloads."
            )

        checkpoint_extra = checkpoint_payload.get("extra") or {}
        current_incomplete = self._provenance_payload(provenance_complete=False)
        checkpoint_provenance = checkpoint_extra.get("provenance")
        log_provenance = log_payload.get("provenance")
        if (
            checkpoint_extra.get("provenance_complete") is not False
            or checkpoint_payload.get("provenance_complete") is not False
            or log_payload.get("provenance_complete") is not False
            or checkpoint_provenance != log_provenance
            or not saved_provenance_matches(
                current_incomplete,
                checkpoint_provenance,
                require_split_digests=True,
                require_complete=False,
            )
        ):
            raise CampaignProvenanceError(
                "Cannot finalize publication artifacts whose interim provenance "
                "is missing or mismatched."
            )

        complete_provenance = self._provenance_payload(provenance_complete=True)
        checkpoint_extra = dict(checkpoint_extra)
        checkpoint_extra["provenance"] = complete_provenance
        checkpoint_extra["provenance_complete"] = True
        checkpoint_payload["extra"] = checkpoint_extra
        checkpoint_payload["provenance_complete"] = True
        log_payload["provenance"] = complete_provenance
        log_payload["provenance_complete"] = True

        save_torch_atomic(checkpoint_path, checkpoint_payload)
        save_json_atomic(log_path, log_payload)
        self.provenance_complete = True

    def fit(self) -> None:
        if getattr(self, "_skip_due_to_existing_checkpoint", False):
            return

        self._setup()

        ensure_dir(path=self.cfg.finetune.checkpoint_dir)
        ensure_dir(self.run_dir)
        ckpt_path = self._existing_checkpoint_path()
        if self.cfg.finetune.skip_if_exists and ckpt_path is not None:
            self._skip_due_to_existing_checkpoint = True
            self._reused_checkpoint_path = ckpt_path
            self.provenance_complete = is_publication_campaign(self.cfg)
            print(f"[Finetune] Checkpoint already exists, skipping: {ckpt_path}")
            return
        if is_publication_campaign(self.cfg) and ckpt_path is not None:
            raise CampaignProvenanceError(
                "A complete publication artifact already exists and "
                "finetune.skip_if_exists is false; refusing to overwrite it."
            )
        self._reused_checkpoint_path = None
        if is_publication_campaign(self.cfg):
            # Durable preflight record: if the job crashes before its first
            # selected checkpoint, the exact source/design/split identity is
            # still available in the run log.
            self._save_training_log()
        if os.path.isfile(self._checkpoint_path()):
            print(f"[Finetune] Overwriting existing checkpoint: {self._checkpoint_path()}")

        patience = int(getattr(self.cfg.finetune, "early_stopping", 0) or 0)
        # Respect the task class declaration: some methods run fixed-epoch training.
        if patience > 0 and not getattr(self.task, "supports_early_stopping", True):
            method_name = str(getattr(self.cfg.finetune, "method", "supervised") or "supervised")
            print(f"[Finetune][{method_name}] Disabling early stopping (fixed-epoch training).")
            patience = 0
        epochs_since_improvement = 0

        # Allow task to override effective epochs when needed.
        total_epochs = self.cfg.finetune.epochs
        if hasattr(self.task, "get_effective_epochs"):
            total_epochs = self.task.get_effective_epochs(total_epochs)
            if total_epochs != self.cfg.finetune.epochs:
                print(f"[Finetune] Using effective epochs: {total_epochs} (original: {self.cfg.finetune.epochs})")

        last_epoch = 0
        for epoch in range(1, total_epochs + 1):
            last_epoch = epoch
            start = time.time()
            monitor_on_train = self._monitor_uses_train_split()
            # Apply the task-declared frozen encoder mode policy before each
            # epoch so methods don't need to manage encoder mode manually.
            if self.freeze_pretrained_effective:
                self._apply_frozen_encoder_mode(self.model)
            if self.task_supports_epoch:
                train_loss, train_logs = self.task.train_epoch(
                    model=self.model,
                    loader=self.train_loader,
                    device=self.device,
                    optimizers=self.optimizers,
                )
                if hasattr(self.task, "on_epoch_end"):
                    self.task.on_epoch_end(self.model, self.train_loader, self.device)
                # Skip val evaluation when monitor metric is train-split only.
                if monitor_on_train:
                    val_metrics = {}
                else:
                    val_metrics = self.task.evaluate_split(
                        model=self.model,
                        loader=self.val_loader,
                        device=self.device,
                        prefix="val",
                        mask_attr="val_mask",
                    )
                test_metrics = {}
            else:
                train_loss, train_logs = self.train_epoch()
                # Skip val evaluation when monitor metric is train-split only.
                if monitor_on_train:
                    val_metrics = {}
                else:
                    val_metrics = self._evaluate_split(self.val_loader, prefix="val", mask_attr="val_mask")
                test_metrics = {}

            duration = time.time() - start
            log_parts = [
                f"[Finetune][Epoch {epoch}/{total_epochs}]",
                f"train_loss={train_loss:.4f}",
            ]
            for metrics in (train_logs, val_metrics, test_metrics):
                for k, v in metrics.items():
                    if not should_print_metric(k):
                        continue
                    log_parts.append(f"{k}={v:.4f}")
            log_parts.append(f"time={duration:.1f}s")
            print(" ".join(log_parts))

            merged_metrics = merge_epoch_metrics(train_logs, val_metrics, test_metrics)
            self.train_history.append(
                {
                    "epoch": epoch,
                    "loss": float(train_loss),
                    "duration_sec": float(duration),
                    "metrics": merged_metrics,
                }
            )

            monitor_value = self._resolve_monitor_value(
                train_loss=train_loss,
                train_logs=train_logs,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )
            improved = True if self.monitor_name is None else self._is_improved(monitor_value)

            self._save_best_checkpoint(
                epoch=epoch,
                train_loss=train_loss,
                train_logs=train_logs,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                monitor_value=monitor_value,
            )

            if patience > 0:
                if improved:
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(
                        f"[Finetune] Early stopping at epoch {epoch} "
                        f"(no improvement in {patience} epochs)."
                    )
                    break

        if last_epoch > 0 and not self._checkpoint_written_this_run:
            # The monitor never produced a usable value (e.g. empty val split
            # → NaN every epoch). Persist the final state so the run leaves a
            # checkpoint and a training log instead of silently nothing.
            last = self.train_history[-1] if self.train_history else {}
            self.best_epoch = int(last.get("epoch", last_epoch))
            fallback_metrics = {
                "train_loss": float(last.get("loss", float("nan"))),
                "best_epoch": self.best_epoch,
                **(last.get("metrics") or {}),
            }
            print(
                "[Finetune] WARNING: monitor "
                f"{self.monitor_name!r} never resolved to a finite value; "
                "saving final-epoch state as fallback checkpoint."
            )
            save_checkpoint(
                path=self._checkpoint_path(),
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.best_epoch,
                cfg=self.cfg,
                dataset_meta=self.dataset_meta,
                metrics=fallback_metrics,
                extra={**self._checkpoint_extra(), "fallback_save": True},
            )
            self._checkpoint_written_this_run = True
            self.best_metrics = fallback_metrics
            self.best_metric = float("nan")
            self._save_training_log()

        self._maybe_refit_selected_task()
        final_test_metrics = self._evaluate_selected_test_checkpoint()
        if final_test_metrics:
            print(
                "[Finetune] Selected-checkpoint test: "
                + " ".join(f"{key}={value:.4f}" for key, value in final_test_metrics.items())
            )
        if self.train_history:
            self._save_training_log()
        self._finalize_publication_artifacts()

        if self.monitor_name is not None:
            print(
                f"[Finetune] Complete. Best {self.monitor_name}: "
                f"{self.best_metric:.4f} at epoch {self.best_epoch}."
            )
        else:
            print(f"[Finetune] Complete. Final epoch: {last_epoch} (early stopping disabled).")
        for metric_name in _BEST_TEST_METRICS_TO_PRINT:
            metric_value = self.best_metrics.get(metric_name)
            if metric_value is not None:
                print(f"[Finetune] Best-epoch {metric_name}={metric_value:.4f}")
