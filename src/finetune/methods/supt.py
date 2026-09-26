"""SUPT prompt finetuning method."""

from __future__ import annotations

import torch

from src.finetune.methods.gpf import FinetuneGPF
from src.finetune.prompts.supt import SUPTPrompt
from src.finetune.registry import register
from src.finetune.methods.supervised import FinetuneSupervised
from src.finetune.task_base import FinetuneTask
from src.utils.config_helpers import build_prompt_head_optimizer, cfg_default, optimizer_variant_tags, tag_if_nondefault
from src.utils.parsing import resolve_task_type, to_bool
from src.utils.supervised_eval import evaluate_epoch_split
from src.utils.training import run_epoch_loop


@register("supt")
class FinetuneSUPT(FinetuneTask):
    """SUPT-soft / SUPT-hard finetuning with a prompted input and supervised task head.

    The official SUPT code targets graph-level molecular property prediction.
    As for GPF, the prompt is applied to ``data.x`` before the frozen encoder
    and the shared supervised head reads out node/edge/graph representations.
    The frozen-encoder mode, fixed-epoch policy and head construction mirror
    GPF, but induced node/edge subgraphs are kept: each query subgraph is one
    graph for the GCN scorer and the per-graph top-k of the hard variant.
    """

    requires_frozen_encoder = True
    supports_early_stopping = False  # same fixed-epoch policy as GPF
    frozen_encoder_mode = "train_bn_eval"  # dropout active, BN stats frozen

    @classmethod
    def resolve_frozen_encoder_mode(cls, cfg) -> str:
        supt_cfg = getattr(getattr(cfg, "finetune", None), "supt", None)
        freeze_bn = (
            to_bool(getattr(supt_cfg, "freeze_encoder_bn_when_frozen", True))
            if supt_cfg is not None
            else True
        )
        return "train_bn_eval" if freeze_bn else "train"

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        cls.require_node_or_graph_batches(cfg, method_label="SUPT")

        supt_cfg = getattr(getattr(cfg, "finetune", None), "supt", None)
        if supt_cfg is None:
            return

        variant = str(supt_cfg.variant)
        if variant not in {"soft", "hard"}:
            raise ValueError(f"[SUPT] variant must be 'soft' or 'hard', got {variant!r}")
        if int(supt_cfg.num_bases) < 1:
            raise ValueError(f"[SUPT] num_bases must be >= 1, got {supt_cfg.num_bases}")
        # topk reads ratio >= 1 as an absolute node count, so keep it a fraction.
        if not (0.0 < float(supt_cfg.ratio) < 1.0):
            raise ValueError(f"[SUPT] ratio must be in (0, 1), got {supt_cfg.ratio}")
        hard_score = str(supt_cfg.hard_score)
        if hard_score not in {"tanh", "graph_softmax"}:
            raise ValueError(f"[SUPT] hard_score must be 'tanh' or 'graph_softmax', got {hard_score!r}")
        if int(supt_cfg.head_layers) < 1:
            raise ValueError(f"[SUPT] head_layers must be >= 1, got {supt_cfg.head_layers}")
        if not (0.0 <= float(supt_cfg.head_dropout) < 1.0):
            raise ValueError(f"[SUPT] head_dropout must be in [0, 1), got {supt_cfg.head_dropout}")

    @classmethod
    def variant_tag(cls, cfg) -> str:
        supt_cfg = getattr(getattr(cfg, "finetune", None), "supt", None)
        if supt_cfg is None:
            return ""

        tags = []
        variant = str(supt_cfg.variant)
        if variant != cfg_default("finetune.supt.variant"):
            tags.append(variant)
        t = tag_if_nondefault("k", int(supt_cfg.num_bases), int(cfg_default("finetune.supt.num_bases")))
        if t:
            tags.append(t)
        if variant == "hard":
            # ratio / hard_score only act on the hard variant.
            t = tag_if_nondefault("r", float(supt_cfg.ratio), float(cfg_default("finetune.supt.ratio")))
            if t:
                tags.append(t)
            hard_score = str(supt_cfg.hard_score)
            if hard_score != cfg_default("finetune.supt.hard_score"):
                tags.append(f"score_{hard_score}")
        if to_bool(supt_cfg.orth_loss):
            tags.append("orth")
        if not to_bool(supt_cfg.gcn_bias):
            tags.append("nogcnbias")
        head_layers = int(supt_cfg.head_layers)
        if head_layers > 1:
            tags.append(f"head{head_layers}")
            head_hidden_dim = int(supt_cfg.head_hidden_dim)
            if head_hidden_dim > 0:
                tags.append(f"hhd{head_hidden_dim}")
            t = tag_if_nondefault(
                "hdo", float(supt_cfg.head_dropout), float(cfg_default("finetune.supt.head_dropout"))
            )
            if t:
                tags.append(t)
        if not to_bool(supt_cfg.freeze_encoder_bn_when_frozen):
            tags.append("bntrain")
        tags.extend(optimizer_variant_tags(supt_cfg, "supt"))
        return "-".join(tags)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.supervised_head = FinetuneSupervised(cfg)
        self.task_type = resolve_task_type(getattr(self.supervised_head, "task_type", None))

        method_cfg = cfg.finetune.supt
        self.prompt_in_dim = int(getattr(cfg.model, "in_dim", 0) or 0)
        if self.prompt_in_dim <= 0:
            raise ValueError("SUPT requires model.in_dim > 0.")
        self.prompt = SUPTPrompt(
            in_channels=self.prompt_in_dim,
            num_bases=int(method_cfg.num_bases),
            variant=str(method_cfg.variant),
            ratio=float(method_cfg.ratio),
            hard_score=str(method_cfg.hard_score),
            orth_loss=to_bool(method_cfg.orth_loss),
            gcn_bias=to_bool(method_cfg.gcn_bias),
        )

        self.head_layers = max(1, int(method_cfg.head_layers))
        self.head_hidden_dim = int(method_cfg.head_hidden_dim)
        self.head_dropout = float(method_cfg.head_dropout)
        self._maybe_upgrade_prediction_head()

    # Same prediction-head construction as GPF.
    _maybe_upgrade_prediction_head = FinetuneGPF._maybe_upgrade_prediction_head

    def parameters_to_optimize(self):
        return list(self.prompt.parameters()) + list(self.supervised_head.classifier.parameters())

    def build_optimizers(self, model):
        return build_prompt_head_optimizer(
            method_cfg=self.cfg.finetune.supt,
            prompt_params=self.prompt.parameters(),
            head_params=self.supervised_head.classifier.parameters(),
            base_lr=float(self.cfg.finetune.lr),
            base_wd=float(self.cfg.finetune.weight_decay),
        )

    def _apply_prompt(self, data, device):
        data = data.to(device)
        x = getattr(data, "x", None)
        if x is None:
            raise ValueError("SUPT requires node features in `data.x`.")
        if int(x.size(-1)) != self.prompt_in_dim:
            raise ValueError(
                f"SUPT prompt dim mismatch: expected {self.prompt_in_dim}, got {int(x.size(-1))}. "
                "Ensure finetune dataset features match pretrained model input dimension."
            )
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        prompted = data.clone()
        # The scorer sees exactly the edge_index handed to the frozen encoder
        # (induced edge subgraphs already exclude the query edge).
        prompted.x = self.prompt.add(prompted.x, prompted.edge_index, batch)
        return prompted

    def train_epoch(self, model, loader, device, optimizers=None):
        optimizer = optimizers.get("primary") if isinstance(optimizers, dict) else optimizers
        if optimizer is None:
            raise ValueError("SUPT requires an optimizer.")

        # Encoder mode is handled by the runner via _apply_frozen_encoder_mode.
        self.train()
        metric_key = "train_mae" if self.task_type == "regression" else "train_acc"

        def forward_fn(data, device):
            prompted = self._apply_prompt(data, device)
            loss, primary = self.supervised_head.evaluate(
                model=model, data=prompted, device=device,
                mask_attr="train_mask",
            )
            if self.prompt.orth_loss:
                loss = loss + self.prompt.orthogonal_loss()
            return loss, {metric_key: float(primary)}

        return run_epoch_loop(
            forward_fn=forward_fn,
            loader=loader,
            optimizer=optimizer,
            device=device,
            grad_clip=float(getattr(getattr(self.cfg, "finetune", None), "grad_clip", 0.0) or 0.0),
        )

    def evaluate_split(self, model, loader, device, prefix: str, mask_attr: str) -> dict[str, float]:
        model.eval()
        self.eval()

        def _forward(data, device):
            prompted = self._apply_prompt(data, device)
            loss, _primary, logits, labels = self.supervised_head.evaluate(
                model=model, data=prompted, device=device,
                mask_attr=mask_attr, return_outputs=True,
            )
            return loss, logits, labels

        return evaluate_epoch_split(
            forward_fn=_forward,
            loader=loader,
            device=device,
            prefix=prefix,
            task_type=self.task_type,
        )
