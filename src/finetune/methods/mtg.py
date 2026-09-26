"""MTG message-tuning finetuning method (Chen et al., ICML 2026)."""

from __future__ import annotations

from functools import partial

from src.finetune.encoders.edgeprompt import resolve_edgeprompt_prompt_spec, supported_edgeprompt_backbones
from src.finetune.encoders.mtg import forward_with_mtg
from src.finetune.encoders.nodeformer_fixed import build_fixed_projections
from src.finetune.methods.supervised import FinetuneSupervised
from src.finetune.prompts.mtg import MTGPrompt
from src.finetune.registry import register
from src.finetune.task_base import FinetuneTask
from src.model.nodeformer import NodeFormerEncoder
from src.utils.config_helpers import build_prompt_head_optimizer, cfg_default, optimizer_variant_tags, tag_if_nondefault
from src.utils.parsing import resolve_task_type
from src.utils.supervised_eval import evaluate_epoch_split
from src.utils.training import run_epoch_loop


@register("mtg")
class FinetuneMTG(FinetuneTask):
    """MTG: message prototypes fused before every frozen layer, plus a head.

    The frozen encoder is replayed by ``forward_with_mtg`` and exposed as an
    encoder view ``data -> (node_repr, graph_repr)``, so the readout and
    objective are the shared ``FinetuneSupervised`` / ``TaskAwareObjective``
    path used by GPF (logits head; the official double-softmax head is not
    copied).  Prototype dims per layer follow the EdgePrompt ``dim_list``.
    NodeFormer convs use fixed projections seeded ``cfg.seed + 1009 * l``.
    """

    requires_frozen_encoder = True
    supports_early_stopping = True
    frozen_encoder_mode = "eval"  # official MTG backbones run without dropout

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        cls.require_node_or_graph_batches(cfg, method_label="MTG")
        name = str(cfg.model.name).lower()
        if name not in supported_edgeprompt_backbones():
            raise ValueError(
                f"[MTG] unsupported backbone {name!r}; supported: {list(supported_edgeprompt_backbones())}"
            )
        if int(cfg.finetune.mtg.num_prototypes) < 1:
            raise ValueError(f"[MTG] num_prototypes must be >= 1, got {cfg.finetune.mtg.num_prototypes}")

    @classmethod
    def variant_tag(cls, cfg) -> str:
        mtg_cfg = getattr(getattr(cfg, "finetune", None), "mtg", None)
        if mtg_cfg is None:
            return ""
        tags = [
            tag_if_nondefault("m", int(mtg_cfg.num_prototypes), int(cfg_default("finetune.mtg.num_prototypes")))
        ]
        tags.extend(optimizer_variant_tags(mtg_cfg, "mtg"))
        return "-".join(tag for tag in tags if tag)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.supervised_head = FinetuneSupervised(cfg)
        self.task_type = resolve_task_type(getattr(self.supervised_head, "task_type", None))
        self.prompt = MTGPrompt(
            dims=resolve_edgeprompt_prompt_spec(cfg).dim_list,
            num_prototypes=int(cfg.finetune.mtg.num_prototypes),
        )
        self._projections = None  # NodeFormer only; rebuilt from cfg.seed, not saved

    def encode(self, model, data):
        """Encoder view: ``(node_repr, graph_repr)`` with the MTG prototypes fused in."""
        if self._projections is None and isinstance(model, NodeFormerEncoder):
            self._projections = build_fixed_projections(model, int(self.cfg.seed))
        return forward_with_mtg(model, data, self.prompt.fuse, projections=self._projections)

    def parameters_to_optimize(self):
        return list(self.prompt.parameters()) + list(self.supervised_head.classifier.parameters())

    def build_optimizers(self, model):
        return build_prompt_head_optimizer(
            method_cfg=self.cfg.finetune.mtg,
            prompt_params=self.prompt.parameters(),
            head_params=self.supervised_head.classifier.parameters(),
            base_lr=float(self.cfg.finetune.lr),
            base_wd=float(self.cfg.finetune.weight_decay),
        )

    def train_epoch(self, model, loader, device, optimizers=None):
        optimizer = optimizers.get("primary") if isinstance(optimizers, dict) else optimizers
        if optimizer is None:
            raise ValueError("MTG requires an optimizer.")

        # Encoder mode is handled by the runner via _apply_frozen_encoder_mode.
        self.train()
        metric_key = "train_mae" if self.task_type == "regression" else "train_acc"

        def forward_fn(data, device):
            loss, primary = self.supervised_head.evaluate(
                model=partial(self.encode, model), data=data, device=device,
                mask_attr="train_mask",
            )
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
            loss, _primary, logits, labels = self.supervised_head.evaluate(
                model=partial(self.encode, model), data=data, device=device,
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
