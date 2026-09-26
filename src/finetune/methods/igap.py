"""IGAP prompt finetuning method (Yan et al., WWW 2024)."""

from __future__ import annotations

from functools import partial

from torch import nn

from src.finetune.methods.supervised import FinetuneSupervised
from src.finetune.prompts.igap import IGAPLabelPrompt, IGAPPrompt
from src.finetune.registry import register
from src.finetune.task_base import FinetuneTask
from src.utils.config_helpers import build_prompt_head_optimizer, cfg_default, optimizer_variant_tags, tag_if_nondefault
from src.utils.parsing import resolve_task_type, to_bool
from src.utils.pool import get_batch_vector, pool_nodes
from src.utils.supervised_eval import evaluate_epoch_split
from src.utils.training import run_epoch_loop


@register("igap")
class FinetuneIGAP(FinetuneTask):
    """IGAP with graph-signal, spectral-alignment and label prompts.

    The frozen encoder is wrapped in the projection sandwich
    ``Z~ = U_K P_t U_K^T f(A, U_K P_t^T U_K^T X~)`` and exposed as an encoder
    view ``data -> (Z~, pool(Z~))``, so the readout and objective are the
    shared ``FinetuneSupervised`` / ``TaskAwareObjective`` path used by GPF.
    The head is the paper's new 2-layer ReLU MLP; for classification it
    feeds the cosine label prompt ``P_l`` (tau = ``igap.tau``), for
    regression it outputs the targets directly (paper "No P_l, end2end").
    """

    requires_frozen_encoder = True
    supports_early_stopping = True
    frozen_encoder_mode = "eval"

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        # The spectral prompt needs one Laplacian basis per prediction graph.
        cls.require_graph_level_batches(cfg, method_label="IGAP")
        igap_cfg = cfg.finetune.igap
        if int(igap_cfg.num_signal_prompts) < 1:
            raise ValueError(f"[IGAP] num_signal_prompts must be >= 1, got {igap_cfg.num_signal_prompts}")
        if int(igap_cfg.num_eigvecs) < 1:
            raise ValueError(f"[IGAP] num_eigvecs must be >= 1, got {igap_cfg.num_eigvecs}")
        if float(igap_cfg.tau) <= 0:
            raise ValueError(f"[IGAP] tau must be > 0, got {igap_cfg.tau}")

    @classmethod
    def variant_tag(cls, cfg) -> str:
        igap_cfg = getattr(getattr(cfg, "finetune", None), "igap", None)
        if igap_cfg is None:
            return ""
        tags = [
            tag_if_nondefault(name, cast(getattr(igap_cfg, key)), cast(cfg_default(f"finetune.igap.{key}")))
            for name, key, cast in (
                ("sp", "num_signal_prompts", int),
                ("k", "num_eigvecs", int),
                ("tau", "tau", float),
                ("nops", "use_signal_prompt", to_bool),
                ("nopt", "use_spectral_prompt", to_bool),
                ("nopl", "use_label_prompt", to_bool),
                ("hhd", "head_hidden_dim", int),
            )
        ]
        tags.extend(optimizer_variant_tags(igap_cfg, "igap"))
        return "-".join(tag for tag in tags if tag)

    def __init__(self, cfg):
        super().__init__(cfg)
        igap_cfg = cfg.finetune.igap
        self.supervised_head = FinetuneSupervised(cfg)
        self.task_type = resolve_task_type(getattr(self.supervised_head, "task_type", None))

        self.prompt = IGAPPrompt(
            in_channels=int(cfg.model.in_dim),
            num_signal_prompts=int(igap_cfg.num_signal_prompts),
            num_eigvecs=int(igap_cfg.num_eigvecs),
            use_signal_prompt=to_bool(igap_cfg.use_signal_prompt),
            use_spectral_prompt=to_bool(igap_cfg.use_spectral_prompt),
        )

        objective = self.supervised_head.objective
        repr_dim = int(cfg.model.out_dim)
        hidden_dim = int(igap_cfg.head_hidden_dim) if int(igap_cfg.head_hidden_dim) > 0 else repr_dim
        use_label_prompt = to_bool(igap_cfg.use_label_prompt) and self.task_type != "regression"
        layers = [
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim if use_label_prompt else objective.output_dim),
        ]
        if use_label_prompt:
            layers.append(
                IGAPLabelPrompt(
                    dim=repr_dim,
                    num_outputs=objective.output_dim,
                    pairwise=objective.is_multilabel_classification or objective.output_dim == 1,
                    tau=float(igap_cfg.tau),
                )
            )
        self.supervised_head.classifier = nn.Sequential(*layers)

    def encode(self, model, data):
        """Encoder view: ``(Z~, pool(Z~))`` with the IGAP prompts applied."""
        batch = get_batch_vector(data)
        eigvecs = self.prompt.basis(data.edge_index, batch, data.x.dtype)
        prompted = data.clone()
        prompted.x = self.prompt.prompt_input(data.x, eigvecs, batch)
        node_repr, _ = model(prompted)
        node_repr = self.prompt.align_output(node_repr, eigvecs, batch)
        return node_repr, pool_nodes(node_repr, batch, mode=self.cfg.model.graph_pooling)

    def parameters_to_optimize(self):
        return list(self.prompt.parameters()) + list(self.supervised_head.classifier.parameters())

    def build_optimizers(self, model):
        return build_prompt_head_optimizer(
            method_cfg=self.cfg.finetune.igap,
            prompt_params=self.prompt.parameters(),
            head_params=self.supervised_head.classifier.parameters(),
            base_lr=float(self.cfg.finetune.lr),
            base_wd=float(self.cfg.finetune.weight_decay),
        )

    def train_epoch(self, model, loader, device, optimizers=None):
        optimizer = optimizers.get("primary") if isinstance(optimizers, dict) else optimizers
        if optimizer is None:
            raise ValueError("IGAP requires an optimizer.")

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
