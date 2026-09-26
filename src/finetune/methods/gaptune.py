"""GapTune / GapTune+ prompt finetuning (paper Sec. 3, Algorithm 1)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from src.data_loader import create_dataset
from src.finetune.encoders.gaptune import get_gaptune_driver, resolve_gaptune_spec
from src.finetune.encoders.nodeformer_fixed import build_fixed_projections
from src.finetune.methods.gaptune_proxy import build_proxy_source_graphs
from src.finetune.prompts.gaptune import (
    GATES,
    MIXTURES,
    PROMPT_LOCATIONS,
    QUERY_MODES,
    VALUE_MODES,
    ContextGapPrompt,
)
from src.finetune.readouts import readout_input_dim, task_native_readout
from src.finetune.registry import register
from src.finetune.task_base import FinetuneTask
from src.finetune.task_heads import TaskAwareObjective
from src.utils.config_helpers import (
    build_prompt_head_optimizer,
    cfg_default,
    optimizer_variant_tags,
    tag_if_nondefault,
)
from src.utils.dataset_helpers import (
    is_few_shot_split,
    read_effective_task_level,
    shared_induced_root,
    shared_split_root,
)
from src.utils.monitoring import make_monitor_spec
from src.utils.parsing import resolve_task_type, resolve_workflow_split, to_bool
from src.utils.pool import get_batch_vector
from src.utils.supervised_eval import evaluate_epoch_split
from src.utils.training import run_epoch_loop

# Paper D.2 seed streams s, s+1, ...: initializer and observation sampler.
# Minibatch order (stream s+5) has no dedicated generator: the shared runner
# draws it from the global seed s (finetuner ``set_seed`` + unseeded
# train-loader shuffle). It is identical across GapTune arms (prompt options,
# GapTune/GapTune+, proxy mode and budget): ``validate_encoder`` draws the same
# from the global RNG in every arm and source preparation runs under
# ``preserve_loader_rng``.
_INIT_STREAM = 0
_OBSERVATION_STREAM = 1
# Source graphs are encoded in a random order until every observation pool
# holds this many times its cap, then each pool is subsampled uniformly.
_SOURCE_POOL_FACTOR = 4
_SOURCE_BATCH_GRAPHS = 64

# (tag, key) of every non-optimizer option; tags appear only when non-default.
_VARIANT_KEYS = (
    ("k", "num_queries"),
    ("tc", "tau_c"),
    ("tp", "tau_p"),
    ("oeps", "obs_eps"),
    ("v", "value_mode"),
    ("loc", "prompt_locations"),
    ("q", "query_mode"),
    ("mix", "mixture"),
    ("g", "gate"),
    ("sn", "source_max_nodes"),
    ("sm", "source_max_messages"),
    ("gc", "grad_clip"),
)
_PROXY_VARIANT_KEYS = (
    ("pxm", "mode"),
    ("pxb", "num_graphs"),
    ("pxn", "num_nodes"),
    ("pxu", "updates"),
    ("pxlr", "lr"),
    ("pxgc", "grad_clip"),
    ("pxlx", "lambda_x"),
    ("pxla", "lambda_a"),
    ("pxrho", "density"),
    ("pxe", "final_edges"),
    ("pxh", "edge_hidden"),
    ("pxr", "feature_radius"),
    ("pxts", "tau_start"),
    ("pxte", "tau_end"),
    ("pxpos", "edgepred_pos_pairs"),
    ("pxneg", "edgepred_neg_pairs"),
    ("pxlog", "edgepred_logit"),
    ("pxct", "graphcl_tau"),
    ("pxed", "graphcl_edge_drop"),
    ("pxfm", "graphcl_feature_mask"),
)


def _nondefault_tags(node, prefix: str, keys) -> list[str]:
    tags = []
    for tag, key in keys:
        default = cfg_default(f"{prefix}.{key}")
        value = getattr(node, key)
        value = to_bool(value) if isinstance(default, bool) else type(default)(value)
        tags.append(tag_if_nondefault(tag, value, default))
    return tags


def load_pretraining_graphs(cfg, ds_cfg, *, seed: int):
    """The pretraining collection, built with the pretrain runner's dataset arguments.

    *ds_cfg* is the ``pretrain.dataset`` block the checkpoint was trained with.
    """
    raw_level = str(ds_cfg["task_level"]).lower()
    induced = to_bool(ds_cfg.get("induced", False))
    # Unsupervised pretraining uses the full dataset without split files.
    split_root, split = shared_split_root(cfg), None
    if raw_level == "edge" and induced:
        split = resolve_workflow_split(ds_cfg.get("fixed_split"), default=(0.8, 0.1, 0.1))
    elif raw_level != "graph":
        split_root = ""
    return create_dataset(
        name=ds_cfg["name"],
        root=ds_cfg["root"],
        task_level=raw_level,
        feat_reduction=ds_cfg["feat_reduction"],
        feat_reduction_dim=ds_cfg.get("feat_reduction_svd_dim", ds_cfg.get("feat_reduction_dim", 100)),
        persist_feature_svd=ds_cfg["feat_reduction"],
        feature_svd_dir=ds_cfg.get("feature_svd_dir", "data/feature_svd"),
        induced=induced,
        induced_min_size=ds_cfg.get("induced_min_size", 10),
        induced_max_size=ds_cfg.get("induced_max_size", 30),
        induced_max_hops=ds_cfg.get("induced_max_hops", 5),
        cache_induced=ds_cfg.get("cache_induced", True),
        split_root=split_root,
        induced_root=shared_induced_root(cfg, ds_cfg.get("induced_root", "")),
        split=split,
        seed=seed,
    )


def sample_source_observations(
    driver,
    model: nn.Module,
    graphs,
    *,
    projections,
    device,
    max_nodes: int,
    max_messages: int,
    generator: torch.Generator,
) -> tuple[dict, dict]:
    """One fixed label-free source sample (paper D.2, Table 17).

    Graphs are encoded with the frozen UNPROMPTED driver in a random order
    until every pool holds ``_SOURCE_POOL_FACTOR`` times its cap (or the
    collection is exhausted); each pool is then subsampled uniformly without
    replacement to at most ``max_nodes`` node observations and
    ``max_messages`` messages per layer.  Returns banks keyed ``N``,
    ``M1``..``ML`` (on CPU) and sampling metadata.
    """
    order = torch.randperm(len(graphs), generator=generator).tolist()
    pools: dict[str, list[torch.Tensor]] = {}
    sizes: dict[str, int] = {}
    encoded = 0
    for start in range(0, len(order), _SOURCE_BATCH_GRAPHS):
        chunk = [graphs[index] for index in order[start : start + _SOURCE_BATCH_GRAPHS]]
        data = Batch.from_data_list([Data(x=g.x.float(), edge_index=g.edge_index) for g in chunk]).to(device)
        with torch.no_grad():
            obs = driver.forward(model, data, collect=True, projections=projections).obs
        pools.setdefault("N", []).append(obs.h0.cpu())
        for layer, layer_obs in enumerate(obs.layers, start=1):
            pools.setdefault(f"M{layer}", []).append(layer_obs.messages.cpu())
        encoded += len(chunk)
        sizes = {key: sum(part.size(0) for part in parts) for key, parts in pools.items()}
        if all(size >= _SOURCE_POOL_FACTOR * (max_nodes if key == "N" else max_messages) for key, size in sizes.items()):
            break
    banks = {}
    for key, parts in pools.items():
        pool = torch.cat(parts)
        cap = max_nodes if key == "N" else max_messages
        if pool.size(0) > cap:
            pool = pool[torch.randperm(pool.size(0), generator=generator)[:cap]]
        banks[key] = pool
    metadata = {
        "graphs_total": len(graphs),
        "graphs_encoded": encoded,
        "pool_sizes": sizes,
        "bank_sizes": {key: int(bank.size(0)) for key, bank in banks.items()},
    }
    return banks, metadata


@register("gaptune")
class FinetuneGapTune(FinetuneTask):
    """GapTune (proxy source graphs) / GapTune+ (pretraining graphs as source).

    The frozen vanilla encoder is replayed by the GapTune drivers: a detached
    unprompted pass gives each prediction graph's observations and
    descriptors, ``ContextGapPrompt`` turns source/target context gaps into
    node and message prompts, and a second, prompted pass (Eq. 13-16) feeds
    the task-native readout (Eq. 17) and an affine head.  Prompt and head
    dims come from the encoder, so they are built in ``validate_encoder``;
    the runner's ``prepare_with_encoder`` hook fills the fixed source bank.
    Evaluation reads the retained source contexts (Eq. 18), refreshed after
    every training epoch so a saved checkpoint carries ``C*_s``.
    """

    requires_frozen_encoder = True
    supports_early_stopping = True
    frozen_encoder_mode = "eval"

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        # Node and edge queries run on induced subgraphs (paper Sec. 3.5).
        cls.require_graph_level_batches(cfg, method_label="GapTune")
        resolve_gaptune_spec(cfg)
        gt_cfg = cfg.finetune.gaptune
        for key, choices in (
            ("value_mode", VALUE_MODES),
            ("prompt_locations", PROMPT_LOCATIONS),
            ("query_mode", QUERY_MODES),
            ("mixture", MIXTURES),
            ("gate", GATES),
        ):
            if getattr(gt_cfg, key) not in choices:
                raise ValueError(f"[GapTune] {key}={getattr(gt_cfg, key)!r} is invalid; expected one of {list(choices)}.")
        if gt_cfg.value_mode == "free" and gt_cfg.query_mode != "shared":
            raise ValueError("[GapTune] value_mode=free replaces the query bank; query_mode must be shared.")
        for key in ("num_queries", "source_max_nodes", "source_max_messages"):
            if int(getattr(gt_cfg, key)) < 1:
                raise ValueError(f"[GapTune] {key} must be >= 1, got {getattr(gt_cfg, key)}")
        for key in ("tau_c", "tau_p", "obs_eps"):
            if float(getattr(gt_cfg, key)) <= 0:
                raise ValueError(f"[GapTune] {key} must be > 0, got {getattr(gt_cfg, key)}")

    @classmethod
    def run_tag(cls, cfg) -> str:
        return f"plus{int(to_bool(cfg.finetune.gaptune.plus))}"

    @classmethod
    def variant_tag(cls, cfg) -> str:
        gt_cfg = cfg.finetune.gaptune
        tags = _nondefault_tags(gt_cfg, "finetune.gaptune", _VARIANT_KEYS)
        if not to_bool(gt_cfg.plus):
            tags.extend(_nondefault_tags(gt_cfg.proxy, "finetune.gaptune.proxy", _PROXY_VARIANT_KEYS))
        tags.extend(optimizer_variant_tags(gt_cfg, "gaptune"))
        return "-".join(tag for tag in tags if tag)

    @classmethod
    def resolve_default_monitor(cls, cfg):
        # Paper D.2: ToxCast selects on ROC-AUC; the auto policy picks micro-F1.
        ds_cfg = cfg.finetune.dataset
        split = resolve_workflow_split(getattr(ds_cfg, "fixed_split", None), default=(0.1, 0.1, 0.8))
        has_validation = not (is_few_shot_split(split) and float(split[1]) <= 1e-12)
        multilabel = (
            resolve_task_type(getattr(ds_cfg, "task_type", None)) == "classification"
            and int(getattr(ds_cfg, "label_dim", 1) or 1) > 1
        )
        if multilabel and has_validation:
            return make_monitor_spec("val_auc", "max")
        return None

    def __init__(self, cfg):
        super().__init__(cfg)
        self.method_cfg = cfg.finetune.gaptune
        ds_cfg = cfg.finetune.dataset
        self.task_level_raw = str(getattr(ds_cfg, "task_level_raw", None) or ds_cfg.task_level).lower()
        self.objective = TaskAwareObjective(cfg, task_level=read_effective_task_level(ds_cfg))
        self.task_type = self.objective.task_type
        self.driver = None
        self.projections = None
        self.initialization_metadata = None

    def validate_encoder(self, model: nn.Module) -> None:
        """Select the driver and build the head and prompt from the encoder dims."""
        self.driver = get_gaptune_driver(model)
        gt_cfg = self.method_cfg
        obs_dims = self.driver.observation_dims(model)
        desc_dims = self.driver.descriptor_dims(model)
        generator = torch.Generator().manual_seed(int(self.cfg.seed) + _INIT_STREAM)
        # Head first, so every prompt ablation shares the head initialization.
        repr_dim = desc_dims["N"] - obs_dims["N"]  # dim(h^(L)), Eq. 10
        self.head = nn.Linear(readout_input_dim(repr_dim, self.task_level_raw), self.objective.output_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / (self.head.in_features + self.head.out_features))
            self.head.weight.uniform_(-bound, bound, generator=generator)
            self.head.bias.zero_()
        self.prompt = ContextGapPrompt(
            obs_dims,
            desc_dims,
            num_queries=int(gt_cfg.num_queries),
            tau_c=float(gt_cfg.tau_c),
            tau_p=float(gt_cfg.tau_p),
            obs_eps=float(gt_cfg.obs_eps),
            value_mode=gt_cfg.value_mode,
            prompt_locations=gt_cfg.prompt_locations,
            query_mode=gt_cfg.query_mode,
            mixture=gt_cfg.mixture,
            generator=generator,
        )
        device = next(model.parameters()).device
        self.head.to(device)
        self.prompt.to(device)
        if resolve_gaptune_spec(self.cfg).model_name == "nodeformer":
            # Fixed per-layer projections for every pass (paper B.10).
            self.projections = build_fixed_projections(model, int(self.cfg.seed))

    def prepare_with_encoder(self, *, model, device, pretrain_cfg, pretrain_extra) -> None:
        """Collect and fix the source observations (Algorithm 1, lines 3-8).

        *pretrain_cfg* / *pretrain_extra* are the checkpoint's plain ``cfg``
        and ``extra`` the runner already loaded; source contexts use no
        target data.
        """
        if not self.prompt.types:
            return
        gt_cfg = self.method_cfg
        model.eval()
        if to_bool(gt_cfg.plus):
            # Same dataset/feature settings the checkpoint was pretrained with.
            ds_cfg = (pretrain_cfg.get("pretrain") or {}).get("dataset") or self.cfg.pretrain.dataset
            graphs = load_pretraining_graphs(self.cfg, ds_cfg, seed=int(pretrain_cfg.get("seed", self.cfg.seed)))
            source = {"source": "pretraining", "source_dataset": str(ds_cfg["name"])}
        else:
            graphs, proxy_metadata = build_proxy_source_graphs(
                cfg=self.cfg,
                model=model,
                driver=self.driver,
                pretrain_cfg=pretrain_cfg,
                pretrain_extra=pretrain_extra,
                device=device,
            )
            source = {"source": "proxy", "proxy": proxy_metadata}
        sampling = self.set_source_from_graphs(model, graphs, device)
        self.initialization_metadata = {**source, **sampling}
        print(
            f"[Finetune][GapTune] Source bank from {source['source']}: {sampling['bank_sizes']} "
            f"({sampling['graphs_encoded']}/{sampling['graphs_total']} graphs encoded)"
        )

    def set_source_from_graphs(self, model: nn.Module, graphs, device) -> dict:
        """Fix the source bank sampled from *graphs* and its retained contexts; returns the sampling metadata."""
        banks, sampling = self.sample_source_banks(model, graphs, device)
        self.prompt.set_source_bank(banks)
        self.prompt.refresh_retained()
        return sampling

    def sample_source_banks(self, model: nn.Module, graphs, device) -> tuple[dict, dict]:
        """Observations of *graphs* sampled with this task's source caps and sampler stream, and the metadata."""
        gt_cfg = self.method_cfg
        observation_seed = int(self.cfg.seed) + _OBSERVATION_STREAM
        banks, sampling = sample_source_observations(
            self.driver,
            model,
            graphs,
            projections=self.projections,
            device=device,
            max_nodes=int(gt_cfg.source_max_nodes),
            max_messages=int(gt_cfg.source_max_messages),
            generator=torch.Generator().manual_seed(observation_seed),
        )
        return banks, {**sampling, "observation_seed": observation_seed}

    def parameters_to_optimize(self):
        return [p for p in self.prompt.parameters() if p.requires_grad] + list(self.head.parameters())

    def build_optimizers(self, model: nn.Module):
        del model
        optimizers = build_prompt_head_optimizer(
            method_cfg=self.method_cfg,
            prompt_params=[p for p in self.prompt.parameters() if p.requires_grad],
            head_params=self.head.parameters(),
            base_lr=float(self.cfg.finetune.lr),
            base_wd=float(self.cfg.finetune.weight_decay),
        )
        if self.method_cfg.gate == "nonnegative":
            optimizers["primary"].register_step_post_hook(lambda *_: self.prompt.project_gates())
        return optimizers

    def encode(self, model: nn.Module, data, reference=None) -> torch.Tensor:
        """Unprompted detached pass, then the prompted pass (Algorithm 1, lines 13-28).

        *reference* is that unprompted pass (``collect=True``) when the caller
        already has it, e.g. a fixed full graph encoded once.
        """
        if reference is None:
            with torch.no_grad():
                reference = self.driver.forward(
                    model, data, collect=bool(self.prompt.types), projections=self.projections
                )
        if not self.prompt.types:
            return reference.node_repr
        node_prompt, message_prompts = self.prompt(
            reference.obs, get_batch_vector(data), use_retained=not self.training
        )
        return self.driver.forward(
            model,
            data,
            node_prompt=node_prompt,
            message_prompts=message_prompts,
            projections=self.projections,
        ).node_repr

    def _forward(self, model, data, device, return_outputs: bool = False):
        data = data.to(device)
        representations, labels = task_native_readout(self.encode(model, data), data, self.task_level_raw)
        return self.objective.loss_from_logits(
            logits=self.head(representations), labels=labels, return_outputs=return_outputs
        )

    def train_epoch(self, model, loader, device, optimizers=None):
        optimizer = optimizers.get("primary") if isinstance(optimizers, dict) else optimizers
        if optimizer is None:
            raise ValueError("GapTune requires an optimizer.")
        # Encoder mode is handled by the runner via _apply_frozen_encoder_mode.
        self.train()
        metric_key = "train_mae" if self.task_type == "regression" else "train_acc"

        def forward_fn(data, device):
            loss, primary = self._forward(model, data, device)
            return loss, {metric_key: float(primary)}

        result = run_epoch_loop(
            forward_fn=forward_fn,
            loader=loader,
            optimizer=optimizer,
            device=device,
            grad_clip=float(self.method_cfg.grad_clip),
        )
        self.prompt.refresh_retained()
        return result

    def evaluate_split(self, model, loader, device, prefix: str, mask_attr: str) -> dict[str, float]:
        del mask_attr  # induced/graph batches carry one label per prediction graph
        model.eval()
        self.eval()

        def _forward(data, device):
            loss, _primary, logits, labels = self._forward(model, data, device, return_outputs=True)
            return loss, logits, labels

        return evaluate_epoch_split(
            forward_fn=_forward,
            loader=loader,
            device=device,
            prefix=prefix,
            task_type=self.task_type,
        )
