"""GapTune context-gap prompts (paper Sec. 3.3-3.4, App. B.6, C.1-C.3).

Per observation type ``q`` in ``{N, M1, ..., ML}`` (restricted by
``prompt_locations``):

- ``nu(z) = z / sqrt(||z||^2 + eps^2)``;
- shared-query pooling (Eq. 8-9)
  ``c_{a,k} = sum_i softmax_i(q_k . nu(z_{a,i}) / tau_c) z_{a,i}`` over the
  fixed source bank (``a = s``) or over ONE prediction graph's observations
  (``a = t``: segment softmax per graph, never across the minibatch);
- gaps ``d_k = c_{s,k} - c_{t,k}``;
- local relevance (Eq. 11) ``a_{o,k} = softmax_k(w_k . nu(zeta_o) / tau_p)``;
- prompts (Eq. 12) ``p_o = sum_k a_{o,k} tanh(theta_k) d_k``.

A type is active only when its source bank is non-empty; a graph's target
collection holds the observations of every prompted object of that graph,
so it is non-empty whenever the graph has objects of that type.  Retained
inference state (Eq. 18, Prop. B.3): ``retained_source_context`` [K, d] and,
for free values, ``retained_source_norm``, recomputed from the current
queries by ``refresh_retained``; ``use_retained=True`` reads them instead of
pooling the bank.

Ablation switches (App. C): ``value_mode`` gap | target | source |
paired_mean | free (learned ``V`` [K, d] replacing the queries, clipped per
graph to ``max_s ||z|| + max_t ||z||``); ``prompt_locations``
node_message | node | message | none; ``query_mode`` shared | frozen |
untied (separate source/target banks); ``mixture`` local | uniform (1/K) |
global (learned logits [K] per type).  The nonnegative-gate ablation is
``project_gates`` after every optimizer step.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as graph_softmax

VALUE_MODES = ("gap", "target", "source", "paired_mean", "free")
PROMPT_LOCATIONS = ("node_message", "node", "message", "none")
QUERY_MODES = ("shared", "frozen", "untied")
MIXTURES = ("local", "uniform", "global")
GATES = ("signed", "nonnegative")


def normalize_observations(z: torch.Tensor, eps: float) -> torch.Tensor:
    """``nu(z) = z / sqrt(||z||^2 + eps^2)`` row-wise."""
    return z / torch.sqrt(z.pow(2).sum(dim=-1, keepdim=True) + eps * eps)


def pool_source_context(queries: torch.Tensor, bank: torch.Tensor, tau_c: float, eps: float) -> torch.Tensor:
    """Eq. 8-9 over the whole source bank: ``[K, d]``."""
    weights = torch.softmax(queries @ normalize_observations(bank, eps).t() / tau_c, dim=-1)
    return weights @ bank


def pool_target_contexts(
    queries: torch.Tensor,
    z: torch.Tensor,
    graph_id: torch.Tensor,
    num_graphs: int,
    tau_c: float,
    eps: float,
) -> torch.Tensor:
    """Eq. 8-9 separately for every prediction graph: ``[G, K, d]``."""
    weights = graph_softmax(normalize_observations(z, eps) @ queries.t() / tau_c, graph_id, num_nodes=num_graphs)
    return torch.stack(
        [
            scatter(weights[:, k : k + 1] * z, graph_id, dim=0, dim_size=num_graphs, reduce="sum")
            for k in range(queries.size(0))
        ],
        dim=1,
    )


def _gaussian(rows: int, cols: int, generator: torch.Generator | None) -> torch.Tensor:
    """Independent ``N(0, 1/cols)`` entries (paper Table 17)."""
    return torch.randn(rows, cols, generator=generator) / math.sqrt(cols)


class ObservationTypePrompt(nn.Module):
    """Queries, relevance vectors and gates of one observation type."""

    def __init__(
        self,
        obs_dim: int,
        desc_dim: int,
        *,
        num_queries: int,
        tau_c: float,
        tau_p: float,
        obs_eps: float,
        value_mode: str,
        query_mode: str,
        mixture: str,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.tau_c = float(tau_c)
        self.tau_p = float(tau_p)
        self.obs_eps = float(obs_eps)
        self.value_mode = value_mode
        self.mixture = mixture
        num = self.num_queries
        source_queries = None
        if value_mode == "free":
            # Free values replace the K x d query bank (same trainable count).
            self.free_values = nn.Parameter(_gaussian(num, obs_dim, generator))
        else:
            self.queries = nn.Parameter(_gaussian(num, obs_dim, generator), requires_grad=query_mode != "frozen")
            if query_mode == "untied":
                source_queries = nn.Parameter(_gaussian(num, obs_dim, generator))
        self.source_queries = source_queries
        if mixture == "local":
            self.relevance = nn.Parameter(_gaussian(num, desc_dim, generator))
        elif mixture == "global":
            self.mixture_logits = nn.Parameter(torch.zeros(num))
        self.gates = nn.Parameter(torch.zeros(num))
        self.register_buffer("source_bank", torch.zeros(0, obs_dim), persistent=False)
        self.register_buffer("source_empty", torch.tensor(True))
        self.register_buffer("retained_source_context", torch.zeros(num, obs_dim))
        self.register_buffer("retained_source_norm", torch.zeros(()))

    def set_source_bank(self, bank: torch.Tensor) -> None:
        self.source_bank = bank.to(device=self.gates.device, dtype=self.gates.dtype)
        self.source_empty.fill_(bank.size(0) == 0)

    def source_context(self) -> torch.Tensor:
        queries = self.queries if self.source_queries is None else self.source_queries
        return pool_source_context(queries, self.source_bank, self.tau_c, self.obs_eps)

    @torch.no_grad()
    def refresh_retained(self) -> None:
        if self.source_bank.size(0) == 0:
            return
        self.retained_source_norm.copy_(self.source_bank.norm(dim=-1).max())
        if self.value_mode != "free":
            self.retained_source_context.copy_(self.source_context())

    def values(self, z: torch.Tensor, graph_id: torch.Tensor, num_graphs: int, *, use_retained: bool) -> torch.Tensor:
        """Per-graph prompt values ``[G, K, d]`` (gaps by default)."""
        if self.value_mode == "free":
            source_norm = self.retained_source_norm if use_retained else self.source_bank.norm(dim=-1).max()
            bound = source_norm + scatter(z.norm(dim=-1), graph_id, dim=0, dim_size=num_graphs, reduce="max")
            scale = (bound.unsqueeze(-1) / self.free_values.norm(dim=-1).clamp_min(1e-12)).clamp(max=1.0)
            return scale.unsqueeze(-1) * self.free_values
        if self.value_mode != "target":
            source = self.retained_source_context if use_retained else self.source_context()
            if self.value_mode == "source":
                return source.expand(num_graphs, -1, -1)
        target = pool_target_contexts(self.queries, z, graph_id, num_graphs, self.tau_c, self.obs_eps)
        if self.value_mode == "target":
            return target
        if self.value_mode == "paired_mean":
            return 0.5 * (source + target)
        return source - target

    def mixture_weights(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Eq. 11 mixture ``a [n, K]`` (rows sum to one)."""
        if self.mixture == "local":
            logits = normalize_observations(descriptors, self.obs_eps) @ self.relevance.t() / self.tau_p
            return torch.softmax(logits, dim=-1)
        if self.mixture == "global":
            return torch.softmax(self.mixture_logits, dim=0).expand(descriptors.size(0), -1)
        return descriptors.new_full((descriptors.size(0), self.num_queries), 1.0 / self.num_queries)

    def forward(
        self,
        z: torch.Tensor,
        descriptors: torch.Tensor,
        graph_id: torch.Tensor,
        num_graphs: int,
        *,
        use_retained: bool,
    ) -> torch.Tensor | None:
        """Prompts ``[n, d]`` for the objects observed as ``z``; ``None`` when inactive.

        ``z`` (unprompted observations, one row per prompted object) is also
        each graph's target collection; ``graph_id`` maps rows to graphs.
        """
        if bool(self.source_empty):
            return None
        values = self.values(z, graph_id, num_graphs, use_retained=use_retained)
        weights = self.mixture_weights(descriptors) * torch.tanh(self.gates)
        prompt = z.new_zeros(z.shape)
        for k in range(self.num_queries):
            prompt = prompt + weights[:, k : k + 1] * values[graph_id, k]
        return prompt


class ContextGapPrompt(nn.Module):
    """GapTune prompt generator over the driver observations (Eq. 10-12).

    ``obs_dims`` / ``desc_dims`` come from ``driver.observation_dims(model)``
    and ``driver.descriptor_dims(model)``.  Types are keyed ``N`` and
    ``M1``..``ML``; parameters are drawn from *generator* in that order.
    """

    def __init__(
        self,
        obs_dims: dict,
        desc_dims: dict,
        *,
        num_queries: int = 8,
        tau_c: float = 0.5,
        tau_p: float = 0.5,
        obs_eps: float = 1e-6,
        value_mode: str = "gap",
        prompt_locations: str = "node_message",
        query_mode: str = "shared",
        mixture: str = "local",
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        kwargs = dict(
            num_queries=num_queries,
            tau_c=tau_c,
            tau_p=tau_p,
            obs_eps=obs_eps,
            value_mode=value_mode,
            query_mode=query_mode,
            mixture=mixture,
            generator=generator,
        )
        types = {}
        if prompt_locations in ("node_message", "node"):
            types["N"] = ObservationTypePrompt(obs_dims["N"], desc_dims["N"], **kwargs)
        if prompt_locations in ("node_message", "message"):
            for layer, (obs_dim, desc_dim) in enumerate(zip(obs_dims["M"], desc_dims["M"]), start=1):
                types[f"M{layer}"] = ObservationTypePrompt(obs_dim, desc_dim, **kwargs)
        self.types = nn.ModuleDict(types)

    def set_source_bank(self, banks: dict) -> None:
        """Fix the source observations (``banks[key]`` ``[n_s, d]`` per type)."""
        for key, prompt in self.types.items():
            prompt.set_source_bank(banks[key])

    def refresh_retained(self) -> None:
        """Recompute the retained source state from the current queries."""
        for prompt in self.types.values():
            prompt.refresh_retained()

    @torch.no_grad()
    def project_gates(self) -> None:
        """Nonnegative-gate ablation: project every ``theta`` onto ``[0, inf)``."""
        for prompt in self.types.values():
            prompt.gates.clamp_(min=0.0)

    def forward(self, obs, batch: torch.Tensor, *, use_retained: bool):
        """``(node_prompt | None, message_prompts | None)`` for ``driver.forward``.

        *obs* are the detached unprompted observations of the prediction
        graphs (``GapTuneObservations``); *batch* maps nodes to graphs.
        """
        num_graphs = int(batch.max()) + 1
        node_prompt = None
        if "N" in self.types:
            descriptors = torch.cat([obs.h0, obs.h_final], dim=-1)
            node_prompt = self.types["N"](obs.h0, descriptors, batch, num_graphs, use_retained=use_retained)
        message_prompts = None
        if "M1" in self.types:
            message_prompts = [
                self._message_prompt(self.types[f"M{layer}"], obs.layers[layer - 1], batch, num_graphs, use_retained)
                for layer in range(1, len(obs.layers) + 1)
            ]
        return node_prompt, message_prompts

    def _message_prompt(self, prompt, layer, batch, num_graphs: int, use_retained: bool):
        descriptors = torch.cat(
            [layer.inputs[layer.sender], layer.inputs[layer.receiver], layer.messages], dim=-1
        )
        return prompt(layer.messages, descriptors, batch[layer.receiver], num_graphs, use_retained=use_retained)


__all__ = [
    "GATES",
    "MIXTURES",
    "PROMPT_LOCATIONS",
    "QUERY_MODES",
    "VALUE_MODES",
    "ContextGapPrompt",
    "ObservationTypePrompt",
    "normalize_observations",
    "pool_source_context",
    "pool_target_contexts",
]
