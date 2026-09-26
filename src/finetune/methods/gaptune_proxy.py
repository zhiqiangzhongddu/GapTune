"""Source-free proxy source graphs for GapTune (paper App. B.8, D.2, Table 18).

``B`` proxies with ``n`` nodes each carry trainable continuous features
``X_b`` in the encoder's input space and share an edge MLP ``e_omega``
whose symmetrized pair logits (Eq. 55) parameterize a relaxed adjacency
with a hard straight-through forward (Eq. 56).  ``mode="inverted"``
optimizes them against the checkpoint's retained pretext component with
the frozen encoder in eval mode:

- EdgePred (Eq. 57): fixed positive / negative conditioning pairs drawn
  from a random template, masked out of the propagation adjacency and
  scored by the original edge scorer (dot product, or the retained MLP);
- GraphCL (Eq. 58): two label-free views per proxy (undirected edge
  dropout, elementwise feature masking) through the native graph readout
  and the retained projection, symmetric NT-Xent over the ``2B`` views;

plus ``lambda_x R_X + lambda_a R_A`` (Eq. 59).  ``mode="random"`` keeps the
same initialization and discretization without any update.  The edge
probabilities are then discretized; the caller extracts the unprompted
observations from the returned graphs.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

from src.utils.parsing import to_bool

# Paper D.2 seed streams s + offset (GapTune itself uses s and s + 1).
_INIT_STREAM = 0
_TEMPLATE_STREAM = 2
_EDGE_NOISE_STREAM = 3
_AUGMENTATION_STREAM = 4
_TIE_BREAK_STREAM = 6
_LOG_EVERY = 100
_PRETEXTS = ("edge_pred", "graphcl")
_MODES = ("inverted", "random")


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _project_rows(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Project every row onto the radius ball."""
    return x * (radius / x.norm(dim=-1, keepdim=True)).clamp(max=1.0)


def _edge_mlp(in_dim: int, hidden: int, density: float, generator: torch.Generator) -> nn.Sequential:
    """``2 d0 -> hidden -> hidden -> 1`` with Xavier-uniform weights, zero hidden
    biases and output bias ``log(rho / (1 - rho))`` (paper D.2)."""
    mlp = nn.Sequential(
        nn.Linear(2 * in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    with torch.no_grad():
        for layer in mlp[::2]:
            bound = math.sqrt(6.0 / (layer.in_features + layer.out_features))
            layer.weight.uniform_(-bound, bound, generator=generator)
            layer.bias.zero_()
        mlp[-1].bias.fill_(math.log(density / (1.0 - density)))
    return mlp


def _edgepred_templates(num_graphs: int, num_pairs: int, proxy_cfg, generator: torch.Generator):
    """Fixed conditioning pairs per proxy (indices into the ``u < v`` pairs).

    A uniformly sampled ``final_edges``-edge template; the positives are
    template edges and the negatives template non-edges, both drawn
    without replacement.  Returns ``(positive [B, p], negative [B, q])``.
    """
    template_edges = int(proxy_cfg.final_edges)
    num_pos, num_neg = int(proxy_cfg.edgepred_pos_pairs), int(proxy_cfg.edgepred_neg_pairs)
    positive, negative = [], []
    for _ in range(num_graphs):
        order = torch.randperm(num_pairs, generator=generator)
        edges, non_edges = order[:template_edges], order[template_edges:]
        positive.append(edges[torch.randperm(template_edges, generator=generator)[:num_pos]])
        negative.append(non_edges[torch.randperm(non_edges.numel(), generator=generator)[:num_neg]])
    return torch.stack(positive), torch.stack(negative)


def _pair_logits(edge_mlp, features, row, col, conditioning, conditioning_logit: float) -> torch.Tensor:
    """Symmetrized logits ``[B, P]`` of the pairs ``(row, col)`` (Eq. 55);
    conditioning pairs are fixed at ``+/- conditioning_logit``."""
    x_u, x_v = features[:, row], features[:, col]
    forward = edge_mlp(torch.cat([x_u, x_v], dim=-1))
    backward = edge_mlp(torch.cat([x_v, x_u], dim=-1))
    logits = 0.5 * (forward + backward).squeeze(-1)
    if conditioning is not None:
        positive, negative = conditioning
        logits = logits.scatter(1, positive, conditioning_logit).scatter(1, negative, -conditioning_logit)
    return logits


def _encode(driver, model, features, pair_weight, row, col):
    """Frozen encoder on ``G`` complete graphs whose ``u < v`` pairs carry
    ``pair_weight [G, P]`` in both directions (zero weights stay in the
    backward pass)."""
    num_graphs, num_nodes = features.shape[:2]
    offsets = (torch.arange(num_graphs, device=features.device) * num_nodes).view(-1, 1)
    u = (row.view(1, -1) + offsets).reshape(-1)
    v = (col.view(1, -1) + offsets).reshape(-1)
    data = Data(
        x=features.reshape(num_graphs * num_nodes, -1),
        edge_index=torch.stack([torch.cat([u, v]), torch.cat([v, u])]),
        batch=torch.arange(num_graphs, device=features.device).repeat_interleave(num_nodes),
    )
    return driver.forward(model, data, edge_weight=pair_weight.reshape(-1).repeat(2))


def _retained_weights(pretrain_extra, keys, component: str, device) -> list[torch.Tensor]:
    state = pretrain_extra.get("pretrain_task_state") or {}
    missing = [key for key in keys if key not in state]
    if missing:
        raise ValueError(
            f"[GapTune] Source-free inversion needs the checkpoint's retained {component} "
            f"(paper App. B.8), but extra.pretrain_task_state lacks {missing}. Checkpoints saved "
            "without the pretrain task state (e.g. AnyGraphAnyExpert's) cannot be inverted: "
            "re-pretrain the checkpoint with this repo (scripts/run_pretrain.py), or run GapTune+ "
            "with finetune.gaptune.plus=True."
        )
    return [state[key].to(device) for key in keys]


def _two_layer(weights):
    w1, b1, w2, b2 = weights
    return lambda z: F.linear(F.relu(F.linear(z, w1, b1)), w2, b2)


def _edgepred_objective(use_mlp_scorer, model, driver, pretrain_extra, device, row, col, conditioning):
    """Balanced masked-pair loss (Eq. 57) with the original edge scorer."""
    positive, negative = conditioning
    keep = torch.ones(positive.size(0), row.numel(), device=device)
    keep = keep.scatter(1, positive, 0.0).scatter(1, negative, 0.0)
    if use_mlp_scorer:
        keys = ("scorer.0.weight", "scorer.0.bias", "scorer.2.weight", "scorer.2.bias")
        mlp = _two_layer(_retained_weights(pretrain_extra, keys, "MLP edge scorer", device))
        score = lambda h_u, h_v: mlp(torch.cat([h_u, h_v], dim=-1)).squeeze(-1)
    else:
        score = lambda h_u, h_v: (h_u * h_v).sum(dim=-1)  # original dot-product scorer

    def objective(features, adjacency):
        node_repr = _encode(driver, model, features, adjacency * keep, row, col).node_repr
        node_repr = node_repr.view(features.size(0), features.size(1), -1)
        batch_index = torch.arange(features.size(0), device=device).view(-1, 1)

        def pair_scores(pairs):
            return score(node_repr[batch_index, row[pairs]], node_repr[batch_index, col[pairs]])

        per_graph = F.logsigmoid(pair_scores(positive)).mean(1) + F.logsigmoid(-pair_scores(negative)).mean(1)
        return -0.5 * per_graph.mean()

    return objective


def _graphcl_objective(proxy_cfg, model, driver, pretrain_extra, device, row, col, generator):
    """Two-view NT-Xent over all ``2B`` views (Eq. 58) with the retained projection."""
    keys = ("projector.fc1.weight", "projector.fc1.bias", "projector.fc2.weight", "projector.fc2.bias")
    projector = _two_layer(_retained_weights(pretrain_extra, keys, "GraphCL projection", device))
    edge_drop, feature_mask = float(proxy_cfg.graphcl_edge_drop), float(proxy_cfg.graphcl_feature_mask)
    tau = float(proxy_cfg.graphcl_tau)

    def objective(features, adjacency):
        num_graphs = features.size(0)
        edge_keep = (torch.rand((2,) + tuple(adjacency.shape), generator=generator) >= edge_drop).to(device)
        feature_keep = (torch.rand((2,) + tuple(features.shape), generator=generator) >= feature_mask).to(device)
        views = (features.unsqueeze(0) * feature_keep).flatten(0, 1)  # view r of proxy b at r * B + b
        graph_repr = _encode(driver, model, views, (adjacency.unsqueeze(0) * edge_keep).flatten(0, 1), row, col).graph_repr
        z = F.normalize(projector(graph_repr), dim=-1)
        similarity = (z @ z.t() / tau).fill_diagonal_(float("-inf"))
        partner = torch.arange(2 * num_graphs, device=device).roll(num_graphs)
        return F.cross_entropy(similarity, partner)

    return objective


def _top_pairs(scores: torch.Tensor, priority: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` highest scores per row; ties go to the higher priority."""
    order = priority.argsort(dim=1, descending=True)
    ranked = order.gather(1, scores.gather(1, order).argsort(dim=1, descending=True, stable=True))
    return ranked[:, :k]


def build_proxy_source_graphs(*, cfg, model, driver, pretrain_cfg, pretrain_extra, device):
    """Return ``(graphs, metadata)``: the discrete proxy graphs and a JSON-able dict.

    ``graphs`` is a list of ``torch_geometric.data.Data`` with ``x`` ``[n_b, d0]``
    and an undirected ``edge_index`` (both directions, no self-loops); the
    GapTune task encodes them with the frozen unprompted driver exactly like
    pretraining graphs.
    """
    proxy_cfg = cfg.finetune.gaptune.proxy
    # The checkpoint's own pretext settings win (an explicit
    # finetune.pretrained_checkpoint need not match cfg.pretrain).
    trained = pretrain_cfg.get("pretrain") or {}
    pretext = str(trained.get("method") or cfg.pretrain.method).lower()
    mode = str(proxy_cfg.mode)
    if pretext not in _PRETEXTS:
        raise ValueError(
            f"[GapTune] Source-free proxies (finetune.gaptune.plus=False) need an EdgePred or GraphCL "
            f"checkpoint (paper App. B.8); got pretrain.method={pretext!r}. Use finetune.gaptune.plus=True."
        )
    if mode not in _MODES:
        raise ValueError(f"[GapTune] finetune.gaptune.proxy.mode={mode!r} is invalid; expected one of {list(_MODES)}.")
    num_graphs, num_nodes, feature_dim = int(proxy_cfg.num_graphs), int(proxy_cfg.num_nodes), int(cfg.model.in_dim)
    final_edges, radius, seed = int(proxy_cfg.final_edges), float(proxy_cfg.feature_radius), int(cfg.seed)
    row, col = torch.triu_indices(num_nodes, num_nodes, offset=1)  # the u < v pairs
    num_pairs = row.numel()
    num_pos, num_neg = int(proxy_cfg.edgepred_pos_pairs), int(proxy_cfg.edgepred_neg_pairs)
    if pretext == "edge_pred" and not (0 < num_pos <= final_edges and 0 < num_neg and final_edges + num_neg <= num_pairs):
        raise ValueError(
            f"[GapTune] EdgePred proxies need 0 < edgepred_pos_pairs <= final_edges and "
            f"final_edges + edgepred_neg_pairs <= {num_pairs} pairs; got {num_pos}, {final_edges}, {num_neg}."
        )
    if pretext == "graphcl" and not (final_edges <= num_pairs and (mode == "random" or num_graphs >= 2)):
        raise ValueError(
            f"[GapTune] GraphCL proxies need final_edges <= {num_pairs} pairs and, for inversion, "
            f"num_graphs >= 2; got {final_edges} and {num_graphs}."
        )

    init = _generator(seed + _INIT_STREAM)
    edge_mlp = _edge_mlp(feature_dim, int(proxy_cfg.edge_hidden), float(proxy_cfg.density), init).to(device)
    features = nn.Parameter(_project_rows(torch.randn(num_graphs, num_nodes, feature_dim, generator=init), radius).to(device))
    conditioning = None
    if pretext == "edge_pred":
        conditioning = tuple(
            pairs.to(device) for pairs in _edgepred_templates(num_graphs, num_pairs, proxy_cfg, _generator(seed + _TEMPLATE_STREAM))
        )
    priority = torch.rand(num_graphs, num_pairs, generator=_generator(seed + _TIE_BREAK_STREAM))
    row, col = row.to(device), col.to(device)
    logit_fix = float(proxy_cfg.edgepred_logit)

    updates = int(proxy_cfg.updates) if mode == "inverted" else 0
    losses = []
    if updates > 0:
        if pretext == "edge_pred":
            use_mlp_scorer = (trained.get("edge_pred") or {}).get("use_mlp_scorer", cfg.pretrain.edge_pred.use_mlp_scorer)
            objective = _edgepred_objective(
                to_bool(use_mlp_scorer), model, driver, pretrain_extra, device, row, col, conditioning
            )
        else:
            objective = _graphcl_objective(
                proxy_cfg, model, driver, pretrain_extra, device, row, col, _generator(seed + _AUGMENTATION_STREAM)
            )
        params = [features, *edge_mlp.parameters()]
        optimizer = torch.optim.Adam(params, lr=float(proxy_cfg.lr), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        noise = _generator(seed + _EDGE_NOISE_STREAM)
        tau_start, tau_end = float(proxy_cfg.tau_start), float(proxy_cfg.tau_end)
        density = float(proxy_cfg.density)
        for step in range(updates):
            tau = tau_start * (tau_end / tau_start) ** (step / max(updates - 1, 1))
            logits = _pair_logits(edge_mlp, features, row, col, conditioning, logit_fix)
            uniform = torch.rand(tuple(logits.shape), generator=noise).to(device)
            relaxed = torch.sigmoid((logits + uniform.log() - torch.log1p(-uniform)) / tau)  # Eq. 56
            adjacency = ((relaxed >= 0.5).float() - relaxed).detach() + relaxed  # hard straight-through
            pretext_loss = objective(features, adjacency)
            r_x = features.pow(2).mean()  # Eq. 59
            r_a = (torch.sigmoid(logits).mean(dim=1) - density).pow(2).mean()
            loss = pretext_loss + float(proxy_cfg.lambda_x) * r_x + float(proxy_cfg.lambda_a) * r_a
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(proxy_cfg.grad_clip))
            optimizer.step()
            with torch.no_grad():
                features.copy_(_project_rows(features, radius))
            if step % _LOG_EVERY == 0 or step == updates - 1:
                record = {
                    "update": step + 1,
                    "tau": tau,
                    "loss": float(loss),
                    "pretext": float(pretext_loss),
                    "r_x": float(r_x),
                    "r_a": float(r_a),
                }
                losses.append(record)
                print(
                    f"[Finetune][GapTune] Proxy inversion ({pretext}) update {step + 1}/{updates}: "
                    + " ".join(f"{key}={value:.4f}" for key, value in record.items() if key != "update")
                )

    # Final discretization of the last iterate's edge probabilities.
    with torch.no_grad():
        probability = torch.sigmoid(_pair_logits(edge_mlp, features, row, col, conditioning, logit_fix)).cpu()
    features = features.detach().cpu()
    row, col = row.cpu(), col.cpu()
    if pretext == "edge_pred":
        positive, negative = (pairs.cpu() for pairs in conditioning)
        free = probability.scatter(1, positive, float("-inf")).scatter(1, negative, float("-inf"))
        selected = torch.cat([positive, _top_pairs(free, priority, final_edges - num_pos)], dim=1)
    else:
        selected = _top_pairs(probability, priority, final_edges)
    graphs = []
    for index in range(num_graphs):
        u, v = row[selected[index]], col[selected[index]]
        graphs.append(Data(x=features[index], edge_index=torch.stack([torch.cat([u, v]), torch.cat([v, u])])))
    metadata = {
        "mode": mode,
        "pretext": pretext,
        "num_graphs": num_graphs,
        "num_nodes": num_nodes,
        "feature_dim": feature_dim,
        "updates": updates,
        "losses": losses,
        "edges_per_graph": [int(graph.edge_index.size(1)) // 2 for graph in graphs],
        "mean_edge_probability": float(probability.mean()),
    }
    return graphs, metadata
