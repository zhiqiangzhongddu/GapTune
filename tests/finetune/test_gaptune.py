"""GapTune prompt / task tests (paper Sec. 3, App. B-C).

- cost-study census: 14,496 prompt parameters and 3,872 retained scalars
  (GCN 100->128x3, K=8), plus the free-value node/message-only counts;
- zero gates at init and identical source/target collections (Prop. B.1)
  give prompted == unprompted;
- ``||p_o|| <= max_k ||d_k||`` (Prop. 3.1); node-prompt permutation
  equivariance and edge-readout symmetry (Prop. 3.2);
- exact retention (Prop. B.3), including a reload without the source bank;
- target pooling never mixes the graphs of a minibatch; multi-graph
  prompt mixing matches the per-query reference loop (values and grads);
- ablation switches (value modes, free-value clip, locations, frozen /
  untied queries, uniform / global mixtures, nonnegative gates); gradients
  reach Q, W, theta and the head, never the encoder;
- source sampling caps/determinism, the runner hook, minibatch order
  (stream s+5) identical across arms, config / run names, TSV column and
  result-table rows.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from yacs.config import CfgNode as CN

import src.finetune.methods.gaptune as gaptune_module
from src.config import set_cfg
from src.finetune.methods.gaptune import FinetuneGapTune, sample_source_observations
from src.finetune.prompts.gaptune import ObservationTypePrompt, pool_source_context, pool_target_contexts
from src.finetune.readouts import readout_input_dim, task_native_readout
from src.finetune.target_stats import preserve_loader_rng
from src.finetune.utils import _finetune_custom_parser, parse_finetune_tasks
from src.model.encoder import build_encoder_from_cfg
from src.results.finetune_tables import PROMPT_METHODS, _plus_from_result_row, _plus_from_task
from src.utils.naming import build_finetune_run_name_from_cfg

IN_DIM = 6
MODELS = ("gcn", "gin", "gat", "transformer", "fagcn", "h2gcn", "nodeformer")
ATOL = 1e-5


def _cfg(name: str = "gcn", level: str = "node", **gaptune):
    cfg = CN()
    set_cfg(cfg)
    cfg.seed = 3
    cfg.model.name = name
    cfg.model.num_layers = 2
    cfg.model.in_dim = IN_DIM
    cfg.model.hidden_dim = 8
    cfg.model.out_dim = 5
    cfg.model.gat.heads = 2
    cfg.model.nodeformer.heads = 2
    cfg.model.nodeformer.num_random_features = 6
    ds = cfg.finetune.dataset
    ds.name = "toy"
    ds.task_level = level
    ds.induced = True
    ds.task_level_raw = level
    ds.task_level_effective = "graph"
    ds.task_type = "classification"
    ds.num_classes = 2 if level == "edge" else 3
    ds.label_dim = 1
    for key, value in gaptune.items():
        setattr(cfg.finetune.gaptune, key, value)
    return cfg


def _encoder(cfg, in_dim: int = IN_DIM):
    torch.manual_seed(0)
    encoder = build_encoder_from_cfg(cfg, in_dim).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def _graphs(sizes=((5, 8), (4, 6), (6, 9)), seed: int = 0, in_dim: int = IN_DIM) -> list[Data]:
    g = torch.Generator().manual_seed(seed)
    graphs = []
    for index, (n, e) in enumerate(sizes):
        graph = Data(
            x=torch.randn(n, in_dim, generator=g),
            edge_index=torch.randint(0, n, (2, e), generator=g),
            y=torch.tensor([index % 2]),
        )
        graph.target_node_index = torch.tensor([1])
        graph.edge_label_index = torch.tensor([[0], [2]])
        graphs.append(graph)
    return graphs


def _task(cfg, *, bank: bool = True, in_dim: int = IN_DIM):
    encoder = _encoder(cfg, in_dim)
    task = FinetuneGapTune(cfg)
    task.validate_encoder(encoder)
    if bank:
        banks, _ = sample_source_observations(
            task.driver,
            encoder,
            _graphs(sizes=((7, 12), (6, 10), (8, 14)), seed=10, in_dim=in_dim),
            projections=task.projections,
            device=torch.device("cpu"),
            max_nodes=512,
            max_messages=2048,
            generator=torch.Generator().manual_seed(1),
        )
        task.prompt.set_source_bank(banks)
        task.prompt.refresh_retained()
    return task, encoder


def _set_gates(task, seed: int = 0, scale: float = 2.0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for prompt in task.prompt.types.values():
            prompt.gates.copy_(scale * torch.randn(prompt.gates.shape, generator=g))


def _unprompted(task, encoder, data):
    with torch.no_grad():
        return task.driver.forward(encoder, data, collect=True, projections=task.projections)


def _prompts(task, encoder, data, *, use_retained: bool = False):
    obs = _unprompted(task, encoder, data).obs
    return task.prompt(obs, data.batch, use_retained=use_retained)


def _trainable(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Census (paper App. C.10)
# --------------------------------------------------------------------------


def _cost_study_task(**gaptune):
    cfg = _cfg("gcn", **gaptune)
    cfg.model.num_layers = 3
    cfg.model.in_dim = 100
    cfg.model.hidden_dim = 128
    cfg.model.out_dim = 128
    cfg.finetune.dataset.num_classes = 8
    task, _ = _task(cfg, bank=False, in_dim=100)
    return task


def test_cost_study_census():
    task = _cost_study_task()
    assert list(task.prompt.types) == ["N", "M1", "M2", "M3"]
    assert _trainable(task.prompt) == 14_496
    assert sum(p.retained_source_context.numel() for p in task.prompt.types.values()) == 3_872
    assert _trainable(task.head) == 1_032
    assert sum(p.numel() for p in task.parameters_to_optimize()) == 15_528
    # Free values replace the query parameters (same counts, App. C.2).
    assert sum(p.numel() for p in _cost_study_task(value_mode="free").parameters_to_optimize()) == 15_528
    node_free = _cost_study_task(value_mode="free", prompt_locations="node")
    message_free = _cost_study_task(value_mode="free", prompt_locations="message")
    assert sum(p.numel() for p in node_free.parameters_to_optimize()) == 3_664
    assert sum(p.numel() for p in message_free.parameters_to_optimize()) == 12_896


# --------------------------------------------------------------------------
# Zero prompts (Prop. B.1, zero gates)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_zero_gates_at_init_reproduce_unprompted(name):
    task, encoder = _task(_cfg(name))
    data = Batch.from_data_list(_graphs())
    task.train()
    node_prompt, message_prompts = _prompts(task, encoder, data)
    assert torch.count_nonzero(node_prompt) == 0
    assert all(torch.count_nonzero(p) == 0 for p in message_prompts)
    with torch.no_grad():
        prompted = task.encode(encoder, data)
    torch.testing.assert_close(prompted, _unprompted(task, encoder, data).node_repr, atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", ("gcn", "gat", "nodeformer"))
def test_identical_source_and_target_collections_give_zero_prompts(name):
    task, encoder = _task(_cfg(name), bank=False)
    data = Batch.from_data_list(_graphs(sizes=((6, 10),)))
    obs = _unprompted(task, encoder, data).obs
    banks = {"N": obs.h0, **{f"M{l}": layer.messages for l, layer in enumerate(obs.layers, start=1)}}
    task.prompt.set_source_bank(banks)
    task.prompt.refresh_retained()
    _set_gates(task)
    for use_retained in (False, True):
        node_prompt, message_prompts = task.prompt(obs, data.batch, use_retained=use_retained)
        assert node_prompt.abs().max() < ATOL
        assert all(p.abs().max() < ATOL for p in message_prompts)
    task.train()
    with torch.no_grad():
        prompted = task.encode(encoder, data)
    torch.testing.assert_close(prompted, obs.h_final, atol=ATOL, rtol=0)


# --------------------------------------------------------------------------
# Bounded prompts (Prop. 3.1), symmetries (Prop. 3.2)
# --------------------------------------------------------------------------


def test_prompt_norm_bounded_by_largest_gap():
    task, encoder = _task(_cfg("gcn"))
    _set_gates(task, scale=5.0)
    data = Batch.from_data_list(_graphs())
    obs = _unprompted(task, encoder, data).obs
    num_graphs = data.num_graphs
    cases = [(task.prompt.types["N"], obs.h0, torch.cat([obs.h0, obs.h_final], -1), data.batch)]
    for l, layer in enumerate(obs.layers, start=1):
        descriptors = torch.cat([layer.inputs[layer.sender], layer.inputs[layer.receiver], layer.messages], -1)
        cases.append((task.prompt.types[f"M{l}"], layer.messages, descriptors, data.batch[layer.receiver]))
    for prompt, z, descriptors, graph_id in cases:
        gaps = prompt.values(z, graph_id, num_graphs, use_retained=False)
        p = prompt(z, descriptors, graph_id, num_graphs, use_retained=False)
        bound = gaps.norm(dim=-1).max(dim=1).values[graph_id]
        assert torch.count_nonzero(p) > 0
        assert bool((p.norm(dim=-1) <= bound + 1e-6).all())


def test_node_prompts_and_outputs_are_permutation_equivariant():
    task, encoder = _task(_cfg("gcn"))
    _set_gates(task)
    graph = _graphs(sizes=((7, 12),))[0]
    perm = torch.randperm(7, generator=torch.Generator().manual_seed(4))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(7)
    permuted = Data(x=graph.x[perm], edge_index=inverse[graph.edge_index])
    task.train()
    data, data_perm = Batch.from_data_list([graph]), Batch.from_data_list([permuted])
    node_prompt, _ = _prompts(task, encoder, data)
    node_prompt_perm, _ = _prompts(task, encoder, data_perm)
    torch.testing.assert_close(node_prompt_perm, node_prompt[perm], atol=ATOL, rtol=0)
    with torch.no_grad():
        torch.testing.assert_close(
            task.encode(encoder, data_perm), task.encode(encoder, data)[perm], atol=ATOL, rtol=0
        )


def test_task_native_readouts():
    graphs = _graphs()
    data = Batch.from_data_list(graphs)
    h = torch.randn(data.num_nodes, 5)
    reps, labels = task_native_readout(h, data, "node")
    torch.testing.assert_close(reps, h[data.target_node_index])
    torch.testing.assert_close(labels, data.y)
    reps, _ = task_native_readout(h, data, "edge")
    swapped = data.clone()
    swapped.edge_label_index = data.edge_label_index.flip(0)
    torch.testing.assert_close(task_native_readout(h, swapped, "edge")[0], reps)
    u, v = data.edge_label_index
    torch.testing.assert_close(reps, torch.cat([h[u] + h[v], h[u] * h[v]], dim=-1))
    reps, _ = task_native_readout(h, data, "graph")
    torch.testing.assert_close(reps[1], h[data.batch == 1].mean(dim=0))
    assert readout_input_dim(5, "edge") == 10 and readout_input_dim(5, "node") == 5


# --------------------------------------------------------------------------
# Retention (Prop. B.3) and per-graph target pooling
# --------------------------------------------------------------------------


def test_retained_contexts_reproduce_source_bank_predictions():
    cfg = _cfg("gcn")
    task, encoder = _task(cfg)
    _set_gates(task)
    with torch.no_grad():
        for prompt in task.prompt.types.values():
            prompt.queries.add_(0.3)  # queries move during training ...
    task.prompt.refresh_retained()  # ... and the epoch end refreshes C*_s
    data = Batch.from_data_list(_graphs())
    with torch.no_grad():
        task.train()
        _, _, bank_logits, _ = task._forward(encoder, data, "cpu", return_outputs=True)
        task.eval()
        _, _, retained_logits, _ = task._forward(encoder, data, "cpu", return_outputs=True)
    torch.testing.assert_close(retained_logits, bank_logits, atol=1e-6, rtol=0)

    # Deployment: the checkpoint state holds C*_s, not the source bank.
    state = task.state_dict()
    assert not any(key.endswith("source_bank") for key in state)
    restored, _ = _task(cfg, bank=False)
    restored.load_state_dict(state)
    restored.eval()
    with torch.no_grad():
        _, _, deployed_logits, _ = restored._forward(encoder, data, "cpu", return_outputs=True)
    torch.testing.assert_close(deployed_logits, bank_logits, atol=1e-6, rtol=0)


def test_target_pooling_never_mixes_graphs():
    task, encoder = _task(_cfg("gcn"))
    _set_gates(task)
    task.train()
    graphs = _graphs()
    data = Batch.from_data_list(graphs)
    changed = [g.clone() for g in graphs]
    changed[2].x = changed[2].x * 3.0 + 1.0
    data_changed = Batch.from_data_list(changed)
    node_prompt, message_prompts = _prompts(task, encoder, data)
    node_prompt_changed, message_prompts_changed = _prompts(task, encoder, data_changed)
    keep_nodes = data.batch != 2
    torch.testing.assert_close(node_prompt_changed[keep_nodes], node_prompt[keep_nodes], atol=1e-6, rtol=0)
    assert not torch.allclose(node_prompt_changed[~keep_nodes], node_prompt[~keep_nodes])
    obs = _unprompted(task, encoder, data).obs
    for layer, p, p_changed in zip(obs.layers, message_prompts, message_prompts_changed):
        keep = data.batch[layer.receiver] != 2
        torch.testing.assert_close(p_changed[keep], p[keep], atol=1e-6, rtol=0)


@pytest.mark.parametrize("value_mode,query_mode", [("gap", "shared"), ("gap", "frozen"), ("free", "shared")])
def test_multi_graph_mixing_matches_reference_loop(value_mode, query_mode):
    g = torch.Generator().manual_seed(0)
    num, dim, desc_dim, num_graphs = 200, 16, 24, 5
    prompt = ObservationTypePrompt(
        dim,
        desc_dim,
        num_queries=8,
        tau_c=0.5,
        tau_p=0.5,
        obs_eps=1e-6,
        value_mode=value_mode,
        query_mode=query_mode,
        mixture="local",
        generator=g,
    )
    prompt.set_source_bank(torch.randn(50, dim, generator=g))
    with torch.no_grad():
        prompt.gates.copy_(2.0 * torch.randn(prompt.gates.shape, generator=g))
    z = torch.randn(num, dim, generator=g)
    descriptors = torch.randn(num, desc_dim, generator=g)
    graph_id = torch.randint(num_graphs, (num,), generator=g)  # unsorted rows
    upstream = torch.randn(num, dim, generator=g)
    params = [p for p in prompt.parameters() if p.requires_grad]

    out = prompt(z, descriptors, graph_id, num_graphs, use_retained=False)
    grads = torch.autograd.grad((out * upstream).sum(), params)

    values = prompt.values(z, graph_id, num_graphs, use_retained=False)
    weights = prompt.mixture_weights(descriptors) * torch.tanh(prompt.gates)
    expected = z.new_zeros(z.shape)
    for k in range(prompt.num_queries):
        expected = expected + weights[:, k : k + 1] * values[graph_id, k]
    expected_grads = torch.autograd.grad((expected * upstream).sum(), params)

    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)
    for grad, expected_grad in zip(grads, expected_grads):
        torch.testing.assert_close(grad, expected_grad, atol=1e-6, rtol=0)


# --------------------------------------------------------------------------
# Ablation switches (App. C)
# --------------------------------------------------------------------------


def test_value_modes():
    task, encoder = _task(_cfg("gcn"))
    data = Batch.from_data_list(_graphs())
    obs = _unprompted(task, encoder, data).obs
    prompt = task.prompt.types["N"]
    source = pool_source_context(prompt.queries, prompt.source_bank, prompt.tau_c, prompt.obs_eps)
    target = pool_target_contexts(prompt.queries, obs.h0, data.batch, 3, prompt.tau_c, prompt.obs_eps)
    expected = {
        "gap": source - target,
        "target": target,
        "source": source.expand(3, -1, -1),
        "paired_mean": 0.5 * (source + target),
    }
    for mode, value in expected.items():
        prompt.value_mode = mode
        torch.testing.assert_close(prompt.values(obs.h0, data.batch, 3, use_retained=False), value)
    # Target contexts are per-graph convex combinations of that graph's rows.
    graph0 = obs.h0[data.batch == 0]
    torch.testing.assert_close(
        target[0], pool_source_context(prompt.queries, graph0, prompt.tau_c, prompt.obs_eps)
    )


def test_free_values_are_clipped_to_the_graph_envelope():
    task, encoder = _task(_cfg("gcn", value_mode="free"))
    data = Batch.from_data_list(_graphs())
    obs = _unprompted(task, encoder, data).obs
    prompt = task.prompt.types["N"]
    assert not hasattr(prompt, "queries")
    with torch.no_grad():
        prompt.free_values[0].mul_(1e4 / prompt.free_values[0].norm())
        prompt.free_values[1].mul_(1e-3 / prompt.free_values[1].norm())
    envelope = prompt.source_bank.norm(dim=-1).max() + torch.stack(
        [obs.h0[data.batch == g].norm(dim=-1).max() for g in range(3)]
    )
    for use_retained in (False, True):
        values = prompt.values(obs.h0, data.batch, 3, use_retained=use_retained)
        torch.testing.assert_close(values[:, 0].norm(dim=-1), envelope)
        torch.testing.assert_close(values[:, 1], prompt.free_values[1].expand(3, -1))


def test_prompt_locations():
    data = Batch.from_data_list(_graphs())
    node_task, _ = _task(_cfg("gcn", prompt_locations="node"))
    message_task, _ = _task(_cfg("gcn", prompt_locations="message"))
    assert list(node_task.prompt.types) == ["N"]
    assert list(message_task.prompt.types) == ["M1", "M2"]

    none_task, encoder = _task(_cfg("gcn", prompt_locations="none"))
    assert len(none_task.prompt.types) == 0
    assert list(none_task.parameters_to_optimize()) == list(none_task.head.parameters())
    none_task.build_optimizers(encoder)
    none_task.train()
    with torch.no_grad():
        torch.testing.assert_close(
            none_task.encode(encoder, data), _unprompted(none_task, encoder, data).node_repr
        )


def test_mixture_switches():
    uniform, encoder = _task(_cfg("gcn", mixture="uniform"))
    descriptors = torch.randn(4, 17)
    torch.testing.assert_close(uniform.prompt.types["N"].mixture_weights(descriptors), torch.full((4, 8), 1 / 8))
    assert not hasattr(uniform.prompt.types["N"], "relevance")

    global_task, _ = _task(_cfg("gcn", mixture="global"))
    prompt = global_task.prompt.types["N"]
    with torch.no_grad():
        prompt.mixture_logits.copy_(torch.arange(8.0))
    weights = prompt.mixture_weights(descriptors)
    torch.testing.assert_close(weights, torch.softmax(torch.arange(8.0), 0).expand(4, -1))
    assert _trainable(prompt) == 8 * (prompt.source_bank.size(1) + 1 + 1)


def test_frozen_and_untied_queries():
    data = Batch.from_data_list(_graphs())
    frozen, encoder = _task(_cfg("gcn", query_mode="frozen"))
    _set_gates(frozen)
    frozen.train()
    loss, _ = frozen._forward(encoder, data, "cpu")
    loss.backward()
    for prompt in frozen.prompt.types.values():
        assert not prompt.queries.requires_grad and prompt.queries.grad is None
        assert prompt.relevance.grad is not None
    assert all(p.requires_grad for p in frozen.parameters_to_optimize())

    untied, _ = _task(_cfg("gcn", query_mode="untied"))
    prompt = untied.prompt.types["N"]
    assert not torch.equal(prompt.queries, prompt.source_queries)
    before = prompt.source_context()
    with torch.no_grad():
        prompt.queries.add_(1.0)
    torch.testing.assert_close(prompt.source_context(), before)
    with torch.no_grad():
        prompt.source_queries.add_(1.0)
    assert not torch.allclose(prompt.source_context(), before)
    d, e = prompt.queries.size(1), prompt.relevance.size(1)
    assert _trainable(prompt) == 8 * (2 * d + e + 1)


def test_nonnegative_gates_are_projected_after_every_step():
    for gate, expect_negative in (("nonnegative", False), ("signed", True)):
        task, encoder = _task(_cfg("gcn", gate=gate))
        optimizer = task.build_optimizers(encoder)["primary"]
        for prompt in task.prompt.types.values():
            prompt.gates.grad = torch.ones_like(prompt.gates)  # pushes theta below zero
        optimizer.step()
        gates = torch.cat([p.gates for p in task.prompt.types.values()])
        assert bool((gates < 0).any()) == expect_negative
        assert bool((gates >= 0).all()) != expect_negative


def test_self_loop_messages_are_prompted():
    # Paper convention: every directed message, self-loops included.
    task, encoder = _task(_cfg("gcn"))
    _set_gates(task)
    data = Batch.from_data_list(_graphs())
    obs = _unprompted(task, encoder, data).obs
    _, message_prompts = task.prompt(obs, data.batch, use_retained=False)
    for layer, prompt in zip(obs.layers, message_prompts):
        loops = layer.sender == layer.receiver
        assert bool(loops.any())
        assert bool((prompt[loops].abs().sum(dim=-1) > 0).all())


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_gradients_reach_prompt_and_head_only(name):
    task, encoder = _task(_cfg(name))
    _set_gates(task)
    task.train()
    loss, _ = task._forward(encoder, Batch.from_data_list(_graphs()), "cpu")
    loss.backward()
    for prompt in task.prompt.types.values():
        for param in (prompt.queries, prompt.relevance, prompt.gates):
            assert param.grad is not None and param.grad.abs().sum() > 0
    assert task.head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in encoder.parameters())


# --------------------------------------------------------------------------
# Source sampling and the runner hook
# --------------------------------------------------------------------------


def test_source_sampling_caps_and_determinism():
    cfg = _cfg("gcn")
    task, encoder = _task(cfg, bank=False)
    graphs = _graphs(sizes=[(5, 8)] * 200, seed=2)

    def sample(seed, max_nodes=4, max_messages=5):
        return sample_source_observations(
            task.driver, encoder, graphs, projections=None, device=torch.device("cpu"),
            max_nodes=max_nodes, max_messages=max_messages,
            generator=torch.Generator().manual_seed(seed),
        )

    banks, meta = sample(1)
    assert meta["bank_sizes"] == {"N": 4, "M1": 5, "M2": 5}
    assert meta["graphs_encoded"] == 64 < meta["graphs_total"] == 200
    again, _ = sample(1)
    other, _ = sample(2)
    assert all(torch.equal(banks[key], again[key]) for key in banks)
    assert not torch.equal(banks["N"], other["N"])
    _, full = sample(1, max_nodes=10_000, max_messages=10_000)
    assert full["graphs_encoded"] == 200


@pytest.mark.parametrize("plus", (True, False))
def test_prepare_with_encoder_fills_the_source_bank(monkeypatch, plus):
    cfg = _cfg("gcn", plus=plus)
    task, encoder = _task(cfg, bank=False)
    pretrain_cfg = {"seed": 42, "pretrain": {"dataset": {"name": "src", "task_level": "graph"}}}
    pretrain_extra = {"pretrain_task_state": {}}
    seen = {}

    def fake_load(cfg_, ds_cfg, *, seed):
        seen.update(name=ds_cfg["name"], seed=seed)
        return _graphs(seed=5)

    def fake_proxy(*, cfg, model, driver, pretrain_cfg, pretrain_extra, device):
        seen.update(proxy=driver is task.driver, payload_seed=pretrain_cfg["seed"], extra=pretrain_extra)
        return _graphs(seed=6), {"losses": [1.0]}

    def no_reload(*args, **kwargs):
        raise AssertionError("the runner already loaded the checkpoint")

    monkeypatch.setattr(gaptune_module, "load_pretraining_graphs", fake_load)
    monkeypatch.setattr(gaptune_module, "build_proxy_source_graphs", fake_proxy)
    monkeypatch.setattr(torch, "load", no_reload)
    task.prepare_with_encoder(
        model=encoder, device=torch.device("cpu"), pretrain_cfg=pretrain_cfg, pretrain_extra=pretrain_extra
    )
    meta = task.initialization_metadata
    if plus:
        assert seen == {"name": "src", "seed": 42}
        assert meta["source"] == "pretraining" and meta["source_dataset"] == "src"
    else:
        assert seen == {"proxy": True, "payload_seed": 42, "extra": pretrain_extra}
        assert meta["source"] == "proxy" and meta["proxy"] == {"losses": [1.0]}
    assert meta["observation_seed"] == cfg.seed + 1
    for prompt in task.prompt.types.values():
        assert not bool(prompt.source_empty)
        assert prompt.retained_source_context.abs().sum() > 0


def test_head_only_control_skips_source_preparation():
    task, encoder = _task(_cfg("gcn", prompt_locations="none"), bank=False)
    task.prepare_with_encoder(model=encoder, device=torch.device("cpu"), pretrain_cfg={}, pretrain_extra={})
    assert task.initialization_metadata is None


def test_minibatch_order_is_identical_across_arms(monkeypatch):
    """Paper D.2 stream s+5 is the runner's global seed s: head/prompt building
    draws the same from the global RNG in every arm, and source preparation
    (GapTune+ / proxy mode / budget) runs under ``preserve_loader_rng``."""
    pretrain_cfg = {"seed": 42, "pretrain": {"dataset": {"name": "src"}}}

    def fake_load(cfg_, ds_cfg, *, seed):
        torch.rand(3)  # a source build may consume the global RNG
        return _graphs(seed=5)

    def fake_proxy(*, cfg, model, driver, pretrain_cfg, pretrain_extra, device):
        proxy_cfg = cfg.finetune.gaptune.proxy
        torch.rand(int(proxy_cfg.num_graphs) * (2 if proxy_cfg.mode == "inverted" else 1))
        return _graphs(seed=6), {}

    monkeypatch.setattr(gaptune_module, "load_pretraining_graphs", fake_load)
    monkeypatch.setattr(gaptune_module, "build_proxy_source_graphs", fake_proxy)

    def order(num_graphs=16, mode="inverted", **arm):
        cfg = _cfg("gcn", **arm)
        cfg.finetune.gaptune.proxy.num_graphs = num_graphs
        cfg.finetune.gaptune.proxy.mode = mode
        encoder = _encoder(cfg)
        loader = DataLoader(list(range(40)), batch_size=8, shuffle=True)
        torch.manual_seed(cfg.seed)  # finetuner set_seed(s)
        task = FinetuneGapTune(cfg)
        task.validate_encoder(encoder)
        with preserve_loader_rng(loader):
            task.prepare_with_encoder(
                model=encoder, device=torch.device("cpu"), pretrain_cfg=pretrain_cfg, pretrain_extra={}
            )
        return [batch.tolist() for _ in range(2) for batch in loader]

    reference = order()
    arms = [
        {"plus": False},
        {"plus": False, "num_graphs": 4},
        {"plus": False, "mode": "random"},
        {"num_queries": 4},
        {"prompt_locations": "node"},
        {"prompt_locations": "message"},
        {"prompt_locations": "none"},
        {"value_mode": "free"},
        {"query_mode": "frozen"},
        {"query_mode": "untied"},
        {"mixture": "global"},
        {"gate": "nonnegative"},
    ]
    for arm in arms:
        assert order(**arm) == reference, arm


# --------------------------------------------------------------------------
# Config, run names, monitor, registration
# --------------------------------------------------------------------------


def test_validate_cfg():
    FinetuneGapTune.validate_cfg(_cfg("gcn"))
    bad = [
        _cfg("mlp"),
        _cfg("gcn", value_mode="bogus"),
        _cfg("gcn", value_mode="free", query_mode="untied"),
        _cfg("gcn", num_queries=0),
        _cfg("gcn", tau_c=0.0),
    ]
    non_induced = _cfg("gcn")
    non_induced.finetune.dataset.induced = False
    bad.append(non_induced)
    for cfg in bad:
        with pytest.raises(ValueError):
            FinetuneGapTune.validate_cfg(cfg)


def test_run_names_distinguish_every_arm():
    arms = [
        {},
        {"plus": False},
        {"value_mode": "free"},
        {"prompt_locations": "none"},
        {"query_mode": "untied"},
        {"mixture": "uniform"},
        {"gate": "nonnegative"},
        {"source_max_nodes": 256},
        {"lr": 0.01},
    ]
    names = set()
    for arm in arms:
        cfg = _cfg("gcn", **arm)
        names.add(
            build_finetune_run_name_from_cfg(
                cfg, split=(5, 0.0, 1.0), task_level_raw="node", task_cls=FinetuneGapTune,
                finetune_method="gaptune", pretrained_run_name="pre", freeze_pretrained_effective=True,
            )
        )
    assert len(names) == len(arms)
    assert FinetuneGapTune.run_tag(_cfg()) == "plus1" and FinetuneGapTune.variant_tag(_cfg()) == ""
    assert FinetuneGapTune.variant_tag(_cfg(value_mode="free", prompt_locations="none")) == "vfree-locnone"
    proxy_cfg = _cfg(plus=False)
    proxy_cfg.finetune.gaptune.proxy.num_graphs = 4
    assert FinetuneGapTune.variant_tag(proxy_cfg) == "pxb4"
    proxy_cfg.finetune.gaptune.plus = True  # proxy settings are inert for GapTune+
    assert FinetuneGapTune.variant_tag(proxy_cfg) == ""


def test_cli_overrides_and_monitor():
    cfg = _cfg()
    cfg.merge_from_list([
        "finetune.gaptune.value_mode", "free",
        "finetune.gaptune.proxy.num_graphs", "4",
        "finetune.gaptune.proxy.mode", "random",
        "finetune.gaptune.lr", "0.01",
    ])
    assert cfg.finetune.gaptune.proxy.num_graphs == 4
    assert FinetuneGapTune.resolve_default_monitor(cfg) is None
    cfg.finetune.dataset.label_dim = 617
    assert FinetuneGapTune.resolve_default_monitor(cfg).name == "val_auc"
    cfg.finetune.dataset.fixed_split = (5, 0.0, 1.0)
    assert FinetuneGapTune.resolve_default_monitor(cfg) is None


def test_tsv_column_and_result_table_rows(tmp_path):
    assert _finetune_custom_parser("gaptune_plus", "False", 1) == (False, True)
    tsv = tmp_path / "tasks.tsv"
    tsv.write_text(
        "# dataset\ttask_level\tinduced\tfinetune_method\tgaptune_plus\n"
        "chameleon\tnode\tTrue\tgaptune\tFalse\n"
    )
    (task,) = parse_finetune_tasks(str(tsv))
    assert task["finetune_method"] == "gaptune" and task["gaptune_plus"] is False
    assert {("gaptune", False), ("gaptune", True)} <= {(s.method, s.plus) for s in PROMPT_METHODS}
    assert _plus_from_task({}, "gaptune") is True
    assert _plus_from_result_row({"finetune.gaptune.plus": "False"}, "gaptune") is False
