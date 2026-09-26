"""App. C.5 prompt rotation at fixed magnitude on a tiny synthetic GapTune task.

- ``R(theta)`` is orthogonal, ``R(0) = I`` exactly, rotates each basis pair by
  ``theta`` and fixes the last basis direction of an odd dimension;
- rotating prompts preserves every prompt norm and equals rotating every
  value vector (Eq. 70) for gap and free values;
- the zero angle reproduces the trained predictor exactly, and the analysis
  logits equal the task's own evaluation outputs.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import Batch

from src.analysis.predictor import prompted_logits, reference_pass
from src.analysis.rotation import random_basis, rotate_prompts, rotation_maps, rotation_matrix, rotation_measures
from tests.finetune.test_gaptune import _cfg, _graphs, _set_gates, _task

CPU = torch.device("cpu")


def _toy(value_mode: str = "gap"):
    task, encoder = _task(_cfg("gcn", value_mode=value_mode))
    _set_gates(task)
    task.eval()
    return task, encoder, [Batch.from_data_list(_graphs(seed=seed)) for seed in (0, 1, 2)]


def test_rotation_matrix_is_orthogonal_and_zero_is_identity():
    basis = random_basis(5, torch.Generator().manual_seed(0))
    torch.testing.assert_close(basis.t() @ basis, torch.eye(5, dtype=torch.float64))
    assert torch.equal(rotation_matrix(basis, 0), torch.eye(5, dtype=torch.float64))
    rotation = rotation_matrix(basis, 30)
    torch.testing.assert_close(rotation @ rotation.t(), torch.eye(5, dtype=torch.float64))
    # Each basis pair turns by theta; the odd last direction stays fixed.
    cosine = basis[:, 0] @ (rotation @ basis[:, 0])
    assert math.isclose(float(cosine), math.cos(math.radians(30)), abs_tol=1e-12)
    torch.testing.assert_close(rotation @ basis[:, 4], basis[:, 4])
    turned = rotation_matrix(torch.eye(2, dtype=torch.float64), 90) @ torch.tensor([1.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(turned, torch.tensor([0.0, 1.0], dtype=torch.float64))


@pytest.mark.parametrize("value_mode", ("gap", "free"))
def test_rotation_preserves_prompt_norms_and_rotates_every_value(value_mode):
    task, encoder, loader = _toy(value_mode)
    data = loader[0]
    maps = rotation_maps(task.prompt, [0, 45], seed=3, device=CPU)
    obs = reference_pass(task, encoder, data).obs
    node_prompt, message_prompts = task.prompt(obs, data.batch, use_retained=True)
    node, messages = rotate_prompts(node_prompt, message_prompts, maps[45.0])
    for rotated, prompt in zip([node, *messages], [node_prompt, *message_prompts]):
        assert not torch.allclose(rotated, prompt)
        torch.testing.assert_close(rotated.norm(dim=-1), prompt.norm(dim=-1), atol=1e-5, rtol=1e-5)
    # Eq. 70: the same prompts as rotating every value vector v_k of type N.
    type_prompt = task.prompt.types["N"]
    values = type_prompt.values(obs.h0, data.batch, data.num_graphs, use_retained=True)
    weights = type_prompt.mixture_weights(torch.cat([obs.h0, obs.h_final], dim=-1)) * torch.tanh(type_prompt.gates)
    rotated_values = values @ maps[45.0]["N"].t()
    expected = sum(weights[:, k : k + 1] * rotated_values[data.batch, k] for k in range(values.size(1)))
    torch.testing.assert_close(node, expected, atol=1e-5, rtol=1e-5)


def test_zero_angle_reproduces_the_trained_predictor():
    task, encoder, loader = _toy()
    maps = rotation_maps(task.prompt, [0, 90], seed=3, device=CPU)
    assert all(torch.equal(r, torch.eye(r.size(0))) for r in maps[0.0].values())
    measures = rotation_measures(task, encoder, loader, CPU, maps)
    correct = total = 0
    with torch.no_grad():
        for data in loader:
            obs = reference_pass(task, encoder, data).obs
            logits, labels = prompted_logits(task, encoder, data, *task.prompt(obs, data.batch, use_retained=True))
            # The analysis path is the task's own evaluation forward (encode, readout, head).
            _, _, task_logits, task_labels = task._forward(encoder, data, CPU, return_outputs=True)
            assert torch.equal(logits, task_logits) and torch.equal(labels, task_labels)
            correct += int((logits.argmax(dim=-1) == labels).sum())
            total += labels.numel()
    assert measures[0.0] == {"accuracy": 100.0 * correct / total, "norm_deviation": 0.0}
    assert measures[90.0]["norm_deviation"] < 1e-5
