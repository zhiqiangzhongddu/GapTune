"""Prompt direction at fixed magnitude (paper App. C.5, Fig. 5).

Each repetition restores a trained GapTune(+) predictor (``src.analysis.predictor``)
with gap values or free values (``finetune.gaptune.value_mode free``) and
refits nothing.  Every value vector of type ``q`` is mapped by one orthogonal
``R_q(theta) = Q_q blockdiag(rot(theta), ..., rot(theta)[, 1]) Q_q^T``, shared by
all graphs of the type within the repetition: a fixed seeded random
orthonormal basis ``Q_q`` with Givens rotations of consecutive coordinate pairs
(an odd last coordinate stays fixed), computed as ``I + Q_q (B - I) Q_q^T`` so
that ``R_q(0) = I`` exactly.  Prompts are linear in the values, so
``p_theta = sum_k a_k tanh(theta_k) R v_k = R p`` (Eq. 70): the map is applied
to every prompt, and ``||p_theta|| = ||p||`` is verified numerically.
Reports test accuracy per angle and repetition.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from src.analysis.predictor import (
    finetuned_checkpoints,
    prompted_logits,
    reference_pass,
    restore_predictor,
    study_dir,
    typed_prompts,
    write_tsv,
)
from src.utils.checkpoint import save_json_atomic
from src.utils.pool import get_batch_vector

_ROTATION_STREAM = 7  # after the GapTune seed streams s .. s+6 (paper D.2)


def random_basis(dim: int, generator: torch.Generator) -> torch.Tensor:
    """Orthonormal ``[dim, dim]`` basis (float64) from the QR factorisation of a Gaussian matrix."""
    return torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float64))[0]


def rotation_matrix(basis: torch.Tensor, degrees: float) -> torch.Tensor:
    """``R(theta) = I + Q (B(theta) - I) Q^T`` (float64); ``B`` rotates coordinate pairs (0,1), (2,3), ..."""
    theta = math.radians(float(degrees))
    identity = torch.eye(basis.size(0), dtype=torch.float64)
    block = identity.clone()
    for i in range(0, basis.size(0) - 1, 2):
        block[i, i] = block[i + 1, i + 1] = math.cos(theta)
        block[i, i + 1], block[i + 1, i] = -math.sin(theta), math.sin(theta)
    return identity + basis @ (block - identity) @ basis.t()


def rotation_maps(prompt, degrees, seed: int, device) -> dict:
    """``{angle: {type: R_q(angle)}}`` with one seeded basis per type (types in ``N``, ``M1``.. order)."""
    generator = torch.Generator().manual_seed(int(seed) + _ROTATION_STREAM)
    bases = {key: random_basis(p.retained_source_context.size(-1), generator) for key, p in prompt.types.items()}
    return {
        float(angle): {key: rotation_matrix(basis, angle).to(device=device, dtype=torch.float32) for key, basis in bases.items()}
        for angle in degrees
    }


def rotate_prompts(node_prompt, message_prompts, rotations: dict):
    """``p -> R_q p`` for every prompt row of every type."""
    node = None if node_prompt is None else node_prompt @ rotations["N"].t()
    messages = None
    if message_prompts is not None:
        messages = [
            None if p is None else p @ rotations[f"M{layer}"].t() for layer, p in enumerate(message_prompts, start=1)
        ]
    return node, messages


@torch.no_grad()
def rotation_measures(task, model, loader, device, maps: dict) -> dict:
    """Accuracy (%) and the largest ``| ||p_theta|| - ||p|| |`` per angle of *maps*."""
    correct, norm_deviation, total = dict.fromkeys(maps, 0), dict.fromkeys(maps, 0.0), 0
    for data in loader:
        data = data.to(device)
        obs = reference_pass(task, model, data).obs
        prompts = task.prompt(obs, get_batch_vector(data), use_retained=True)
        reference = typed_prompts(*prompts)
        for angle, rotations in maps.items():
            rotated = rotate_prompts(*prompts, rotations)
            for key, value in typed_prompts(*rotated).items():
                gap = (value.norm(dim=-1) - reference[key].norm(dim=-1)).abs().max()
                norm_deviation[angle] = max(norm_deviation[angle], float(gap))
            logits, labels = prompted_logits(task, model, data, *rotated)
            correct[angle] += int((logits.argmax(dim=-1) == labels).sum())
        total += labels.numel()
    return {angle: {"accuracy": 100.0 * correct[angle] / total, "norm_deviation": norm_deviation[angle]} for angle in maps}


def run_rotation(cfg) -> int:
    """``analysis.study rotation``: write per-seed JSON/TSV and summary.tsv."""
    degrees = [float(angle) for angle in cfg.analysis.rotation.degrees]
    if 0.0 not in degrees:
        raise ValueError("[Analysis] analysis.rotation.degrees must include 0 (the trained predictor).")
    repetitions = finetuned_checkpoints(cfg)
    out_dir = study_dir(cfg, "rotation", repetitions[0][1])
    rows = []
    for seed, checkpoint in repetitions:
        runner, payload = restore_predictor(cfg, checkpoint)
        value_mode = runner.task.method_cfg.value_mode
        maps = rotation_maps(runner.task.prompt, degrees, seed, runner.device)
        measures = rotation_measures(runner.task, runner.model, runner.test_loader, runner.device, maps)
        seed_rows = [
            {
                "seed": seed,
                "value_mode": value_mode,
                "degrees": angle,
                "accuracy": m["accuracy"],
                "delta_acc": m["accuracy"] - measures[0.0]["accuracy"],
                "norm_deviation": m["norm_deviation"],
            }
            for angle, m in measures.items()
        ]
        save_json_atomic(
            os.path.join(out_dir, f"seed{seed}.json"),
            {
                "seed": seed,
                "checkpoint": checkpoint,
                "value_mode": value_mode,
                "checkpoint_test_acc": 100.0 * float((payload.get("metrics") or {}).get("test_acc", float("nan"))),
                "measures": {str(angle): m for angle, m in measures.items()},
            },
        )
        write_tsv(os.path.join(out_dir, f"seed{seed}.tsv"), seed_rows)
        rows.extend(seed_rows)
    summary = []
    for angle in degrees:
        group = [row for row in rows if row["degrees"] == angle]
        accuracy = [row["accuracy"] for row in group]
        summary.append(
            {
                "value_mode": group[0]["value_mode"],
                "degrees": angle,
                "repetitions": len(group),
                "accuracy_mean": float(np.mean(accuracy)),
                "accuracy_sd": float(np.std(accuracy, ddof=1)) if len(group) > 1 else float("nan"),
                "delta_acc": float(np.mean([row["delta_acc"] for row in group])),
                "norm_deviation": max(row["norm_deviation"] for row in group),
            }
        )
        entry = summary[-1]
        print(
            f"[Analysis][rotation] {entry['value_mode']} theta={angle:g} acc={entry['accuracy_mean']:.2f}"
            f"+-{entry['accuracy_sd']:.2f} dAcc={entry['delta_acc']:+.2f} max|dnorm|={entry['norm_deviation']:.2e}"
        )
    write_tsv(os.path.join(out_dir, "summary.tsv"), summary)
    print(f"[Analysis][rotation] Results in {out_dir}")
    return 0
