"""GapTune per-backbone spec resolution.

Single source of truth for which backbones the GapTune drivers support,
whether the native operator adds self-loop messages (so a caller can
exclude them from observation collections), and a short message formula.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GapTuneSpec:
    model_name: str
    adds_self_loops: bool  # native operator appends one loop message per node
    formula: str


_ADDS_SELF_LOOPS: dict[str, bool] = {
    "gcn": True,
    "gin": False,
    "gat": True,
    "transformer": False,
    "fagcn": True,
    "h2gcn": True,
    "nodeformer": False,
}


_FORMULA: dict[str, str] = {
    "gcn": "m = norm_vu * W h_u (+p); bias in UPD",
    "gin": "m = w_vu * h_u (+p); (1+eps) h_v and MLP in UPD",
    "gat": "m = alpha_vu * W h_u per head (+p); head mean and bias in UPD",
    "transformer": "m = alpha_vu * (W_V h_u + b_V) per head (+p); head mean and lin_skip(h_v) in UPD",
    "fagcn": "m = tanh(a_l h_u + a_r h_v) * norm_vu * h_u (+p); eps * h~0 in UPD",
    "h2gcn": "m = S_vu h_u per hop (+p), sender=col, receiver=row; lin/act/concat unchanged",
    "nodeformer": "m = alpha_vu * v_u on input edges, fixed projections; a~ = u~ + r~ + sum p before Wo",
}


def supported_gaptune_backbones() -> tuple[str, ...]:
    return tuple(sorted(_ADDS_SELF_LOOPS.keys()))


def resolve_gaptune_spec(cfg) -> GapTuneSpec:
    name = str(getattr(getattr(cfg, "model", None), "name", "") or "").lower()
    if name not in _ADDS_SELF_LOOPS:
        raise ValueError(
            f"GapTune does not support model '{name}'. "
            f"Supported backbones: {list(supported_gaptune_backbones())}."
        )
    return GapTuneSpec(
        model_name=name,
        adds_self_loops=_ADDS_SELF_LOOPS[name],
        formula=_FORMULA[name],
    )
