"""GapTune forward drivers over the vanilla encoders.

Public API:
- ``build_gaptune_driver(cfg)`` / ``get_gaptune_driver(model)`` return the
  per-backbone ``GapTuneDriver`` class;
- ``driver.forward(model, data, *, node_prompt, message_prompts, collect,
  edge_weight, projections) -> DriverOutput(node_repr, graph_repr, obs)``;
- ``driver.observation_dims(model)`` / ``driver.descriptor_dims(model)``;
- ``resolve_gaptune_spec(cfg)`` for the self-loop convention.

See ``GAPTUNE.md`` for the per-backbone observation definitions.
"""

from __future__ import annotations

from .base import DriverOutput, GapTuneDriver, GapTuneObservations, LayerObs
from .factory import GAPTUNE_DRIVERS, build_gaptune_driver, get_gaptune_driver
from .spec import GapTuneSpec, resolve_gaptune_spec, supported_gaptune_backbones

__all__ = [
    "GAPTUNE_DRIVERS",
    "DriverOutput",
    "GapTuneDriver",
    "GapTuneObservations",
    "GapTuneSpec",
    "LayerObs",
    "build_gaptune_driver",
    "get_gaptune_driver",
    "resolve_gaptune_spec",
    "supported_gaptune_backbones",
]
