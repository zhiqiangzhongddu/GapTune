"""Prompt-aware encoder modules for finetuning.

Canonical implementations live in :mod:`src.finetune.encoders.edgeprompt`,
:mod:`src.finetune.encoders.graphprompt_plus`, :mod:`src.finetune.encoders.gaptune`
and :mod:`src.finetune.encoders.mtg`; :mod:`src.finetune.encoders.nodeformer_fixed`
holds the fixed-projection NodeFormer conv shared by GapTune and MTG.
"""

from .edgeprompt import (
    PROMPT_ENCODERS,
    EdgePromptSpec,
    PromptAwareEncoder,
    build_prompt_encoder,
    resolve_edgeprompt_prompt_spec,
    supported_edgeprompt_backbones,
)
from .graphprompt_plus import (
    GRAPHPROMPT_PLUS_ADAPTERS,
    GraphPromptPlusAdapter,
    GraphPromptPlusSpec,
    build_graphprompt_plus_adapter,
    resolve_graphprompt_plus_spec,
    supported_graphprompt_plus_backbones,
)
from .gaptune import (
    GAPTUNE_DRIVERS,
    DriverOutput,
    GapTuneDriver,
    GapTuneObservations,
    GapTuneSpec,
    LayerObs,
    build_gaptune_driver,
    get_gaptune_driver,
    resolve_gaptune_spec,
    supported_gaptune_backbones,
)
from .nodeformer_fixed import build_fixed_projections, nodeformer_conv_forward

__all__ = [
    "PROMPT_ENCODERS",
    "EdgePromptSpec",
    "PromptAwareEncoder",
    "build_prompt_encoder",
    "resolve_edgeprompt_prompt_spec",
    "supported_edgeprompt_backbones",
    "GRAPHPROMPT_PLUS_ADAPTERS",
    "GraphPromptPlusAdapter",
    "GraphPromptPlusSpec",
    "build_graphprompt_plus_adapter",
    "resolve_graphprompt_plus_spec",
    "supported_graphprompt_plus_backbones",
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
    "build_fixed_projections",
    "nodeformer_conv_forward",
]
