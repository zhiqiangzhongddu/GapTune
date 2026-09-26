"""Prompt modules for finetuning."""

from .gpf import GPFPlusPrompt, GPFPrompt
from .edgeprompt import EdgePrompt, EdgePromptPlus
from .gaptune import ContextGapPrompt
from .gppt import GPPTPrompt
from .graphprompt import (
    GraphPrompt,
    GraphPromptPlusStageWise,
    compute_class_centers,
)
from .igap import IGAPLabelPrompt, IGAPPrompt
from .mtg import MessagePrototypeFusion, MTGPrompt
from .pronog import ProNoGConditionNet
from .supt import SUPTPrompt

__all__ = [
    "GPFPrompt",
    "GPFPlusPrompt",
    "EdgePrompt",
    "EdgePromptPlus",
    "ContextGapPrompt",
    "GPPTPrompt",
    "GraphPrompt",
    "GraphPromptPlusStageWise",
    "IGAPLabelPrompt",
    "IGAPPrompt",
    "MessagePrototypeFusion",
    "MTGPrompt",
    "ProNoGConditionNet",
    "SUPTPrompt",
    "compute_class_centers",
]

try:
    from .all_in_one import HeavyPrompt, LightPrompt

    __all__.extend(["HeavyPrompt", "LightPrompt"])
except ModuleNotFoundError:
    pass
