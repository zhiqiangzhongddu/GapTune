"""GapTune driver registry: dispatch by ``cfg.model.name`` or by encoder."""

from __future__ import annotations

from torch import nn

from .base import GapTuneDriver
from .fagcn import FAGCNDriver
from .gnn import GATDriver, GCNDriver, GINDriver
from .h2gcn import H2GCNDriver
from .nodeformer import NodeFormerDriver
from .spec import resolve_gaptune_spec
from .transformer import TransformerDriver


GAPTUNE_DRIVERS: dict[str, type[GapTuneDriver]] = {
    "gcn": GCNDriver,
    "gin": GINDriver,
    "gat": GATDriver,
    "transformer": TransformerDriver,
    "fagcn": FAGCNDriver,
    "h2gcn": H2GCNDriver,
    "nodeformer": NodeFormerDriver,
}


def build_gaptune_driver(cfg) -> type[GapTuneDriver]:
    """Driver class for ``cfg.model.name``; raises ``ValueError`` if unsupported."""
    return GAPTUNE_DRIVERS[resolve_gaptune_spec(cfg).model_name]


def get_gaptune_driver(model: nn.Module) -> type[GapTuneDriver]:
    """Driver class for a vanilla encoder instance."""
    for driver in GAPTUNE_DRIVERS.values():
        if driver.supports_model(model):
            return driver
    raise ValueError(f"GapTune has no driver for encoder {type(model).__name__}.")
