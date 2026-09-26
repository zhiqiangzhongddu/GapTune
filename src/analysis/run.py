"""Analysis entrypoint helpers and CLI wiring.

``analysis.study <name>`` selects one study runtime from :data:`STUDIES`.
"""

from __future__ import annotations

import sys
import warnings
from typing import Callable, Iterable, Optional

from src.config import cfg as base_cfg, update_cfg
from src.utils.save_results import extract_explicit_cfg_keys, set_explicit_cfg_keys

from .cost import run_cost_study
from .replacement import run_replacement
from .rotation import run_rotation
from .transfer import run_transfer

# ``analysis.study`` -> runtime(cfg) -> exit status. ``None`` marks a planned
# study without a runtime yet.
STUDIES: dict[str, Optional[Callable]] = {
    "transfer": run_transfer,  # App. A: prompt transferability (Fig. 1)
    "controlled_shift": None,  # App. C.7: prompt values under controlled shifts (Table 9, Fig. 7)
    "replacement": run_replacement,  # App. C.6: fixed-predictor source-context replacement (Tables 7-8, Fig. 6)
    "rotation": run_rotation,  # App. C.5: prompt direction at fixed magnitude (Fig. 5)
    "cost": run_cost_study,  # App. C.10: parameter / time / memory accounting (Tables 12-13)
}


def resolve_study(cfg) -> Callable:
    """Runtime registered for ``cfg.analysis.study``; ``ValueError`` if unknown or unimplemented."""
    name = str(cfg.analysis.study).strip()
    if name not in STUDIES:
        raise ValueError(
            f"[Analysis] analysis.study must be one of {', '.join(STUDIES)}; got '{name}'."
        )
    if STUDIES[name] is None:
        raise ValueError(f"[Analysis] analysis.study '{name}' is not implemented yet.")
    return STUDIES[name]


def build_analysis_cfg(argv: Iterable[str]):
    """Parse CLI overrides (``--config`` + KEY VALUE pairs) and validate the study."""
    forwarded_argv = list(argv)
    cfg = update_cfg(base_cfg, forwarded_argv)
    set_explicit_cfg_keys(cfg, extract_explicit_cfg_keys(forwarded_argv, flag_arity={"--config": 1}))
    resolve_study(cfg)
    return cfg


def run_analysis_from_cli(argv: Iterable[str]) -> int:
    """Parse CLI overrides and execute the selected study."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_sparse")
    try:
        cfg = build_analysis_cfg(argv)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return resolve_study(cfg)(cfg)
