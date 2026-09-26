import os
import argparse

from yacs.config import CfgNode as CN

from ._common import set_general_cfg
from ._save_results import set_save_results_cfg
from ._data_preparation import set_data_preparation_cfg
from ._model import set_model_cfg
from ._train import set_train_cfg
from ._pretrain import set_pretrain_cfg
from ._finetune import set_finetune_cfg
from ._analysis import set_analysis_cfg


def set_cfg(cfg: CN) -> CN:

    # Delegate to sub-modules
    set_general_cfg(cfg)
    set_save_results_cfg(cfg)
    set_data_preparation_cfg(cfg)
    set_model_cfg(cfg)
    set_train_cfg(cfg)
    set_pretrain_cfg(cfg)
    set_finetune_cfg(cfg)
    set_analysis_cfg(cfg)

    return cfg


def update_cfg(cfg: CN, argv: list[str] | None = None) -> CN:
    """Update *cfg* from CLI arguments.

    *argv* can be:
    - ``None`` — parse from ``sys.argv`` (interactive use).
    - a ``list[str]`` — pre-tokenised argv (preserves arguments with spaces).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default="",
        metavar="FILE",
        help="Path to config file"
    )
    # opts arg needs to match set_cfg
    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)
    # Clone the original cfg
    cfg = cfg.clone()

    # An explicitly requested config must exist. Silently falling back to
    # defaults turns path typos into unrelated experiment runs.
    if args.config:
        if not os.path.isfile(args.config):
            # Workflow CLI boundaries consistently catch ``ValueError`` and
            # turn it into a concise non-zero exit rather than a traceback.
            raise ValueError(f"Config file not found: {args.config}")
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(cfg=CN())
