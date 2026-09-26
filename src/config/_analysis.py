from yacs.config import CfgNode as CN


def set_analysis_cfg(cfg: CN) -> CN:
    """Post-hoc analysis studies (paper App. A, C.5-C.7, C.10); see src/analysis/run.py."""
    cfg.analysis = CN()
    cfg.analysis.study = ""  # study registry key in src/analysis/run.py (required)
    cfg.analysis.output_dir = "outputs/analysis"  # results go to <output_dir>/<study>/<run_tag>/
    # seeds of the ten-repetition studies (App. A, C.7): cfg.seeds extended deterministically
    cfg.analysis.repetitions = [42, 0, 100, 123, 2024, 1, 2, 3, 4, 5]
    cfg.analysis.pretrained_checkpoint = ""  # frozen pretrained encoder; "" resolves from pretrain.* / model.*
    cfg.analysis.finetuned_checkpoint = ""  # restored GapTune(+) predictor (C.5 / C.6); "" resolves from finetune.*
    # App. A.4 context-gap sampling budgets B_H / B_M (unspecified in the paper)
    cfg.analysis.node_budget = 2048
    cfg.analysis.message_budget = 4096

    return cfg
