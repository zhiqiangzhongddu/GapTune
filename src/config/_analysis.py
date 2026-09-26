from yacs.config import CfgNode as CN


def set_analysis_cfg(cfg: CN) -> CN:
    """Post-hoc analysis studies (paper App. A, C.5-C.7, C.10); see src/analysis/run.py."""
    cfg.analysis = CN()
    cfg.analysis.study = ""  # study registry key in src/analysis/run.py (required)
    cfg.analysis.output_dir = "outputs/analysis"  # results go to <output_dir>/<study>/<run_tag>/
    # seeds of the ten-repetition studies (App. A, C.7): cfg.seeds extended deterministically
    cfg.analysis.repetitions = [42, 0, 100, 123, 2024, 1, 2, 3, 4, 5]
    # split files of the App. A / C.7 repetitions, created here on first use (the shared data/splits stays untouched)
    cfg.analysis.split_root = "outputs/analysis/splits"
    cfg.analysis.pretrained_checkpoint = ""  # frozen pretrained encoder; "" resolves from pretrain.* / model.*
    # restored GapTune(+) predictor (C.5 / C.6), one per seed of finetune.num_runs ("{seed}" in the path);
    # "" resolves each seed's checkpoint from finetune.* like the finetune runner
    cfg.analysis.finetuned_checkpoint = ""
    # App. A.4 context-gap sampling budgets B_H / B_M (unspecified in the paper)
    cfg.analysis.node_budget = 2048
    cfg.analysis.message_budget = 4096
    # App. A prompt transferability (src/analysis/transfer.py); fits reuse finetune.epochs and EdgePrompt's lr / wd
    cfg.analysis.transfer = CN()
    cfg.analysis.transfer.strengths = [0.05, 0.10, 0.20, 0.30]  # nonzero perturbation strengths A
    cfg.analysis.transfer.bootstrap_samples = 2000  # percentile bootstrap resamples of complete repetitions
    # App. C.7 prompt values under controlled shifts (src/analysis/controlled_shift.py); arms use finetune.gaptune
    cfg.analysis.controlled_shift = CN()
    cfg.analysis.controlled_shift.datasets = ["photo", "chameleon"]  # full graphs, each with its within-dataset checkpoint
    cfg.analysis.controlled_shift.strengths = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]  # 0 is the shared control
    cfg.analysis.controlled_shift.updates = 500  # common full-batch update budget of every arm
    # support split of each repetition seed; one with validation nodes selects on val_acc (finetune runner rule)
    cfg.analysis.controlled_shift.fixed_split = (5, 0.0, 1.0)
    # App. C.6 fixed-predictor source-context replacement (src/analysis/replacement.py); proxies use finetune.gaptune.proxy
    cfg.analysis.replacement = CN()
    cfg.analysis.replacement.budgets = [4, 16, 64]  # proxy graphs B per collection
    cfg.analysis.replacement.proxy_modes = ["inverted", "random"]
    # App. C.5 prompt direction at fixed magnitude (src/analysis/rotation.py)
    cfg.analysis.rotation = CN()
    cfg.analysis.rotation.degrees = [0, 15, 30, 45, 60, 75, 90]  # theta of R_q(theta)
    # App. C.10 cost accounting (src/analysis/cost.py); inversion updates and source caps use finetune.gaptune
    cfg.analysis.cost = CN()
    cfg.analysis.cost.timing_blocks = 5  # repeated measurements summarized as mean +- sample SD
    cfg.analysis.cost.updates = 500  # adaptation updates on the fixed batch
    cfg.analysis.cost.evaluations = 20  # fixed-batch evaluations, one every updates / evaluations updates

    return cfg
