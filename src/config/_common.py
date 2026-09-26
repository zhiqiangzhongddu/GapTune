from yacs.config import CfgNode as CN


def set_general_cfg(cfg: CN) -> CN:
    """General settings shared across all workflows."""
    cfg.device = 0  # CUDA device index
    cfg.seeds = [42, 0, 100, 123, 2024]  # ordered seeds; run i uses seeds[i], single-seed workflows use seeds[0]

    # Publication-campaign provenance. Development runs remain permissive by
    # default; publication=True activates fail-closed validation before any
    # training starts. The source digest is supplied by the frozen campaign
    # design and checked against the live source tree.
    cfg.provenance = CN()
    cfg.provenance.publication = False
    cfg.provenance.campaign_id = ""
    cfg.provenance.design_manifest_path = ""
    cfg.provenance.design_manifest_digest = ""
    cfg.provenance.source_tree_digest = ""
    # Exact git commit checked by frozen workflow launchers and persisted as a
    # result identity. The source-tree digest remains the byte-level authority.
    cfg.provenance.source_commit = ""
    # Bound internally once the exact pretrained checkpoint is resolved.
    cfg.provenance.pretrained_checkpoint_path = ""
    cfg.provenance.pretrained_checkpoint_sha256 = ""
    cfg.provenance.pretrained_checkpoint_config_sha256 = ""
    # Populated internally after a successful publication-mode source audit.
    cfg.provenance.source_tree_path_count = 0
    cfg.provenance.source_tree_paths = []

    return cfg
