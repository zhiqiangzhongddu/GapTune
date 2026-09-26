"""Fail-closed provenance helpers for frozen publication campaigns.

Development runs keep the historical permissive behaviour.  A run only
becomes a publication campaign when ``provenance.publication`` is true; that
mode requires a campaign id, a design-manifest digest, and an expected source
tree digest matching the live checkout before training can start.

The campaign design manifest must either live outside the hashed source roots
or omit the expected source digest from its own content (and pass that digest
separately via CLI).  This avoids a self-referential source/design hash cycle.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.paths import PROJECT_ROOT


_SOURCE_ROOTS = frozenset({"config", "configs", "scripts", "slurm", "src", "tests"})
_CACHE_PARTS = frozenset({
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
})
_RUNTIME_DIR_PARTS = frozenset({"log", "logs", "output", "outputs", "tmp"})
_SOURCE_SUFFIXES = frozenset({
    ".cfg",
    ".ini",
    ".json",
    ".py",
    ".pyi",
    ".sh",
    ".slurm",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
})
_ROOT_SOURCE_NAMES = frozenset({"pyproject.toml", "requirements.txt"})
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class CampaignProvenanceError(ValueError):
    """Raised when a publication campaign cannot be proven reproducible."""


@dataclass(frozen=True)
class SourceTreeSnapshot:
    """Content digest and the exact repository-relative paths it covers."""

    digest: str
    paths: tuple[str, ...]

    @property
    def path_count(self) -> int:
        return len(self.paths)


def _is_relevant_source_path(relative_path: str) -> bool:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return False
    if any(part in _CACHE_PARTS for part in path.parts):
        return False
    if len(path.parts) == 1:
        name = path.name
        return (
            name in _ROOT_SOURCE_NAMES
            or name.startswith("requirements") and name.endswith(".txt")
            or name.startswith("run_") and name.endswith(".py")
            or name.endswith((".yaml", ".yml"))
        )
    if path.parts[0] not in _SOURCE_ROOTS:
        return False
    if any(part.lower() in _RUNTIME_DIR_PARTS for part in path.parts[1:-1]):
        return False
    return path.suffix.lower() in _SOURCE_SUFFIXES


def _git_tracked_candidates(root: Path) -> list[str] | None:
    """Return tracked paths, or None outside git."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached"],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    ]


def _walk_source_candidates(root: Path) -> list[str]:
    """Filesystem fallback with the same explicit source-root boundary."""
    candidates: list[str] = []
    for source_root in sorted(_SOURCE_ROOTS):
        directory = root / source_root
        if not directory.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(directory):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in _CACHE_PARTS
                and name.lower() not in _RUNTIME_DIR_PARTS
            )
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                candidates.append(path.relative_to(root).as_posix())
    for path in root.iterdir():
        if path.is_file():
            candidates.append(path.relative_to(root).as_posix())
    return candidates


def compute_source_tree_snapshot(root: str | os.PathLike[str] = PROJECT_ROOT) -> SourceTreeSnapshot:
    """Hash relevant tracked and untracked source/config/launcher content.

    Paths are sorted and framed with their byte lengths before file content is
    added, so neither concatenation ambiguity nor filesystem traversal order can
    affect the digest.  Data, outputs, git metadata, and cache directories are
    outside the allowlisted source roots and therefore cannot enter the digest.
    """
    root_path = Path(root).resolve()
    # Walk only allowlisted source roots so relevant untracked files are
    # included even if a local ignore rule happens to match them. Union with
    # git's tracked list to retain relevant tracked root-level files.
    raw_candidates = _walk_source_candidates(root_path)
    tracked_candidates = _git_tracked_candidates(root_path)
    if tracked_candidates is not None:
        raw_candidates.extend(tracked_candidates)

    paths = tuple(sorted({
        Path(path).as_posix()
        for path in raw_candidates
        if _is_relevant_source_path(path) and (root_path / path).is_file()
    }))
    digest = hashlib.sha256()
    digest.update(b"gaptune-source-tree-v1\0")
    for relative_path in paths:
        encoded_path = relative_path.encode("utf-8", errors="surrogateescape")
        content = (root_path / relative_path).read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return SourceTreeSnapshot(digest=digest.hexdigest(), paths=paths)


def compute_file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of one regular file without loading it at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    raise CampaignProvenanceError(
        "provenance.publication must be a boolean value, not "
        f"{value!r}."
    )


def _normalize_sha256(value: Any, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    if not _DIGEST_RE.fullmatch(text):
        raise CampaignProvenanceError(
            f"provenance.{field} must be a 64-character SHA-256 hex digest."
        )
    return text


def is_publication_campaign(cfg) -> bool:
    node = getattr(cfg, "provenance", None)
    return _as_bool(getattr(node, "publication", False))


def runtime_version_payload() -> dict[str, str]:
    """Return portable library/runtime versions without machine paths."""
    try:
        import torch
    except ImportError:  # pragma: no cover - all training environments use torch
        torch_version = "unavailable"
        cuda_version = "unavailable"
    else:
        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda or "none")
    try:
        import torch_geometric
    except ImportError:  # pragma: no cover - project dependency
        pyg_version = "unavailable"
    else:
        pyg_version = str(torch_geometric.__version__)
    return {
        "python": platform.python_version(),
        "pytorch": torch_version,
        "pyg": pyg_version,
        "cuda": cuda_version,
    }


def campaign_provenance_payload(
    cfg,
    *,
    split_digests: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    provenance_complete: bool = False,
) -> dict[str, Any]:
    """Return the stable provenance block persisted in run artifacts."""
    # Development runs deliberately have no top-level provenance block.  In
    # particular, do not import/query heavyweight runtime packages here.
    if not is_publication_campaign(cfg):
        return {}
    node = getattr(cfg, "provenance", None)
    payload: dict[str, Any] = {
        "publication": True,
        "campaign_id": str(getattr(node, "campaign_id", "") or "").strip(),
        "design_manifest_path": str(
            getattr(node, "design_manifest_path", "") or ""
        ).strip(),
        "design_manifest_digest": str(
            getattr(node, "design_manifest_digest", "") or ""
        ).strip(),
        "source_tree_digest": str(
            getattr(node, "source_tree_digest", "") or ""
        ).strip(),
        "source_tree_path_count": int(
            getattr(node, "source_tree_path_count", 0) or 0
        ),
        "source_tree_paths": list(getattr(node, "source_tree_paths", []) or []),
        "pretrained_checkpoint_path": str(
            getattr(node, "pretrained_checkpoint_path", "") or ""
        ).strip(),
        "pretrained_checkpoint_sha256": str(
            getattr(node, "pretrained_checkpoint_sha256", "") or ""
        ).strip(),
        "pretrained_checkpoint_config_sha256": str(
            getattr(node, "pretrained_checkpoint_config_sha256", "") or ""
        ).strip(),
        "runtime_versions": runtime_version_payload(),
        "provenance_complete": bool(provenance_complete),
        "split_digests": dict(split_digests or {})
        if split_digests is None or isinstance(split_digests, Mapping)
        else list(split_digests),
    }
    return payload


def _resolve_existing_file(
    value: Any,
    *,
    field: str,
    root: Path,
) -> Path:
    text = str(value or "").strip()
    if not text:
        raise CampaignProvenanceError(
            f"provenance.{field} is required when provenance.publication=True."
        )
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CampaignProvenanceError(
            f"provenance.{field} does not resolve to an existing file: {text!r}."
        ) from exc
    if not resolved.is_file():
        raise CampaignProvenanceError(
            f"provenance.{field} must identify an existing regular file: {resolved}."
        )
    return resolved


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def bind_pretrained_checkpoint_provenance(
    cfg,
    checkpoint_path: str | os.PathLike[str],
) -> str:
    """Bind a publication run to the exact bytes of its pretrained checkpoint.

    Development mode returns the supplied path unchanged and performs no hash
    work.  Publication mode canonicalizes the path, hashes it, and rejects a
    conflicting pre-existing binding on the config.
    """
    if not is_publication_campaign(cfg):
        return os.fspath(checkpoint_path)
    node = getattr(cfg, "provenance", None)
    if node is None:
        raise CampaignProvenanceError("Missing provenance config block.")
    path = _resolve_existing_file(
        checkpoint_path,
        field="pretrained_checkpoint_path",
        root=Path(PROJECT_ROOT).resolve(),
    )
    normalized_path = str(path)
    digest = compute_file_sha256(path)

    saved_path = str(getattr(node, "pretrained_checkpoint_path", "") or "").strip()
    if saved_path:
        saved_normalized = str(
            _resolve_existing_file(
                saved_path,
                field="pretrained_checkpoint_path",
                root=Path(PROJECT_ROOT).resolve(),
            )
        )
        if saved_normalized != normalized_path:
            raise CampaignProvenanceError(
                "Publication pretrained-checkpoint path conflicts with the "
                f"existing binding: {saved_normalized!r} != {normalized_path!r}."
            )
    saved_digest = str(
        getattr(node, "pretrained_checkpoint_sha256", "") or ""
    ).strip()
    if saved_digest:
        saved_digest = _normalize_sha256(
            saved_digest,
            field="pretrained_checkpoint_sha256",
        )
        if saved_digest != digest:
            raise CampaignProvenanceError(
                "Publication pretrained-checkpoint digest mismatch: "
                f"expected {saved_digest}, found {digest}."
            )
    config_digest = _normalize_sha256(
        getattr(node, "pretrained_checkpoint_config_sha256", ""),
        field="pretrained_checkpoint_config_sha256",
    )

    node.pretrained_checkpoint_path = normalized_path
    node.pretrained_checkpoint_sha256 = digest
    node.pretrained_checkpoint_config_sha256 = config_digest
    return normalized_path


def prepare_campaign_provenance(
    cfg,
    *,
    root: str | os.PathLike[str] = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate and materialize source provenance for a publication campaign.

    This function deliberately does no source-tree work for development runs;
    their defaults and runtime cost remain unchanged.  Publication mode is
    fail-closed and mutates only the internal audit fields after all checks pass.
    """
    if not is_publication_campaign(cfg):
        return {}

    node = getattr(cfg, "provenance", None)
    if node is None:
        raise CampaignProvenanceError("Missing provenance config block.")
    campaign_id = str(getattr(node, "campaign_id", "") or "").strip()
    if not campaign_id:
        raise CampaignProvenanceError(
            "provenance.campaign_id is required when provenance.publication=True."
        )
    design_digest = _normalize_sha256(
        getattr(node, "design_manifest_digest", ""),
        field="design_manifest_digest",
    )
    root_path = Path(root).resolve()
    design_path = _resolve_existing_file(
        getattr(node, "design_manifest_path", ""),
        field="design_manifest_path",
        root=root_path,
    )
    for source_root in _SOURCE_ROOTS:
        if _is_within(design_path, (root_path / source_root).resolve()):
            raise CampaignProvenanceError(
                "provenance.design_manifest_path must live outside the hashed "
                f"source roots; found {design_path}."
            )
    actual_design_digest = compute_file_sha256(design_path)
    if actual_design_digest != design_digest:
        raise CampaignProvenanceError(
            "Publication design-manifest digest mismatch: "
            f"expected {design_digest}, found {actual_design_digest}."
        )
    expected_source_digest = _normalize_sha256(
        getattr(node, "source_tree_digest", ""),
        field="source_tree_digest",
    )
    snapshot = compute_source_tree_snapshot(root_path)
    if snapshot.digest != expected_source_digest:
        raise CampaignProvenanceError(
            "Publication source-tree digest mismatch: "
            f"expected {expected_source_digest}, found {snapshot.digest}."
        )

    node.campaign_id = campaign_id
    node.design_manifest_path = str(design_path)
    node.design_manifest_digest = design_digest
    node.source_tree_digest = snapshot.digest
    node.source_tree_path_count = snapshot.path_count
    node.source_tree_paths = list(snapshot.paths)
    bound_checkpoint = str(
        getattr(node, "pretrained_checkpoint_path", "") or ""
    ).strip()
    bound_checkpoint_digest = str(
        getattr(node, "pretrained_checkpoint_sha256", "") or ""
    ).strip()
    bound_checkpoint_config_digest = str(
        getattr(node, "pretrained_checkpoint_config_sha256", "") or ""
    ).strip()
    if len(
        {
            bool(bound_checkpoint),
            bool(bound_checkpoint_digest),
            bool(bound_checkpoint_config_digest),
        }
    ) != 1:
        raise CampaignProvenanceError(
            "Publication pretrained-checkpoint path, byte digest, and config "
            "digest must either all be set or all be empty."
        )
    if bound_checkpoint:
        bind_pretrained_checkpoint_provenance(cfg, bound_checkpoint)
    return campaign_provenance_payload(cfg)


def saved_provenance_matches(
    current: Mapping[str, Any],
    saved: Mapping[str, Any] | None,
    *,
    require_split_digests: bool,
    require_complete: bool = True,
) -> bool:
    """Return whether a saved publication artifact matches current provenance."""
    if not saved:
        return False
    for key in (
        "publication",
        "campaign_id",
        "design_manifest_path",
        "design_manifest_digest",
        "source_tree_digest",
        "source_tree_path_count",
        "source_tree_paths",
        "pretrained_checkpoint_path",
        "pretrained_checkpoint_sha256",
        "pretrained_checkpoint_config_sha256",
        "runtime_versions",
    ):
        if saved.get(key) != current.get(key):
            return False
    if require_complete and saved.get("provenance_complete") is not True:
        return False
    if require_split_digests and saved.get("split_digests") != current.get("split_digests"):
        return False
    return True


def aggregate_run_provenance(
    cfg,
    runners: Sequence[Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Build a lossless ordered multi-seed provenance payload for one TSV row."""
    if not is_publication_campaign(cfg):
        return {}
    aggregate = campaign_provenance_payload(cfg, provenance_complete=True)
    if not aggregate.get("pretrained_checkpoint_path") or not aggregate.get(
        "pretrained_checkpoint_sha256"
    ) or not aggregate.get("pretrained_checkpoint_config_sha256"):
        raise CampaignProvenanceError(
            "Cannot persist publication provenance without an exact pretrained-"
            "checkpoint path, byte SHA-256, and config SHA-256 binding."
        )
    split_entries: list[dict[str, Any]] = []
    invariant_keys = (
        "publication",
        "campaign_id",
        "design_manifest_path",
        "design_manifest_digest",
        "source_tree_digest",
        "source_tree_path_count",
        "source_tree_paths",
        "pretrained_checkpoint_path",
        "pretrained_checkpoint_sha256",
        "pretrained_checkpoint_config_sha256",
        "runtime_versions",
        "provenance_complete",
    )
    if len(runners) != len(seeds):
        raise CampaignProvenanceError(
            "Cannot persist result provenance: runner/seed counts differ."
        )
    for runner, seed in zip(runners, seeds):
        runner_cfg = getattr(runner, "cfg", cfg)
        runner_seed = getattr(runner_cfg, "seed", None)
        if (
            isinstance(runner_seed, bool)
            or not isinstance(runner_seed, Integral)
            or isinstance(seed, bool)
            or not isinstance(seed, Integral)
            or int(runner_seed) != int(seed)
        ):
            raise CampaignProvenanceError(
                "Cannot persist result provenance: runner.cfg.seed does not "
                f"exactly match aggregate seed {seed!r} (found {runner_seed!r})."
            )
        if getattr(runner, "provenance_complete", None) is not True:
            raise CampaignProvenanceError(
                f"Publication run seed={int(seed)} has incomplete provenance artifacts."
            )
        split_digests = getattr(runner, "split_content_digests", {}) or {}
        item = campaign_provenance_payload(
            runner_cfg,
            split_digests=split_digests,
            provenance_complete=True,
        )
        for key in invariant_keys:
            if item.get(key) != aggregate.get(key):
                raise CampaignProvenanceError(
                    f"Cannot aggregate runs with different provenance.{key}."
                )
        if aggregate["publication"] and not split_digests:
            raise CampaignProvenanceError(
                f"Publication run seed={int(seed)} has no materialized split digests."
            )
        split_entries.append({
            "seed": int(seed),
            "digests": dict(split_digests),
        })
    aggregate["split_digests"] = split_entries
    return aggregate


__all__ = [
    "CampaignProvenanceError",
    "SourceTreeSnapshot",
    "aggregate_run_provenance",
    "bind_pretrained_checkpoint_provenance",
    "campaign_provenance_payload",
    "compute_file_sha256",
    "compute_source_tree_snapshot",
    "is_publication_campaign",
    "prepare_campaign_provenance",
    "runtime_version_payload",
    "saved_provenance_matches",
]
