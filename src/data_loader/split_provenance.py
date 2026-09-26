"""Canonical content digests for materialized downstream data splits."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch
from torch.utils.data import Subset


_SPLIT_NAMES = ("train", "val", "test")
_NODE_LABEL_FIELDS = frozenset({
    "edge_label",
    "edge_label_index",
    "test_mask",
    "train_mask",
    "val_mask",
    "y",
})
_PATH_METADATA_FIELDS = frozenset({
    "file_path",
    "filename",
    "path",
    "processed_dir",
    "raw_dir",
    "root",
})


def _frame(digest, tag: bytes, payload: bytes = b"") -> None:
    digest.update(len(tag).to_bytes(4, "big"))
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu()
    if tensor.layout != torch.strided:
        tensor = tensor.to_dense()
    tensor = tensor.contiguous()
    if tensor.numel() == 0:
        return b""
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def _update_canonical(digest, value: Any) -> None:
    """Hash supported values without repr(), filesystem paths, or object ids."""
    if value is None:
        _frame(digest, b"none")
        return
    if isinstance(value, bool):
        _frame(digest, b"bool", b"1" if value else b"0")
        return
    if isinstance(value, int):
        _frame(digest, b"int", str(value).encode("ascii"))
        return
    if isinstance(value, float):
        # Canonicalize all NaNs; finite values and signed infinities retain
        # their exact IEEE-754 representation.
        payload = b"nan" if math.isnan(value) else struct.pack(">d", value)
        _frame(digest, b"float", payload)
        return
    if isinstance(value, str):
        _frame(digest, b"str", value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        _frame(digest, b"bytes", value)
        return
    if isinstance(value, torch.Tensor):
        _frame(digest, b"tensor-dtype", str(value.dtype).encode("ascii"))
        _update_canonical(digest, list(value.shape))
        _frame(digest, b"tensor-content", _tensor_bytes(value))
        return

    # NumPy is available through torch in the project environment, but keeping
    # the import local avoids making it a module import requirement.
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - torch distributions include numpy here
        np = None
    if np is not None and isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object-dtype arrays are not valid split provenance content.")
        _frame(digest, b"ndarray-dtype", str(array.dtype).encode("ascii"))
        _update_canonical(digest, list(array.shape))
        _frame(digest, b"ndarray-content", array.tobytes(order="C"))
        return
    if np is not None and isinstance(value, np.generic):
        _update_canonical(digest, value.item())
        return

    if isinstance(value, Mapping):
        _frame(digest, b"mapping-size", str(len(value)).encode("ascii"))
        for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
            if not isinstance(key, (str, int)):
                raise TypeError(
                    "Split provenance only supports string/integer mapping keys; "
                    f"found {type(key).__name__}."
                )
            _update_canonical(digest, key)
            _update_canonical(digest, value[key])
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        _frame(digest, b"sequence-size", str(len(value)).encode("ascii"))
        for item in value:
            _update_canonical(digest, item)
        return
    if hasattr(value, "to_dict") and callable(value.to_dict):
        _update_canonical(digest, value.to_dict())
        return

    raise TypeError(
        "Unsupported split-provenance value type "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _subset_base_index(dataset, position: int) -> int:
    current = dataset
    index = int(position)
    while isinstance(current, Subset):
        index = int(current.indices[index])
        current = current.dataset
    return index


def _data_fields(data) -> dict[str, Any]:
    if hasattr(data, "to_dict") and callable(data.to_dict):
        return {
            key: value
            for key, value in data.to_dict().items()
            if str(key).lower() not in _PATH_METADATA_FIELDS
        }
    raise TypeError(
        "Materialized split samples must expose to_dict(); "
        f"found {type(data).__module__}.{type(data).__qualname__}."
    )


def _single_graph_digest(loader, split_name: str) -> tuple[str, int]:
    data = loader.data
    mask = getattr(data, f"{split_name}_mask", None)
    digest = hashlib.sha256()
    _frame(digest, b"schema", b"gaptune-materialized-split-v1")
    _update_canonical(digest, split_name)

    if mask is not None:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool).view(-1).cpu()
        indices = torch.nonzero(mask_tensor, as_tuple=False).view(-1).long()
        fields = {
            key: value
            for key, value in _data_fields(data).items()
            if key not in _NODE_LABEL_FIELDS
        }
        _update_canonical(digest, fields)
        _update_canonical(digest, indices)
        _update_canonical(digest, mask_tensor)
        labels = getattr(data, "y", None)
        if labels is not None:
            labels = torch.as_tensor(labels)
            if labels.dim() == 0:
                selected_labels = labels
            else:
                selected_labels = labels[indices]
            _update_canonical(digest, selected_labels)
        else:
            _update_canonical(digest, None)
        return digest.hexdigest(), int(indices.numel())

    # Edge-level single-graph loaders are already split-specific materialized
    # Data objects (message graph, query edges, and labels).
    _update_canonical(digest, 0)
    _update_canonical(digest, _data_fields(data))
    labels = getattr(data, "edge_label", None)
    count = int(torch.as_tensor(labels).numel()) if labels is not None else 1
    return digest.hexdigest(), count


def _dataset_digest(loader, split_name: str) -> tuple[str, int]:
    dataset = loader.dataset
    digest = hashlib.sha256()
    _frame(digest, b"schema", b"gaptune-materialized-split-v1")
    _update_canonical(digest, split_name)
    _update_canonical(digest, len(dataset))
    for position in range(len(dataset)):
        # Original dataset position is explicit sample identity; actual sample
        # content (including labels) is hashed separately and in split order.
        _update_canonical(digest, _subset_base_index(dataset, position))
        _update_canonical(digest, _data_fields(dataset[position]))
    return digest.hexdigest(), len(dataset)


def _absent_loader_digest(split_name: str) -> tuple[str, int]:
    """Canonicalize a workflow's explicit None loader as an empty split."""
    digest = hashlib.sha256()
    _frame(digest, b"schema", b"gaptune-materialized-split-v1")
    _update_canonical(digest, split_name)
    _update_canonical(digest, 0)
    _frame(digest, b"loader-kind", b"absent")
    return digest.hexdigest(), 0


def build_materialized_split_manifest(
    train_loader,
    val_loader,
    test_loader,
) -> dict[str, dict[str, Any]]:
    """Digest exact ordered split content independently of batching/workers."""
    manifest: dict[str, dict[str, Any]] = {}
    for split_name, loader in zip(
        _SPLIT_NAMES,
        (train_loader, val_loader, test_loader),
    ):
        if loader is None:
            content_digest, count = _absent_loader_digest(split_name)
        elif hasattr(loader, "data"):
            content_digest, count = _single_graph_digest(loader, split_name)
        elif hasattr(loader, "dataset"):
            content_digest, count = _dataset_digest(loader, split_name)
        else:
            raise TypeError(
                f"Unsupported materialized loader type: {type(loader).__qualname__}."
            )
        manifest[split_name] = {
            "sha256": content_digest,
            "num_samples": int(count),
        }
    return manifest


def validate_materialized_split_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed when any required materialized split digest is absent."""
    for split_name in _SPLIT_NAMES:
        entry = manifest.get(split_name)
        digest = entry.get("sha256") if isinstance(entry, Mapping) else None
        count = entry.get("num_samples") if isinstance(entry, Mapping) else None
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(
                f"Missing or invalid publication split digest for '{split_name}'."
            )
        try:
            int(digest, 16)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Missing or invalid publication split provenance for '{split_name}'."
            ) from exc
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise ValueError(
                f"Invalid sample count for publication split '{split_name}': "
                "expected a non-boolean integer."
            )
        if int(count) < 0:
            raise ValueError(
                f"Invalid negative sample count for publication split '{split_name}'."
            )


__all__ = [
    "build_materialized_split_manifest",
    "validate_materialized_split_manifest",
]
