"""ZINC15 dataset loader."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

import torch
from torch.utils.data import Dataset
from torch_geometric.utils.smiles import from_smiles

from src.utils import ensure_dir
from src.data_loader.utils import safe_torch_load


class ZINC15Dataset(Dataset):
    """Lazy SMILES-backed loader for the 2M-molecule ZINC15 pretraining subset."""

    url = "https://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip"
    expected_md5 = "9126f2aab5f2cd3fccd307616eec6779"
    member_path = "dataset/zinc_standard_agent/processed/smiles.csv"

    def __init__(self, root: str, transform=None, sample_stats_size: int = 2048, name: str = "zinc15"):
        self.name = str(name).lower()
        self.root = root
        self.transform = transform
        self.sample_stats_size = max(1, int(sample_stats_size))

        self.raw_dir = Path(self.root) / "raw"
        self.processed_dir = Path(self.root) / "processed"
        self.raw_zip_path = self.raw_dir / "chem_dataset.zip"
        self.smiles_path = self.raw_dir / "smiles.csv"
        self.index_path = self.processed_dir / "smiles_index.pt"
        self._file_handle = None

        self._ensure_ready()
        metadata = self._load_or_build_index()
        self.offsets = metadata["offsets"].to(torch.long)
        self.num_graphs = int(metadata["num_graphs"])
        self.avg_nodes_per_graph = float(metadata.get("avg_nodes_per_graph", 0.0))
        self.avg_edges_per_graph = float(metadata.get("avg_edges_per_graph", 0.0))
        self.total_nodes = (
            int(round(self.avg_nodes_per_graph * self.num_graphs))
            if self.avg_nodes_per_graph > 0
            else None
        )
        self.total_edges = (
            int(round(self.avg_edges_per_graph * self.num_graphs))
            if self.avg_edges_per_graph > 0
            else None
        )
        self.num_classes = None
        self.has_labels = False

    def _ensure_ready(self) -> None:
        ensure_dir(str(self.raw_dir))
        ensure_dir(str(self.processed_dir))

        archive_ready = self._has_required_member(self.raw_zip_path)
        if not archive_ready:
            if self.raw_zip_path.is_file():
                self.raw_zip_path.unlink()
            print(f"[Dataset] Downloading {self.name} from {self.url}")
            urlretrieve(self.url, str(self.raw_zip_path))
            if not self._has_required_member(self.raw_zip_path):
                raise ValueError("Downloaded ZINC15 archive is invalid or missing the expected SMILES file.")
            if not self._check_md5(self.raw_zip_path, self.expected_md5):
                if os.environ.get("GAPTUNE_ALLOW_ZINC15_MD5_MISMATCH", "") != "1":
                    raise ValueError(
                        "Downloaded ZINC15 archive failed MD5 verification "
                        f"(expected {self.expected_md5}). If upstream re-published "
                        "the archive intentionally, set "
                        "GAPTUNE_ALLOW_ZINC15_MD5_MISMATCH=1 to accept it."
                    )
                print("[Dataset] WARNING: accepting ZINC15 MD5 mismatch (override set).")
        elif not self._check_md5(self.raw_zip_path, self.expected_md5):
            print(
                "[Dataset] ZINC15 archive MD5 differs from the historical TorchDrug value; "
                "continuing after zip/member validation."
            )

        if not self.smiles_path.is_file():
            print(f"[Dataset] Extracting {self.member_path} -> {self.smiles_path}")
            with ZipFile(self.raw_zip_path, "r") as zf:
                with zf.open(self.member_path) as src, open(self.smiles_path, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

    @staticmethod
    def _check_md5(path: Path, expected_md5: str) -> bool:
        if not path.is_file():
            return False
        digest = hashlib.md5()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest() == expected_md5

    def _has_required_member(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            with ZipFile(path, "r") as zf:
                return self.member_path in zf.namelist()
        except Exception:
            return False

    def _load_or_build_index(self) -> Dict[str, torch.Tensor | float | int]:
        if self.index_path.is_file():
            try:
                payload = safe_torch_load(self.index_path)
                offsets = payload.get("offsets")
                if torch.is_tensor(offsets) and offsets.numel() > 0:
                    return payload
            except Exception:
                pass

        offsets: List[int] = []
        with open(self.smiles_path, "rb") as handle:
            first_offset = handle.tell()
            first_line = handle.readline()
            if not first_line:
                raise ValueError(f"Empty ZINC15 SMILES file: {self.smiles_path}")
            first_text = first_line.decode("utf-8", errors="ignore").strip().split(",", 1)[0].strip().lower()
            if first_text != "smiles":
                offsets.append(first_offset)
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)

        avg_nodes, avg_edges = self._estimate_graph_stats(offsets)
        payload = {
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "num_graphs": int(len(offsets)),
            "avg_nodes_per_graph": float(avg_nodes),
            "avg_edges_per_graph": float(avg_edges),
        }
        torch.save(payload, self.index_path)
        return payload

    def _estimate_graph_stats(self, offsets: List[int]) -> Tuple[float, float]:
        if not offsets:
            return 0.0, 0.0
        sample_count = min(len(offsets), self.sample_stats_size)
        node_total = 0
        edge_total = 0
        with open(self.smiles_path, "rb") as handle:
            for offset in offsets[:sample_count]:
                handle.seek(int(offset))
                line = handle.readline().decode("utf-8", errors="ignore")
                smiles = self._parse_smiles_line(line)
                data = from_smiles(smiles)
                node_total += int(data.num_nodes)
                edge_total += int(data.num_edges)
        return float(node_total) / float(sample_count), float(edge_total) / float(sample_count)

    @staticmethod
    def _parse_smiles_line(line: str) -> str:
        row = next(csv.reader([line.strip()]))
        if not row:
            raise ValueError("Encountered empty SMILES row in ZINC15.")
        return row[0].strip()

    def _get_file_handle(self):
        if self._file_handle is None or self._file_handle.closed:
            self._file_handle = open(self.smiles_path, "rb")
        return self._file_handle

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        total = len(self)
        if idx < 0:
            idx += total
        if idx < 0 or idx >= total:
            raise IndexError(f"ZINC15Dataset index out of range: {idx}")

        handle = self._get_file_handle()
        handle.seek(int(self.offsets[idx].item()))
        line = handle.readline().decode("utf-8", errors="ignore")
        smiles = self._parse_smiles_line(line)
        data = from_smiles(smiles)
        data.smiles = smiles
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_handle"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_file_handle", None)
        if handle is not None and not handle.closed:
            handle.close()
