"""App. C.10 cost study on the synthetic workload (no dataset loading).

- parameter census of Table 12 (1,032 head; 46,984 full fine-tuning; 3,664 /
  12,896 / 15,528 free values; 15,528 GapTune(+)) and 3,872 retained scalars;
- workload shape: 64 simple graphs, 8,192 nodes, 65,536 GCN messages per layer;
- only full fine-tuning trains the encoder;
- a shortened CPU run writes the per-block and summary tables.
"""

from __future__ import annotations

import copy
import csv

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import is_undirected

from src.analysis import cost
from src.analysis import run as analysis_run
from src.config import cfg as base_cfg
from src.finetune.encoders.gaptune import get_gaptune_driver


def test_cost_cfg_defaults():
    assert base_cfg.analysis.cost.timing_blocks == 5
    assert base_cfg.analysis.cost.updates == 500
    assert base_cfg.analysis.cost.evaluations == 20
    assert analysis_run.STUDIES["cost"] is cost.run_cost_study


def test_parameter_census_matches_table_12():
    census = cost.parameter_census(base_cfg)
    assert {name: row["trainable"] for name, row in census.items()} == {
        "head_only": 1_032,
        "full_finetuning": 46_984,
        "free_node": 3_664,
        "free_message": 12_896,
        "free_node_message": 15_528,
        "gaptune": 15_528,
        "gaptune_plus": 15_528,
    }
    assert census["gaptune"]["retained_scalars"] == census["gaptune_plus"]["retained_scalars"] == 3_872
    assert census["head_only"]["retained_scalars"] == census["full_finetuning"]["retained_scalars"] == 0


def test_synthetic_workload_shape():
    wcfg = cost.workload_cfg(base_cfg)
    graphs = cost.synthetic_graphs(cost.NUM_GRAPHS, torch.Generator().manual_seed(0))
    for graph in graphs[:4]:
        u, v = graph.edge_index
        assert is_undirected(graph.edge_index) and not (u == v).any()
        assert torch.unique(u * cost.GRAPH_NODES + v).numel() == 2 * cost.GRAPH_EDGES
    batch = Batch.from_data_list(graphs)
    assert batch.num_graphs == 64 and batch.num_nodes == 8_192
    assert batch.target_node_index.tolist() == [128 * g for g in range(64)]
    encoder = cost.workload_encoder(wcfg)
    assert sum(p.numel() for p in encoder.convs.parameters()) == 45_952
    obs = get_gaptune_driver(encoder).forward(encoder, batch, collect=True).obs
    assert [layer.messages.size(0) for layer in obs.layers] == [65_536] * 3


def test_only_full_finetuning_trains_the_encoder():
    wcfg = cost.workload_cfg(base_cfg)
    encoder = cost.workload_encoder(wcfg)
    batch = Batch.from_data_list(cost.synthetic_graphs(2, torch.Generator().manual_seed(0)))
    census = cost.parameter_census(base_cfg)
    for name in ("head_only", "full_finetuning", "free_node_message"):
        model = copy.deepcopy(encoder)
        task, params = cost.build_method(wcfg, name, model)
        if task.prompt.types:
            cost.prepare_source(task, model, "source", cost.synthetic_graphs(2, torch.Generator()), torch.device("cpu"))
        task.train()
        loss, _ = task._forward(model, batch, torch.device("cpu"))
        loss.backward()
        encoder_grads = [p.grad is not None for p in model.convs.parameters()]
        assert all(encoder_grads) if name == "full_finetuning" else not any(encoder_grads)
        assert all(p.grad is not None for p in task.head.parameters())
        assert sum(p.numel() for p in params) == census[name]["trainable"]


def test_shortened_run_writes_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(cost, "NUM_GRAPHS", 2)
    status = analysis_run.run_analysis_from_cli(
        [
            "analysis.study", "cost",
            "analysis.output_dir", str(tmp_path),
            "analysis.cost.timing_blocks", "2",
            "analysis.cost.updates", "2",
            "analysis.cost.evaluations", "1",
            "finetune.gaptune.proxy.updates", "1",
        ]
    )
    assert status == 0
    out_dir = tmp_path / "cost" / "seed42-b2-u2-e1-pxu1"

    def read(name):
        with open(out_dir / name, encoding="utf-8") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    assert len(read("blocks.tsv")) == 2 * len(cost.METHODS)
    summary = {row["method"]: row for row in read("summary.tsv")}
    assert list(summary) == list(cost.METHODS)
    for name, row in summary.items():
        assert int(row["trainable"]) > 0
        assert (float(row["prep_s_mean"]) > 0) == (name in ("gaptune", "gaptune_plus"))
        assert float(row["adapt_s_mean"]) > 0 and float(row["infer_ms_mean"]) > 0
    assert (out_dir / "cost.json").is_file()


def test_invalid_schedule_is_rejected(capsys):
    cfg = base_cfg.clone()
    cfg.analysis.cost.updates = 25
    cfg.analysis.cost.evaluations = 20
    assert cost.run_cost_study(cfg) == 1
    assert "multiple of evaluations" in capsys.readouterr().err
