"""GapTune paper tables in src.results.finetune_tables, from synthetic TSV rows."""

import csv
import json
import re

from src.results import finetune_tables as ft
from src.results import train_tables

F5 = (5.0, 0.0, 1.0)
F100 = (100.0, 0.0, 1.0)
LP_F5 = (0.05, 0.1, 0.1)
ZINC = ("zinc", "edge_pred")


def _ft_row(model, dataset, split, pretrain, method, mean, std=0.01, metric="test_acc", **columns):
    row = {
        "model.name": model,
        "finetune.dataset.name": dataset,
        "finetune.dataset.fixed_split": json.dumps(list(split)),
        "pretrain.dataset.name": pretrain[0],
        "pretrain.method": pretrain[1],
        "finetune.method": method,
        f"{metric}_mean": str(mean),
        f"{metric}_std": str(std),
    }
    row.update({key.replace("__", "."): str(value) for key, value in columns.items()})
    return row


def _gt(dataset, mean, *, split=F5, pretrain=ZINC, model="gcn", plus=True, metric="test_acc", **settings):
    """A gaptune result row; ``settings`` become finetune.gaptune.* columns (``proxy__mode`` -> proxy.mode)."""
    columns = {f"finetune__gaptune__{key}": value for key, value in settings.items()}
    return _ft_row(model, dataset, split, pretrain, "gaptune", mean, metric=metric, finetune__gaptune__plus=plus, **columns)


def _train_row(model, dataset, split, mean, std=0.01, metric="test_acc"):
    return {
        "model.name": model,
        "train.dataset.name": dataset,
        "train.dataset.fixed_split": json.dumps(list(split)),
        f"{metric}_mean": str(mean),
        f"{metric}_std": str(std),
    }


def _write_tsv(path, rows):
    header = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", restval="")
        writer.writeheader()
        writer.writerows(rows)


def _latest(rows):
    return ft._latest_rows(rows)


def _scratch(rows):
    return train_tables._latest_rows(rows)


def _cells(text, label, block_marker=None):
    """Cells of the row labelled ``label`` (optionally inside the block starting at ``block_marker``)."""
    lines = text.splitlines()
    start = 0 if block_marker is None else next(i for i, line in enumerate(lines) if block_marker in line)
    index = next(i for i in range(start, len(lines)) if lines[i] == label)
    body = lines[index + 1].strip()
    assert body.startswith("& ") and body.endswith(" \\\\")
    return body[2:-3].split(" & ")


def _key(dataset, split=F5, *, plus=True, variant="", pretrain=ZINC, model="gcn"):
    return ft.CellKey(model, dataset, split, pretrain[0], pretrain[1], "gaptune", plus, variant)


def test_gaptune_variant_from_explicit_columns():
    # Blank columns predate the setting and explicit defaults (any spelling) are the main cell.
    assert ft._gaptune_variant({"value_mode": "", "num_queries": "8", "tau_c": "0.50", "obs_eps": "1e-6", "lr": "None"}, plus=True) == ""
    assert ft._gaptune_variant({"value_mode": "gap", "proxy.num_graphs": "16.0", "proxy.density": str(4 / 31)}, plus=False) == ""
    assert ft._gaptune_variant({"value_mode": "target", "tau_c": "0.25", "lr": "0.01"}, plus=True) == "lr=0.01,tau_c=0.25,value_mode=target"
    # Proxy settings only identify source-free runs; plus is its own key component.
    assert ft._gaptune_variant({"proxy.num_graphs": "4", "plus": "False"}, plus=True) == ""
    assert ft._gaptune_variant({"proxy.num_graphs": "4", "proxy.mode": "random"}, plus=False) == "proxy.mode=random,proxy.num_graphs=4"
    # Ablation arm specs resolve to the same identity as their result rows.
    arm = next(arm for arm in ft.ABLATION_TABLES[3].arms if arm.label == "Random proxies ($B=4$)")
    row = _gt("photo", 0.7, plus=False, proxy__mode="random", proxy__num_graphs=4, value_mode="gap")
    assert ft._cell_key_from_result_row(row) == _key("photo", plus=False, variant=arm.variant)


def test_ablation_rows_never_overwrite_main_gaptune_cells():
    rows = [
        _gt("photo", 0.70),
        _gt("photo", 0.60, value_mode="target"),
        _gt("photo", 0.50, prompt_locations="none"),
        _gt("photo", 0.40, plus=False, proxy__mode="random"),
    ]
    latest = _latest(rows)
    assert latest[_key("photo")]["test_acc_mean"] == "0.7"
    assert latest[_key("photo", variant="value_mode=target")]["test_acc_mean"] == "0.6"
    assert latest[_key("photo", variant="prompt_locations=none")]["test_acc_mean"] == "0.5"
    assert latest[_key("photo", plus=False, variant="proxy.mode=random")]["test_acc_mean"] == "0.4"
    # A later main run with explicit default columns replaces only the main cell.
    latest = _latest(rows + [_gt("photo", 0.75, value_mode="gap", num_queries=8)])
    assert latest[_key("photo")]["test_acc_mean"] == "0.75"
    assert latest[_key("photo", variant="value_mode=target")]["test_acc_mean"] == "0.6"
    # Backfill presence: an ablation row does not mark the main cell as present.
    assert _key("photo") not in ft._valid_result_cells(rows[1:], models={"gcn"})


def test_appended_result_rows_keep_ablation_identity(tmp_path):
    # Real CLI cfg -> finetune.tsv append -> cell key: explicit overrides become the variant columns.
    from datetime import datetime

    from src.finetune.run import build_finetune_cfg
    from src.utils.save_results import append_workflow_result

    base = [
        "finetune.dataset.name", "photo", "finetune.dataset.task_level", "node", "finetune.method", "gaptune",
        "model.name", "gcn", "pretrain.dataset.name", "zinc", "pretrain.method", "edge_pred",
        "save_results.output_dir", str(tmp_path), "--fewshot", "5", "0.0", "1.0",
    ]
    for extra, mean in (([], 0.7), (["finetune.gaptune.value_mode", "target"], 0.6), ([], 0.75)):
        cfg = build_finetune_cfg(base + extra)
        cfg.save_results.enabled = True
        now = datetime.now()
        append_workflow_result(
            cfg=cfg, workflow="finetune", started_at=now, ended_at=now, checkpoint_save_paths=[], seeds=[42],
            best_epochs=[1], metric_summary={"test_acc": {"mean": mean, "std": 0.01}},
        )
    rows = ft._read_result_rows(tmp_path / "finetune.tsv")
    assert [row["finetune.gaptune.value_mode"] for row in rows] == ["", "target", "gap"]
    latest = _latest(rows)
    assert float(latest[_key("photo")]["test_acc_mean"]) == 0.75
    assert float(latest[_key("photo", variant="value_mode=target")]["test_acc_mean"]) == 0.6


def test_backfill_expects_cross_dataset_cells_for_matching_backbone():
    gcn = ft._expected_cells_for_model("gcn")
    gin = ft._expected_cells_for_model("gin")
    zinc_gt = ft.CellKey("gcn", "photo", F5, "zinc", "edge_pred", "gaptune", True)
    pubmed_gt = ft.CellKey("gin", "mnist", F100, "pubmed", "graphcl", "gaptune", False)
    assert zinc_gt in gcn and zinc_gt not in gin
    assert pubmed_gt in gin
    assert ft.CellKey("gcn", "photo", F5, "photo", "dgi", "supt", None) in gcn


def test_cross_dataset_table_blocks_scratch_and_ranking():
    pubmed = ("pubmed", "graphcl")
    latest = _latest(
        [
            _ft_row("gcn", "photo", F5, ZINC, "gpf", 0.55, finetune__gpf__plus=False),
            _ft_row("gcn", "photo", F5, ZINC, "igap", 0.7084),
            _gt("photo", 0.7512, plus=False),
            _gt("photo", 0.7786),
            _gt("photo", 0.95, value_mode="free"),  # ablation arm: never in the main table
            _ft_row("gcn", "qm7b", F5, ZINC, "mtg", 10.72, 1.42, metric="test_mae"),
            _gt("qm7b", 5.42, metric="test_mae"),
            _ft_row("gin", "photo", F5, pubmed, "mtg", 0.6791),
            _ft_row("gin", "photo", F5, ("photo", "graphcl"), "mtg", 0.99),  # same-dataset row
        ]
    )
    scratch = _scratch(
        [
            _train_row("gcn", "photo", F5, 0.7708),
            _train_row("gcn", "qm7b", F5, 5.59, 0.28, metric="test_mae"),
            _train_row("gin", "photo", F5, 0.7184),
            _train_row("gcn", "photo", F100, 0.99),  # other shot
        ]
    )
    text = ft._render_cross_dataset_table(shot="f5", latest=latest, scratch=scratch)
    zinc = "Source dataset: ZINC, GNN: GCN, Pretraining: EdgePred"
    pub = "Source dataset: PubMed, GNN: GIN, Pretraining: GraphCL"
    assert zinc in text and pub in text
    for absent in ("95.00", "99.00"):
        assert absent not in text

    # Row order: scratch, the twelve baselines, GapTune, GapTune Plus.
    lines = text.splitlines()
    block = lines[lines.index(next(line for line in lines if zinc in line)):]
    labels = [spec.label for spec in (*ft.BASELINE_METHODS, *ft.GAPTUNE_METHODS)]
    order = [block.index(label) for label in ["GCN (scratch)", *labels]]
    assert order == sorted(order) and len(labels) == 14
    assert {"IGAP", "MTG", "SUPT", "GapTune", "GapTune Plus"} <= set(labels)

    photo, qm7b = 0, 6
    assert _cells(text, "GapTune Plus", zinc)[photo] == "\\textbf{77.86}$_{\\pm1.00}$"
    assert _cells(text, "GCN (scratch)", zinc)[photo] == "\\underline{77.08}$_{\\pm1.00}$"
    assert _cells(text, "GapTune", zinc)[photo] == "75.12$_{\\pm1.00}$"
    assert _cells(text, "GPF", zinc)[photo] == "55.00$_{\\pm1.00}$"
    assert _cells(text, "SUPT", zinc)[photo] == "--"
    # MAE: lower is better, reported unscaled.
    assert _cells(text, "GapTune Plus", zinc)[qm7b] == "\\textbf{5.42}$_{\\pm0.01}$"
    assert _cells(text, "GCN (scratch)", zinc)[qm7b] == "\\underline{5.59}$_{\\pm0.28}$"
    assert _cells(text, "MTG", zinc)[qm7b] == "10.72$_{\\pm1.42}$"
    # Ranking is within a block: PubMed's MTG is best there despite lower means than ZINC.
    assert _cells(text, "MTG", pub)[photo] == "\\underline{67.91}$_{\\pm1.00}$"
    assert _cells(text, "GIN (scratch)", pub)[photo] == "\\textbf{71.84}$_{\\pm1.00}$"


def test_same_dataset_table_scratch_controls_and_best_grid():
    latest = _latest(
        [
            _gt("photo", 0.80, pretrain=("photo", "dgi")),
            _gt("photo", 0.83, pretrain=("photo", "graphcl"), model="gat"),
            _gt("photo", 0.95, pretrain=("photo", "dgi"), mixture="uniform"),  # ablation arm
            _gt("photo", 0.99),  # cross-dataset row
            _gt("photo", 0.97, plus=False, pretrain=("photo", "edge_pred")),  # source-free GapTune: not a Table 2 row
            _ft_row("gcn", "photo", F5, ("photo", "dgi"), "supt", 0.82),
            {**_ft_row("gcn", "photo", F5, ("photo", "dgi"), "igap", -1, -1), "result_status": "OOM"},
            _ft_row("gcn", "dblp", LP_F5, ("dblp", "dgi"), "mtg", 0.5612, metric="test_auc"),
        ]
    )
    scratch = _scratch([_train_row("gcn", "photo", F5, 0.7708), _train_row("nodeformer", "photo", F5, 0.7904)])
    models = list(ft.MODEL_LABELS)
    text = ft._render_same_dataset_table(models=models, shot="f5", latest=latest, scratch=scratch)

    labels = text.splitlines()
    for model in train_tables.MODELS:
        assert model.label in labels
    assert "GapTune" not in labels and "GapTune Plus" in labels
    for absent in ("95.00", "97.00", "99.00"):
        assert absent not in text

    photo, dblp = 0, 2
    assert _cells(text, "GapTune Plus")[photo] == "\\textbf{83.00}$_{\\pm1.00}$"
    assert _cells(text, "SUPT")[photo] == "\\underline{82.00}$_{\\pm1.00}$"
    assert _cells(text, "NodeFormer")[photo] == "79.04$_{\\pm1.00}$"
    assert _cells(text, "IGAP")[photo] == "OOM"
    assert _cells(text, "MTG")[photo] == "--"
    assert _cells(text, "MTG")[dblp] == "\\textbf{56.12}$_{\\pm1.00}$"


def test_ablation_tables_map_arms_to_their_cells():
    latest = _latest(
        [
            _gt("photo", 0.7786),
            _gt("photo", 0.6473, prompt_locations="none"),
            _gt("dblp", 0.5252, split=LP_F5, metric="test_auc", prompt_locations="none"),
            _gt("photo", 0.7244, value_mode="target"),
            _gt("photo", 0.7463, value_mode="free"),
            _gt("photo", 0.7135, value_mode="free", prompt_locations="node"),
            _gt("photo", 0.99, pretrain=("photo", "edge_pred"), value_mode="target"),  # same-dataset row
            _gt("photo", 0.8748, split=F100),
            _gt("photo", 0.8643, split=F100, query_mode="frozen"),
            _gt("photo", 0.7512, plus=False),
            _gt("photo", 0.7158, plus=False, proxy__mode="random", proxy__num_graphs=4),
            _gt("photo", 0.7664, plus=False, proxy__num_graphs=64),
        ]
    )
    tables = {table.name: ft._render_ablation_table(table=table, latest=latest) for table in ft.ABLATION_TABLES}
    assert set(tables) == {"value", "insertion", "composition", "source"}
    for text in tables.values():
        assert "99.00" not in text

    value = tables["value"]
    assert _cells(value, "Head only") == ["64.73$_{\\pm1.00}$", "--", "\\textbf{52.52}$_{\\pm1.00}$", "--"]
    assert _cells(value, "GapTune Plus")[0] == "\\textbf{77.86}$_{\\pm1.00}$"
    assert _cells(value, "Free vectors")[0] == "\\underline{74.63}$_{\\pm1.00}$"
    assert _cells(value, "Target")[0] == "72.44$_{\\pm1.00}$"
    assert _cells(value, "Source")[0] == "--"

    insertion = tables["insertion"]
    assert _cells(insertion, "Free values: node only")[0] == "71.35$_{\\pm1.00}$"
    assert _cells(insertion, "Free values: node + message")[0] == "\\underline{74.63}$_{\\pm1.00}$"
    assert _cells(insertion, "GapTune Plus: node + message")[0] == "\\textbf{77.86}$_{\\pm1.00}$"

    composition = tables["composition"]  # columns: Photo 5, Photo 100, Chameleon 5, Chameleon 100
    assert _cells(composition, "GapTune Plus")[:2] == ["\\textbf{77.86}$_{\\pm1.00}$", "\\textbf{87.48}$_{\\pm1.00}$"]
    assert _cells(composition, "Frozen shared queries")[:2] == ["--", "\\underline{86.43}$_{\\pm1.00}$"]

    source = tables["source"]
    assert _cells(source, "Random proxies ($B=4$)")[0] == "71.58$_{\\pm1.00}$"
    assert _cells(source, "Random proxies ($B=16$)")[0] == "--"
    assert _cells(source, "GapTune ($B=16$)")[0] == "75.12$_{\\pm1.00}$"
    assert _cells(source, "Inverted proxies ($B=64$)")[0] == "\\underline{76.64}$_{\\pm1.00}$"
    assert _cells(source, "GapTune Plus (source available)")[0] == "\\textbf{77.86}$_{\\pm1.00}$"


#: Paper Tables 3-6 arms as the finetune.gaptune.* columns of their runs, written independently of ft.ABLATION_TABLES.
_PAPER_ARMS = {
    "Head only": {"prompt_locations": "none"},
    "Target": {"value_mode": "target"},
    "Source": {"value_mode": "source"},
    "Paired mean": {"value_mode": "paired_mean"},
    "Free vectors": {"value_mode": "free"},
    "GapTune Plus": {},
    "Free values: node only": {"value_mode": "free", "prompt_locations": "node"},
    "Free values: message only": {"value_mode": "free", "prompt_locations": "message"},
    "Free values: node + message": {"value_mode": "free"},
    "Gap values: node only": {"prompt_locations": "node"},
    "Gap values: message only": {"prompt_locations": "message"},
    "GapTune Plus: node + message": {},
    "Frozen shared queries": {"query_mode": "frozen"},
    "Untied source/target queries": {"query_mode": "untied"},
    "Uniform mixture weights": {"mixture": "uniform"},
    "Learned global mixture weights": {"mixture": "global"},
    "Nonnegative gates": {"gate": "nonnegative"},
    "Random proxies ($B=4$)": {"plus": False, "proxy__mode": "random", "proxy__num_graphs": 4},
    "Random proxies ($B=16$)": {"plus": False, "proxy__mode": "random", "proxy__num_graphs": 16},
    "Random proxies ($B=64$)": {"plus": False, "proxy__mode": "random", "proxy__num_graphs": 64},
    "Inverted proxies ($B=4$)": {"plus": False, "proxy__mode": "inverted", "proxy__num_graphs": 4},
    "GapTune ($B=16$)": {"plus": False, "proxy__mode": "inverted", "proxy__num_graphs": 16},
    "Inverted proxies ($B=64$)": {"plus": False, "proxy__mode": "inverted", "proxy__num_graphs": 64},
    "GapTune Plus (source available)": {},
}


def test_every_ablation_arm_reads_its_own_result_row():
    # One row per column and distinct settings (head only, free node + message and GapTune+ share theirs), so a
    # mis-specified arm shows another arm's mean or "--".
    assert {arm.label for table in ft.ABLATION_TABLES for arm in table.arms} == set(_PAPER_ARMS)
    settings = {label: tuple(sorted(columns.items())) for label, columns in _PAPER_ARMS.items()}
    means, rows = {}, []
    for table in ft.ABLATION_TABLES:
        for name, shot in table.columns:
            dataset = next(spec for spec in ft.DATASETS if spec.name == name)
            split = dataset.f5_split if shot == "f5" else dataset.f100_split
            for arm in table.arms:
                key = (name, shot, settings[arm.label])
                if key not in means:
                    means[key] = 0.2 + len(means) / 1000
                    rows.append(_gt(name, means[key], split=split, metric=dataset.metric, **_PAPER_ARMS[arm.label]))
    latest = _latest(rows)
    for table in ft.ABLATION_TABLES:
        text = ft._render_ablation_table(table=table, latest=latest)
        for arm in table.arms:
            shown = [re.sub(r"\\(?:textbf|underline)\{(.*?)\}", r"\1", cell) for cell in _cells(text, arm.label)]
            expected = [f"{100 * means[(name, shot, settings[arm.label])]:.2f}$_{{\\pm1.00}}$" for name, shot in table.columns]
            assert shown == expected, (table.name, arm.label)


def test_ranking_ties_share_rank_and_skip_oom():
    rows = [("A", [(0.5, 0.01)]), ("B", [(0.50001, 0.02)]), ("C", ["OOM"]), ("D", [(0.4, 0.01)]), ("E", [None])]
    lines = ft._ranked_group_lines([("", rows)], ["test_acc"])
    body = dict(zip(lines[1::2], lines[2::2]))
    assert body["A"] == "    & \\textbf{50.00}$_{\\pm1.00}$ \\\\"
    assert body["B"] == "    & \\textbf{50.00}$_{\\pm2.00}$ \\\\"
    assert body["C"] == "    & OOM \\\\"
    assert body["D"] == "    & \\underline{40.00}$_{\\pm1.00}$ \\\\"
    assert body["E"] == "    & -- \\\\"


def test_render_tables_writes_paper_tables(tmp_path):
    finetune_tsv = tmp_path / "finetune.tsv"
    train_tsv = tmp_path / "train.tsv"
    out = tmp_path / "out"
    _write_tsv(
        finetune_tsv,
        [
            _gt("photo", 0.80, pretrain=("photo", "dgi")),
            _gt("photo", 0.95, pretrain=("photo", "dgi"), gate="nonnegative"),
            _gt("photo", 0.7786),
        ],
    )
    _write_tsv(train_tsv, [_train_row("gcn", "photo", F5, 0.7708)])
    assert ft.main(
        [
            "--no-backfill",
            "--models", "gcn",
            "--results-tsv", str(finetune_tsv),
            "--train-results-tsv", str(train_tsv),
            "--results-dir", str(out),
        ]
    ) == 0
    for name in ("cross_f5", "cross_f100", "same_f5", "same_f100", "ablation_value", "ablation_insertion",
                 "ablation_composition", "ablation_source", "gcn_f5", "gaptune_plus_f5"):
        assert (out / f"finetune_{name}_table.tex").is_file(), name
    # The existing same-dataset grids keep the main GapTune+ cell; the ablation arm has its own.
    grid = (out / "finetune_gcn_f5_table.tex").read_text(encoding="utf-8")
    assert "80.00$_{\\pm1.00}$" in grid and "95.00" not in grid
    cross = (out / "finetune_cross_f5_table.tex").read_text(encoding="utf-8")
    assert "\\textbf{77.86}$_{\\pm1.00}$" in cross and "\\underline{77.08}$_{\\pm1.00}$" in cross
    # Paper Tables 1/2/14/15 label DBLP Academic and Cornell Web.
    same = (out / "finetune_same_f5_table.tex").read_text(encoding="utf-8")
    for text in (cross, same):
        assert "& \\textbf{E-commerce} & \\textbf{Academic} & \\textbf{Academic}\n& \\textbf{Transport} & \\textbf{Web} & \\textbf{Web}\n" in text
