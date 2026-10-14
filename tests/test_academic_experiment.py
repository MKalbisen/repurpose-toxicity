import json

import numpy as np
import pandas as pd
import pytest
import torch

from repurpose.academic import protocol
from repurpose.academic.kg import negative_batch, triple_codes
from repurpose.academic.prepare import digest, prepare_graph
from repurpose.academic.state import atomic_json
from repurpose.academic.toxicity import train_mlp
from repurpose.utils import sha256


def test_exact_training_negative_filter():
    positive = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 0]])
    known = torch.sort(triple_codes(positive, 3, 2))[0]
    negatives = negative_batch(positive, 100, 3, 2, known)
    assert len(negatives) > 0
    assert not set(triple_codes(negatives, 3, 2).tolist()) & set(known.tolist())
    assert (negatives[:, 1] < 2).all()


def test_graph_order_independence_and_actual_coverage(tmp_path):
    rows = []
    for i in range(20):
        for j in range(12):
            rows.append(["indication", str(i), "drug", f"Drug {i}", str(j), "disease", f"Disease {j}"])
            rows.append(["indication", str(j), "disease", f"Disease {j}", str(i), "drug", f"Drug {i}"])
    raw = pd.DataFrame(rows, columns=["relation", "x_id", "x_type", "x_name", "y_id", "y_type", "y_name"])
    for name, frame in [("a", raw), ("b", raw.sample(frac=1, random_state=7))]:
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
        manifest = prepare_graph(tmp_path / name, tmp_path / f"{name}.csv")
        assert sum(manifest["sizes"].values()) == 240
        assert not any(manifest["checks"].values())
    for split in ["train", "validation", "test"]:
        assert (tmp_path / "a" / f"{split}.tsv").read_bytes() == (
            tmp_path / "b" / f"{split}.tsv"
        ).read_bytes()


def test_protocol_changes_and_changed_models_block_test(tmp_path, monkeypatch):
    snapshot = {"config": {"seed": 17}, "data": "hash", "code": "hash"}
    monkeypatch.setattr(protocol, "snapshot", lambda config: snapshot)
    atomic_json(tmp_path / "protocol_lock.json", {"snapshot": snapshot, "sha256": digest(snapshot)})
    model = tmp_path / "model.bin"
    model.write_bytes(b"selected on development")
    first = protocol.final_access({}, tmp_path, "toxicity", [model])
    assert protocol.final_access({}, tmp_path, "toxicity", [model]) == first
    model.write_bytes(b"reselected after test")
    with pytest.raises(ValueError, match="changed"):
        protocol.final_access({}, tmp_path, "toxicity", [model])
    snapshot["code"] = "modified"
    with pytest.raises(ValueError, match="changed"):
        protocol.verify_lock({}, tmp_path)


def test_mlp_epoch_resume_matches_uninterrupted_cpu(tmp_path):
    rng = np.random.default_rng(1)
    data = {"X": rng.integers(0, 2, (40, 1024)), "Y": rng.integers(0, 2, (40, 12)).astype(float)}
    data["Y"][0, 0] = np.nan
    config = {
        "device": "cpu",
        "threads": 1,
        "toxicity": {"mlp_epochs": 2, "patience": 5, "batch_size": 16, "learning_rate": 0.001},
    }
    direct, resumed = tmp_path / "direct", tmp_path / "resumed"
    direct.mkdir()
    resumed.mkdir()
    train_mlp(config, data, data, 17, direct)
    train_mlp({**config, "toxicity": {**config["toxicity"], "mlp_epochs": 1}}, data, data, 17, resumed)
    train_mlp(config, data, data, 17, resumed)
    a = torch.load(direct / "mlp_resume.pt", weights_only=False)
    b = torch.load(resumed / "mlp_resume.pt", weights_only=False)
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
    assert a["history"] == b["history"]


def test_training_never_requests_test_partition(tmp_path, monkeypatch):
    from repurpose.academic import toxicity

    rng = np.random.default_rng(5)
    data = {"X": rng.integers(0, 2, (40, 1024)), "Y": rng.integers(0, 2, (40, 12)).astype(float)}
    requested = []
    test_allowed = []

    def load(config, split):
        assert split != "test" or test_allowed, "Training attempted to access final test"
        requested.append(split)
        return {**data, "groups": np.repeat(np.arange(10).astype(str), 4), "rows": np.arange(40)}

    monkeypatch.setattr(toxicity, "verify_lock", lambda *args: None)
    monkeypatch.setattr(toxicity, "load_partition", load)
    config = {
        "device": "cpu",
        "threads": 1,
        "seeds": [17],
        "toxicity": {
            "mlp_epochs": 1,
            "patience": 1,
            "batch_size": 16,
            "learning_rate": 0.001,
            "logistic_C": [1],
            "xgb_estimators": [2],
            "xgb_depth": 2,
        },
    }
    toxicity.train_toxicity(config, tmp_path)
    assert requested == ["train", "validation", "calibration"]
    complete = json.loads((tmp_path / "toxicity/17/complete.json").read_text())
    assert not complete["test_accessed"]
    for name, expected in complete["files"].items():
        assert sha256(tmp_path / "toxicity/17" / name) == expected
    monkeypatch.setattr(toxicity, "final_access", lambda *args: test_allowed.append(True))
    result = toxicity.evaluate_toxicity({**config, "uncertainty": {"bootstrap_repeats": 4}}, tmp_path)
    assert len(result) == 4 and requested[-1] == "test"
    assert (tmp_path / "evaluation/toxicity/macro_cluster_uncertainty.json").exists()


def test_kg_training_and_separate_final_evaluator(tmp_path, monkeypatch):
    from repurpose.academic import kg

    rows = [
        ["indication", str(i), "drug", f"D{i}", str(j), "disease", f"Q{j}"]
        for i in range(15)
        for j in range(12)
    ]
    raw = tmp_path / "raw.csv"
    pd.DataFrame(rows, columns=["relation", "x_id", "x_type", "x_name", "y_id", "y_type", "y_name"]).to_csv(
        raw, index=False
    )
    base = tmp_path / "data/processed/tiny/kg"
    prepare_graph(base, raw)
    monkeypatch.setattr(kg, "ROOT", tmp_path)
    monkeypatch.setattr(kg, "verify_lock", lambda *args: None)
    accessed = []
    monkeypatch.setattr(kg, "final_access", lambda *args: accessed.append("final_gate"))
    original = kg.read_targets

    def guarded(base, split):
        assert split != "test" or accessed
        return original(base, split)

    monkeypatch.setattr(kg, "read_targets", guarded)
    config = {
        "dataset": "tiny",
        "device": "cpu",
        "threads": 1,
        "seeds": [17],
        "split_seed": 2026,
        "kg": {
            "dimensions": [8],
            "learning_rates": [0.001],
            "epochs": 2,
            "patience": 2,
            "validation_every": 1,
            "batch_size": 64,
            "negatives": 2,
        },
        "uncertainty": {"bootstrap_repeats": 10},
    }
    kg.train_kg(config, tmp_path / "run")
    assert not accessed
    result = kg.evaluate_kg(config, tmp_path / "run")
    assert len(result) == 4 and accessed
    assert {r["family"] for r in result} == {"TransE", "RotatE", "popularity", "shared_genes"}
