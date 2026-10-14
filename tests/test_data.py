import numpy as np
import pandas as pd
import torch

from repurpose.data.chemistry import canonicalize, fingerprint, scaffold
from repurpose.data.kg_data import edge_group, split_kg
from repurpose.data.tox21 import scaffold_split
from repurpose.toxicity.evaluate import binary_metrics
from repurpose.toxicity.models import masked_bce


def test_smiles_fingerprint_and_stereo():
    assert canonicalize("invalid smiles") is None
    assert canonicalize("CCO") == canonicalize("OCC")
    assert canonicalize("C[C@H](O)Cl") != canonicalize("C[C@@H](O)Cl")
    vector = fingerprint("CCO")
    assert vector.shape == (1024,) and set(vector) <= {0, 1}
    assert np.array_equal(vector, fingerprint("OCC"))


def test_scaffold_salt_tautomer_and_acyclic_groups():
    assert scaffold("c1ccccc1O") == scaffold("c1ccccc1N")
    assert scaffold("CCO") == scaffold("CCCC") == "__ACYCLIC__"
    assert scaffold("c1ccccc1O.[Na+]") == scaffold("c1ccccc1O")
    frame = pd.DataFrame({"scaffold": ["a"] * 30 + ["b"] * 20 + ["c"] * 10 + ["d"] * 10 + ["e"] * 10})
    ids, report = scaffold_split(frame)
    assert report["scaffold_overlap"] == [0, 0, 0]
    assert len(set(np.concatenate(ids))) == len(frame)
    assert all(np.array_equal(a, b) for a, b in zip(ids, scaffold_split(frame)[0]))


def test_kg_inverse_coverage_no_leakage():
    triples = pd.DataFrame(
        [(f"drug:{i}", "indication", f"disease:{j}") for i in range(8) for j in range(8)]
        + [(f"disease:{j}", "reverse_indication", f"drug:{i}") for i in range(8) for j in range(8)],
        columns=["head", "relation", "tail"],
    )
    train, val, test, report = split_kg(triples)
    coverage = set(train["head"]) | set(train["tail"])
    for held in [val, test]:
        assert set(held["head"]) | set(held["tail"]) <= coverage
        assert set(held.relation) <= set(train.relation)
    sets = [{edge_group(*r) for r in p.itertuples(index=False, name=None)} for p in [train, val, test]]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert sum(map(len, [train, val, test])) == len(triples)
    assert report["pair_group_overlap"] == [0, 0, 0]


def test_masked_loss_has_no_gradient_for_missing_labels():
    logits = torch.tensor([[0.4, 0.8]], requires_grad=True)
    y = torch.tensor([[1.0, float("nan")]])
    loss = masked_bce(logits, y)
    loss.backward()
    assert logits.grad[0, 1] == 0
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.nn.functional.binary_cross_entropy_with_logits(logits[0, 0], y[0, 0]))


def test_undefined_metrics_and_missing_labels():
    result = binary_metrics(np.array([1.0, 0.0, np.nan]), np.array([0.9, 0.1, 0.9]), bootstrap=20)
    assert result["n"] == 2 and result["ROC-AUC"] == 1
    assert result["PR-AUC"] == 1
    single = binary_metrics(np.array([1.0, 1.0]), np.array([0.9, 0.8]))
    assert single["status"] == "insufficient_classes" and np.isnan(single["ROC-AUC"])
