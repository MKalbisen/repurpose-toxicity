import numpy as np
import pandas as pd
import pytest

from repurpose.academic.prepare import chemical_identity, component_groups, group_split, pair_partition


def test_transitive_identity_closure():
    frame = pd.DataFrame(
        {
            "canonical_smiles": ["a", "b", "c", "d"],
            "parent": ["p1", "p2", "p3", "p4"],
            "tautomer": ["t1", "t2", "t3", "t4"],
            "scaffold": ["s1", "s1", "s2", "s3"],
            "fingerprint": ["f1", "f2", "f2", "f4"],
        }
    )
    groups = component_groups(frame)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
    reordered = frame.iloc[[3, 1, 0, 2]].reset_index(drop=True)
    assert dict(zip(frame.canonical_smiles, groups)) == dict(
        zip(reordered.canonical_smiles, component_groups(reordered))
    )


def test_groups_never_cross_any_partition_and_repeat_exactly():
    groups = np.repeat(np.arange(100).astype(str), np.arange(1, 101))
    splits = group_split(groups)
    assert sorted(np.concatenate(splits)) == list(range(len(groups)))
    for i in range(4):
        assert np.array_equal(splits[i], group_split(groups)[i])
        for j in range(i):
            assert not set(groups[splits[i]]) & set(groups[splits[j]])
    with pytest.raises(ValueError, match="independent"):
        group_split(["same"] * 20)


def test_stereo_and_salt_kept_in_model_input_but_grouped():
    a = chemical_identity("C[C@H](O)c1ccccc1")
    b = chemical_identity("C[C@@H](O)c1ccccc1")
    assert a["status"] == b["status"] == "accepted"
    assert a["canonical_smiles"] != b["canonical_smiles"]
    assert a["parent"] == b["parent"] and a["fingerprint"] == b["fingerprint"]
    assert chemical_identity("invalid!!!")["status"] == "invalid_smiles"
    assert pair_partition("a\x1fb", 2026) == pair_partition("a\x1fb", 2026)


def test_incomplete_enumeration_is_quarantined(monkeypatch):
    from repurpose.academic import prepare

    monkeypatch.setitem(prepare.POLICY, "max_tautomers", 1)
    row = chemical_identity("CC(=O)CC(=O)C")
    assert row["status"] == "incomplete_enumeration"
    assert "scaffold" not in row
