import time

import numpy as np
import pandas as pd

from repurpose.data.chemistry import canonicalize, fingerprint, metadata, scaffold
from repurpose.utils import ROOT, TASKS, paths, write_json


def scaffold_split(frame, seed=42):
    groups = frame.groupby("scaffold", sort=True).indices
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    # Largest indivisible scaffolds first. Acyclic molecules stay one group.
    keys.sort(key=lambda k: -len(groups[k]))
    buckets = [[], [], []]
    targets = np.array([0.8, 0.1, 0.1]) * len(frame)
    for key in keys:
        ids = list(groups[key])
        deficits = targets - np.array(list(map(len, buckets)))
        slot = int(np.argmax(deficits))
        buckets[slot].extend(ids)
    sets = [set(frame.iloc[b].scaffold) for b in buckets]
    overlap = [len(sets[i] & sets[j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
    if any(overlap) or any(not b for b in buckets):
        raise ValueError("Geçerli üçlü scaffold split üretilemedi")
    return [np.array(sorted(b), dtype=int) for b in buckets], {
        "seed": seed,
        "sizes": dict(zip(["train", "validation", "test"], map(len, buckets))),
        "scaffold_overlap": overlap,
        "requested_ratios": [0.8, 0.1, 0.1],
        "actual_ratios": [len(b) / len(frame) for b in buckets],
        "acyclic_policy": "all empty scaffolds stay together",
        "group_count": len(groups),
    }


def prepare_tox21(config):
    base, _, reports = paths(config)
    start = time.perf_counter()
    raw = pd.read_csv(ROOT / "data/raw/tox21/tox21.csv.gz")
    if not set(TASKS + ["smiles"]) <= set(raw):
        raise ValueError("Tox21: 12 assay ve smiles alanları zorunlu")
    for task in TASKS:
        if not set(raw[task].dropna().unique()) <= {0, 1}:
            raise ValueError(f"Geçersiz ikili etiket: {task}")
    raw["canonical_smiles"] = raw.smiles.map(canonicalize)
    raw[raw.canonical_smiles.isna()].to_csv(base / "invalid_smiles.csv", index=False)
    valid = raw.dropna(subset=["canonical_smiles"]).copy()
    clean, conflicts = [], []
    for key, group in valid.groupby("canonical_smiles", sort=True):
        row = {"smiles": key, "canonical_smiles": key, "source_count": len(group)}
        for task in TASKS:
            labels = group[task].dropna().unique()
            row[task] = labels[0] if len(labels) == 1 else np.nan
            if len(labels) > 1:
                conflicts.append({"smiles": key, "assay": task, "reason": "conflicting duplicate labels"})
        clean.append(row)
    frame = pd.DataFrame(clean)
    failures = []

    def safe_scaffold(smiles):
        try:
            return scaffold(smiles)
        except (ValueError, RuntimeError) as error:
            failures.append({"smiles": smiles, "reason": str(error)})
            return None

    frame["scaffold"] = frame.smiles.map(safe_scaffold)
    pd.DataFrame(failures, columns=["smiles", "reason"]).to_csv(base / "invalid_scaffolds.csv", index=False)
    frame = frame.dropna(subset=["scaffold"]).reset_index(drop=True)
    frame.to_csv(base / "tox21_clean.csv", index=False)
    pd.DataFrame(conflicts, columns=["smiles", "assay", "reason"]).to_csv(
        base / "label_conflicts.csv", index=False
    )
    features = np.stack(frame.smiles.map(fingerprint))
    labels = frame[TASKS].to_numpy(dtype=np.float32)
    indices, split_report = scaffold_split(frame, config["seed"])
    for name, ids in zip(["train", "val", "test"], indices):
        frame.iloc[ids].to_csv(base / f"tox21_{name}.csv", index=False)
    np.savez_compressed(
        base / "tox21_features.npz", X=features, Y=labels, train=indices[0], val=indices[1], test=indices[2]
    )
    label_report = []
    for name, ids in [("all", np.arange(len(frame))), *zip(["train", "val", "test"], indices)]:
        for task in TASKS:
            y = frame.iloc[ids][task]
            label_report.append(
                {
                    "split": name,
                    "assay": task,
                    "positive": int((y == 1).sum()),
                    "negative": int((y == 0).sum()),
                    "missing": int(y.isna().sum()),
                }
            )
    pd.DataFrame(label_report).to_csv(reports / "tox21_label_report.csv", index=False)
    write_json(reports / "scaffold_split_report.json", split_report)
    write_json(
        reports / "fingerprint_metadata.json",
        {
            **metadata(),
            "shape": list(features.shape),
            "seconds": time.perf_counter() - start,
            "raw_rows": len(raw),
            "clean_rows": len(frame),
            "invalid_rows": int(raw.canonical_smiles.isna().sum()),
            "collapsed_duplicates": len(valid) - len(clean),
            "invalid_scaffold_rows": len(failures),
            "conflicting_labels": len(conflicts),
        },
    )
