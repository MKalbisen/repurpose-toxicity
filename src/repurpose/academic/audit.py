"""Read-only measurements of the existing MVP, without new model test evaluation."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from repurpose.academic.state import atomic_json, now
from repurpose.utils import ROOT, TASKS, sha256


def kg_audit(base):
    entities = pd.read_csv(base / "entities.csv", dtype=str)
    triples = pd.read_csv(
        base / "triples.tsv", sep="\t", header=None, names=["head", "relation", "tail"], dtype=str
    )
    parts = {
        name: pd.read_csv(
            base / f"{name}.tsv", sep="\t", header=None, names=["head", "relation", "tail"], dtype=str
        )
        for name in ["train", "validation", "test"]
    }
    training_entities = set(parts["train"]["head"]) | set(parts["train"]["tail"])
    training_relations = set(parts["train"].relation)
    group_sets = {
        name: {tuple(sorted((h, t))) for h, _, t in frame.itertuples(index=False, name=None)}
        for name, frame in parts.items()
    }
    exact_sets = {name: set(frame.itertuples(index=False, name=None)) for name, frame in parts.items()}
    targets = {}
    for name, part in parts.items():
        indication = part[part.relation == "indication"]
        pairs = set()
        directions = Counter()
        for h, _, t in indication.itertuples(index=False, name=None):
            if h.startswith("drug:") and t.startswith("disease:"):
                pairs.add((h, t))
                directions["drug_to_disease"] += 1
            elif h.startswith("disease:") and t.startswith("drug:"):
                pairs.add((t, h))
                directions["disease_to_drug"] += 1
            else:
                directions["unexpected"] += 1
        targets[name] = {
            "indication_rows": len(indication),
            "unique_drug_disease_pairs": len(pairs),
            "directions": dict(directions),
            "unique_target_diseases": len({t for _, t in pairs}),
        }
    checks = {}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        checks[f"{a}_{b}_pair_overlap"] = len(group_sets[a] & group_sets[b])
        checks[f"{a}_{b}_exact_overlap"] = len(exact_sets[a] & exact_sets[b])
    for name in ["validation", "test"]:
        checks[f"{name}_unseen_entities"] = len(
            (set(parts[name]["head"]) | set(parts[name]["tail"])) - training_entities
        )
        checks[f"{name}_unseen_relations"] = len(set(parts[name].relation) - training_relations)
    checks["partition_unique_triples_complete"] = set.union(*exact_sets.values()) == set(
        triples.itertuples(index=False, name=None)
    )
    mapping = pd.read_csv(base / "drug_chemical_mapping.csv")
    drug_ids = set(entities.loc[entities.type == "drug", "entity_id"])
    accepted = mapping[(mapping.status == "matched") & mapping.drug_id.isin(drug_ids)]
    return {
        "entities": len(entities),
        "entity_types": entities.type.value_counts().to_dict(),
        "triples": len(triples),
        "relations": triples.relation.value_counts().to_dict(),
        "split_sizes": {name: len(part) for name, part in parts.items()},
        "checks": checks,
        "target_indication": targets,
        "mapping": {
            "drug_count": len(drug_ids),
            "matched_drugs": accepted.drug_id.nunique(),
            "coverage": accepted.drug_id.nunique() / len(drug_ids),
            "duplicate_drug_mappings": int(accepted.drug_id.duplicated().sum()),
        },
    }


def toxicity_audit(base):
    clean = pd.read_csv(base / "tox21_clean.csv")
    with np.load(base / "tox21_features.npz") as packed:
        x, y = packed["X"], packed["Y"]
        indices = {name: packed[name] for name in ["train", "val", "test"]}
    membership = np.full(len(clean), "", dtype=object)
    for name, ids in indices.items():
        membership[ids] = name
    canonical = clean.canonical_smiles
    # Fingerprint-only equivalence detects identical model inputs, even when SMILES differ.
    hashes = [row.tobytes().hex() for row in np.packbits(x, axis=1)]
    identity_frame = pd.DataFrame(
        {
            "canonical_smiles": canonical,
            "scaffold": clean.scaffold,
            "fingerprint": hashes,
            "split": membership,
        }
    )
    overlaps = {}
    for column in ["canonical_smiles", "scaffold", "fingerprint"]:
        grouped = identity_frame.groupby(column)["split"].nunique()
        overlaps[column] = int((grouped > 1).sum())
    all_indices = np.concatenate(list(indices.values()))
    checks = {
        "feature_shape_matches": x.shape == (len(clean), 1024),
        "label_shape_matches": y.shape == (len(clean), 12),
        "binary_features": bool(np.isin(x, [0, 1]).all()),
        "binary_or_missing_labels": bool(np.isin(y[np.isfinite(y)], [0, 1]).all()),
        "split_covers_every_row_once": len(all_indices) == len(clean)
        and len(np.unique(all_indices)) == len(clean),
        "npz_labels_match_csv": bool(np.allclose(y, clean[TASKS].to_numpy(), equal_nan=True)),
    }
    counts = []
    scaffold_stats = {}
    for name, ids in indices.items():
        group_counts = clean.iloc[ids].scaffold.value_counts()
        scaffold_stats[name] = {
            "n": len(ids),
            "groups": len(group_counts),
            "largest_group": int(group_counts.max()),
            "acyclic_count": int((clean.iloc[ids].scaffold == "__ACYCLIC__").sum()),
        }
        for j, task in enumerate(TASKS):
            labels = y[ids, j]
            observed = int(np.isfinite(labels).sum())
            positive = int(np.nansum(labels))
            counts.append(
                {
                    "split": name,
                    "assay": task,
                    "n_labeled": observed,
                    "positive": positive,
                    "negative": observed - positive,
                    "missing": len(labels) - observed,
                    "prevalence": positive / observed if observed else None,
                }
            )
    return {
        "molecules": len(clean),
        "feature_shape": list(x.shape),
        "missing_labels": int(np.isnan(y).sum()),
        "checks": checks,
        "cross_split_equivalent_groups": overlaps,
        "scaffold_distribution": scaffold_stats,
        "label_distribution": counts,
        "canonical_duplicate_rows": int(canonical.duplicated().sum()),
        "fingerprint_duplicate_rows": int(pd.Series(hashes).duplicated().sum()),
    }


def audit(project_root=ROOT):
    root = Path(project_root)
    base = root / "data/processed/smoke"
    directory = root / "reports/academic"
    directory.mkdir(parents=True, exist_ok=True)
    kg, tox = kg_audit(base), toxicity_audit(base)
    env = json.loads((root / "reports/environment_check.json").read_text())
    manifest = json.loads((root / "reports/data_manifest.json").read_text())
    files = [
        root / "requirements.txt",
        root / "configs/academic_protocol.yaml",
        root / "reports/data_manifest.json",
    ]
    files += list((root / "src/repurpose").rglob("*.py"))
    files += [
        base / n
        for n in [
            "entities.csv",
            "triples.tsv",
            "train.tsv",
            "validation.tsv",
            "test.tsv",
            "tox21_clean.csv",
            "tox21_features.npz",
            "drug_chemical_mapping.csv",
        ]
    ]
    fingerprints = [
        {"path": str(p.relative_to(root)), "sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(files)
    ]
    findings = [
        {
            "id": "F01",
            "priority": "critical",
            "next_stage": "A02",
            "finding": "Legacy KGE selection/evaluation uses all relations and all entity types; primary drug-only indication metric missing.",
        },
        {
            "id": "F02",
            "priority": "critical",
            "next_stage": "A04",
            "finding": "Legacy training automatically evaluates test after each run; separate train/select/final-evaluate gate required.",
        },
        {
            "id": "F03",
            "priority": "high",
            "next_stage": "A03",
            "finding": "Smoke subgraph selection favors high-degree drugs and input file order; cannot be the general academic cohort.",
        },
        {
            "id": "F04",
            "priority": "high",
            "next_stage": "A02",
            "finding": "Legacy confidence intervals resample individual molecules; group and paired uncertainty missing.",
        },
        {
            "id": "F05",
            "priority": "high",
            "next_stage": "A03",
            "finding": "Tautomer enumeration completion/identity audit absent; rejected molecule and limit handling must be versioned.",
        },
        {
            "id": "F06",
            "priority": "high",
            "next_stage": "A03",
            "finding": "Mirror is hash-verified but official version/provenance/license equivalence remains unverified.",
        },
        {
            "id": "F07",
            "priority": "high",
            "next_stage": "A04",
            "finding": "Legacy paths key on profile only, model artifacts overwrite, optimizer/RNG epoch resume absent.",
        },
        {
            "id": "F08",
            "priority": "high",
            "next_stage": "A06",
            "finding": "Only one short training seed; classical baselines, grouped calibration and fixed task denominator comparison absent.",
        },
        {
            "id": "F09",
            "priority": "high",
            "next_stage": "A09",
            "finding": "No independent external validation. Previous smoke test exposure must be disclosed.",
        },
        {
            "id": "F10",
            "priority": "medium",
            "next_stage": "A03",
            "finding": "No nearest-training chemistry similarity audit; zero scaffold overlap alone does not demonstrate independence.",
        },
        {
            "id": "F11",
            "priority": "medium",
            "next_stage": "A04",
            "finding": "Full KG splitting constructs Python groups and loops over dataframe groups; scalability must be measured before full run.",
        },
    ]
    if any(tox["cross_split_equivalent_groups"].values()):
        findings.append(
            {
                "id": "F12",
                "priority": "critical",
                "next_stage": "A03",
                "finding": "Equivalent chemistry/model inputs cross split boundaries; fix before academic training.",
            }
        )
    # Audit measures retrospective existing artifacts. It does not train or score a new final test.
    result = {
        "created": now(),
        "scope": "MVP readiness audit; no new model final-test scores",
        "hardware": {
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "cuda_available": env["cuda_available"],
            "ram_note": "Host free -h inspection: about 31 GiB total; availability varies.",
        },
        "data_origin": manifest["origin"],
        "primekg_version": manifest["primekg_version"],
        "kg": kg,
        "toxicity": tox,
        "findings": findings,
        "input_fingerprints": fingerprints,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "torch", "pykeen", "rdkit", "scikit-learn", "xgboost"]
        },
    }
    atomic_json(directory / "audit.json", result)
    pd.DataFrame(tox["label_distribution"]).to_csv(directory / "assay_label_audit.csv", index=False)
    lines = [
        "# Akademik hazırlık denetimi",
        "",
        f"Tarih: {result['created']}",
        "Mevcut smoke verileri/kodu incelendi; yeni final model test skoru üretilmedi.",
        "",
        f"KG: {kg['entities']} entity, {kg['triples']} triple. Kaynak: {result['data_origin']}.",
        f"Tox21: {tox['molecules']} molekül; {tox['missing_labels']} eksik assay etiketi.",
        f"Kimyasal eşleme: {kg['mapping']['matched_drugs']}/{kg['mapping']['drug_count']}.",
        "",
        "## Ölçülen sızıntı/coverage kontrolleri",
        "",
        "KG: " + json.dumps(kg["checks"], ensure_ascii=False),
        "Tox21: " + json.dumps(tox["checks"], ensure_ascii=False),
        "Split kesişen kimlik grupları: " + json.dumps(tox["cross_split_equivalent_groups"]),
        "Scaffold dağılımı: " + json.dumps(tox["scaffold_distribution"]),
        "",
        "## Açıklar ve giderecek aşama",
        "",
        "| ID | Öncelik | Aşama | Bulgu |",
        "|---|---|---|---|",
    ]
    lines += [f"| {f['id']} | {f['priority']} | {f['next_stage']} | {f['finding']} |" for f in findings]
    lines += [
        "",
        "Kod/veri hash envanteri audit.json içinde. Ayrıntılı assay dağılımı assay_label_audit.csv içinde.",
        "A01 denetimin tamamlanmasıdır; kritik açıkların giderildiği veya akademik sonuçların hazır olduğu anlamına gelmez.",
    ]
    (directory / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
