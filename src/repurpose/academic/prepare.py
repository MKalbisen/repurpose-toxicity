"""Versioned chemistry and disk-backed, order-independent KG preparation.

No model selection or test performance calculation belongs in this module.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize as standard
from rdkit.Chem.Scaffolds import MurckoScaffold

from repurpose.academic.state import atomic_json
from repurpose.data.chemistry import GENERATOR, canonicalize, fingerprint
from repurpose.data.kg_data import ALLOWED, COLUMNS
from repurpose.utils import ROOT, TASKS, sha256

POLICY = {
    "version": "chemical-components-v1",
    "input": "isomeric canonical SMILES; salt, charge, stereo retained",
    "group": "transitive closure of achiral parent, tautomer, scaffold and input fingerprint",
    "fingerprint": "Morgan radius 2, 1024 bits, chirality false",
    "max_tautomers": 1000,
    "max_transforms": 1000,
    "incomplete_enumeration": "exclude before split",
    "acyclic": "one indivisible group",
    "split": "largest connected groups first, label blind, 70/10/10/10 train/validation/calibration/test",
    "rdkit": rdBase.rdkitVersion,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def chemical_identity(smiles):
    canonical = canonicalize(smiles)
    if canonical is None:
        return {"status": "invalid_smiles"}
    result = {"canonical_smiles": canonical, "status": "standardization_error"}
    try:
        mol = Chem.MolFromSmiles(canonical)
        parent = standard.FragmentParent(mol)
        Chem.RemoveStereochemistry(parent)
        result.update(parent=Chem.MolToSmiles(parent), fragments=len(Chem.GetMolFrags(mol)))
        enumerator = standard.TautomerEnumerator()
        enumerator.SetMaxTautomers(POLICY["max_tautomers"])
        enumerator.SetMaxTransforms(POLICY["max_transforms"])
        enumerated = enumerator.Enumerate(parent)
        result["enumeration_status"] = str(enumerated.status)
        result["tautomer_count"] = len(enumerated)
        if enumerated.status != standard.TautomerEnumeratorStatus.Completed:
            result["status"] = "incomplete_enumeration"
            return result
        tautomer = enumerator.PickCanonical(enumerated)
        Chem.RemoveStereochemistry(tautomer)
        result.update(
            status="accepted",
            tautomer=Chem.MolToSmiles(tautomer),
            scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=tautomer, includeChirality=False)
            or "__ACYCLIC__",
            fingerprint=hashlib.sha256(fingerprint(canonical).tobytes()).hexdigest(),
        )
    except (ValueError, RuntimeError) as error:
        result["error"] = str(error)
    return result


def component_groups(frame, columns=("parent", "tautomer", "scaffold", "fingerprint")):
    """Union-find: chains across different identity definitions cannot cross splits."""
    parents = list(range(len(frame)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for column in columns:
        seen = {}
        for i, key in enumerate(frame[column]):
            if pd.isna(key) or key == "":
                raise ValueError(f"Missing identity in {column}")
            if key in seen:
                a, b = find(i), find(seen[key])
                parents[max(a, b)] = min(a, b)
            else:
                seen[key] = i
    members = {}
    for i in range(len(frame)):
        members.setdefault(find(i), []).append(str(frame.iloc[i].canonical_smiles))
    keys = {i: digest(sorted(values))[:20] for i, values in members.items()}
    return np.array([keys[find(i)] for i in range(len(frame))])


def group_split(groups, seed=2026, ratios=(0.7, 0.1, 0.1, 0.1)):
    if not np.isclose(sum(ratios), 1) or min(ratios) <= 0:
        raise ValueError("Split ratios must be positive and sum to one")
    groups = np.asarray(groups)
    unique, sizes = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    order = sorted(order, key=lambda i: -sizes[i])
    assignment, counts = {}, np.zeros(len(ratios), dtype=int)
    for i in order:
        slot = int(np.argmax(np.asarray(ratios) * len(groups) - counts))
        assignment[unique[i]] = slot
        counts[slot] += sizes[i]
    splits = [np.flatnonzero([assignment[g] == i for g in groups]) for i in range(len(ratios))]
    if any(len(ids) == 0 for ids in splits):
        raise ValueError("Not enough independent groups for nonempty partitions")
    return splits


def verify_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    return manifest


def prepare_chemistry(directory, raw=None, seed=2026):
    raw = Path(raw or ROOT / "data/raw/tox21/tox21.csv.gz")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity_key = digest({"raw": sha256(raw), "policy": POLICY, "seed": seed})
    if (directory / "manifest.json").exists():
        manifest = verify_bundle(directory)
        if manifest["cache_key"] != identity_key:
            raise ValueError("Chemistry policy/source changed; use a new dataset directory")
        return manifest
    start = time.perf_counter()
    data = pd.read_csv(raw)
    if not set(TASKS + ["smiles"]) <= set(data):
        raise ValueError("Missing Tox21 columns")
    if not all(set(data[t].dropna().unique()) <= {0, 1} for t in TASKS):
        raise ValueError("Non-binary Tox21 labels")
    data["source_row"] = np.arange(len(data))
    identities, cache = [], {}
    cache_file = directory / "identity_cache.json"
    if cache_file.exists():
        stored = json.loads(cache_file.read_text(encoding="utf-8"))
        if stored["cache_key"] == identity_key:
            cache = stored["entries"]
    for i, smiles in enumerate(data.smiles):
        key = str(smiles)
        if key not in cache:
            cache[key] = chemical_identity(smiles)
        identities.append({"source_row": i, "raw_smiles": smiles, **cache[key]})
        if (i + 1) % 500 == 0:
            atomic_json(cache_file, {"cache_key": identity_key, "entries": cache})
            print(f"Chemistry identities {i + 1}/{len(data)}", flush=True)
    atomic_json(cache_file, {"cache_key": identity_key, "entries": cache})
    identity = pd.DataFrame(identities)
    identity.to_csv(directory / "chemical_identity.csv", index=False)
    identity[identity.status != "accepted"].to_csv(directory / "excluded_molecules.csv", index=False)
    accepted = identity[identity.status == "accepted"].merge(data[TASKS + ["source_row"]], on="source_row")
    rows, conflicts = [], []
    for smiles, part in accepted.groupby("canonical_smiles", sort=True):
        row = {k: part.iloc[0][k] for k in ["parent", "tautomer", "scaffold", "fingerprint"]}
        row.update(canonical_smiles=smiles, source_rows=";".join(map(str, sorted(part.source_row))))
        for task in TASKS:
            labels = part[task].dropna().unique()
            row[task] = float(labels[0]) if len(labels) == 1 else np.nan
            if len(labels) > 1:
                conflicts.append({"canonical_smiles": smiles, "assay": task})
        rows.append(row)
    clean = pd.DataFrame(rows)
    clean["group"] = component_groups(clean)
    indices = group_split(clean.group, seed)
    names = ["train", "validation", "calibration", "test"]
    x = np.stack(clean.canonical_smiles.map(fingerprint))
    overlap = {}
    for column in ["canonical_smiles", "parent", "tautomer", "scaffold", "fingerprint", "group"]:
        sets = [set(clean.iloc[ids][column]) for ids in indices]
        overlap[column] = sum(len(sets[i] & sets[j]) for i in range(4) for j in range(i))
    if any(overlap.values()):
        raise AssertionError("Chemical identity crosses partitions")
    for name, ids in zip(names, indices):
        clean.loc[ids, "split"] = name
        np.savez_compressed(
            directory / f"{name}.npz",
            X=x[ids],
            Y=clean.iloc[ids][TASKS].to_numpy(float),
            groups=clean.iloc[ids].group.to_numpy(str),
            rows=ids,
        )
    clean.to_csv(directory / "molecules.csv", index=False)
    pd.DataFrame(conflicts, columns=["canonical_smiles", "assay"]).to_csv(
        directory / "label_conflicts.csv", index=False
    )
    # Structure-only audit: no test labels/performance are used for decisions.
    bits = [GENERATOR.GetFingerprint(Chem.MolFromSmiles(s)) for s in clean.canonical_smiles]
    train_bits = [bits[i] for i in indices[0]]
    similarity = []
    for name, ids in zip(names[1:], indices[1:]):
        for i in ids:
            similarity.append(
                {
                    "row": int(i),
                    "split": name,
                    "nearest_train_tanimoto": max(DataStructs.BulkTanimotoSimilarity(bits[i], train_bits)),
                }
            )
    pd.DataFrame(similarity).to_csv(directory / "nearest_train_similarity.csv", index=False)
    label_rows = []
    for name, ids in zip(names[:3], indices[:3]):
        for task in TASKS:
            y = clean.iloc[ids][task]
            label_rows.append(
                {
                    "split": name,
                    "assay": task,
                    "positive": int((y == 1).sum()),
                    "negative": int((y == 0).sum()),
                    "missing": int(y.isna().sum()),
                }
            )
    pd.DataFrame(label_rows).to_csv(directory / "development_labels.csv", index=False)
    manifest = {
        "cache_key": identity_key,
        "raw_sha256": sha256(raw),
        "policy": POLICY,
        "seed": seed,
        "raw_rows": len(data),
        "clean_rows": len(clean),
        "exclusions": identity[identity.status != "accepted"].status.value_counts().to_dict(),
        "split_sizes": dict(zip(names, map(len, indices))),
        "identity_overlap": overlap,
        "group_counts": clean.groupby("split").group.nunique().to_dict(),
        "largest_component": int(clean.group.value_counts().max()),
        "seconds": time.perf_counter() - start,
        "historical_test_exposure": True,
        "external_validation": False,
    }
    manifest["files"] = {p.name: sha256(p) for p in directory.iterdir() if p.suffix in {".csv", ".npz"}}
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def pair_key(h, t):
    return "\x1f".join(sorted((h, t)))


def pair_partition(pair, seed):
    value = int(hashlib.sha256(f"{seed}:{pair}".encode()).hexdigest()[:8], 16) / 2**32
    return "train" if value < 0.8 else "validation" if value < 0.9 else "test"


def prepare_graph(directory, raw=None, seed=2026):
    """Full eligible graph, SQLite bounded memory; no degree or input-order sampling."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = Path(raw or ROOT / "data/raw/primekg/primekg_raw.csv")
    source_hash = sha256(raw)
    if (directory / "manifest.json").exists():
        manifest = verify_bundle(directory)
        if manifest["raw_sha256"] != source_hash or manifest["seed"] != seed:
            raise ValueError("KG source/split changed; choose a new directory")
        return manifest
    start = time.perf_counter()
    db_path = directory / "preparation.sqlite"
    db = sqlite3.connect(db_path)
    try:
        db.execute("PRAGMA cache_size=-32000")
        db.execute("PRAGMA temp_store=FILE")
        db.executescript(
            "DROP TABLE IF EXISTS edges; DROP TABLE IF EXISTS nodes; DROP TABLE IF EXISTS anchors;"
            "CREATE TABLE edges(h TEXT,r TEXT,t TEXT,p TEXT,s TEXT,PRIMARY KEY(h,r,t));"
            "CREATE TABLE nodes(id TEXT PRIMARY KEY,name TEXT,type TEXT);"
        )
        counts = Counter()
        for chunk in pd.read_csv(raw, dtype=str, usecols=COLUMNS, chunksize=50000):
            counts["raw_rows"] += len(chunk)
            chunk = chunk[chunk.x_type.isin(ALLOWED) & chunk.y_type.isin(ALLOWED)]
            counts["eligible_rows"] += len(chunk)
            valid = chunk.dropna(subset=["relation", "x_id", "x_type", "y_id", "y_type"])
            counts["missing_id_rows"] += len(chunk) - len(valid)
            edges, nodes = [], []
            for rel, xi, xt, xn, yi, yt, yn in valid[COLUMNS].itertuples(index=False, name=None):
                h, t = f"{xt}:{xi}", f"{yt}:{yi}"
                nodes.extend(
                    [(h, str(xn) if pd.notna(xn) else h, xt), (t, str(yn) if pd.notna(yn) else t, yt)]
                )
                if rel == "indication":
                    if xt == "disease" and yt == "drug":
                        h, t = t, h
                    if not h.startswith("drug:") or not t.startswith("disease:"):
                        raise ValueError("Unexpected indication endpoint types")
                pair = pair_key(h, t)
                edges.append((h, rel, t, pair, pair_partition(pair, seed)))
            db.executemany("INSERT OR IGNORE INTO edges VALUES (?,?,?,?,?)", edges)
            # Conflicting names resolve lexically, independently of input row order.
            db.executemany(
                "INSERT INTO nodes VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET name=min(name,excluded.name)",
                nodes,
            )
            db.commit()
            if counts["raw_rows"] % 500000 == 0:
                print(f"KG scanned {counts['raw_rows']:,} rows", flush=True)
        db.execute("CREATE INDEX edge_pair ON edges(p)")
        # Reserve one lexical pair per entity and relation for transductive coverage.
        db.executescript(
            "CREATE TABLE anchors(p TEXT PRIMARY KEY);"
            "INSERT OR IGNORE INTO anchors SELECT min(p) FROM edges GROUP BY h;"
            "INSERT OR IGNORE INTO anchors SELECT min(p) FROM edges GROUP BY t;"
            "INSERT OR IGNORE INTO anchors SELECT min(p) FROM edges GROUP BY r;"
            "UPDATE edges SET s='train' WHERE p IN (SELECT p FROM anchors);"
        )
        db.commit()
        sizes, relations = {}, {}
        for split in ["train", "validation", "test"]:
            cursor = db.execute("SELECT h,r,t FROM edges WHERE s=? ORDER BY h,r,t", (split,))
            n = 0
            with (directory / f"{split}.tsv").open("w", encoding="utf-8", newline="") as stream:
                while rows := cursor.fetchmany(20000):
                    stream.writelines("\t".join(row) + "\n" for row in rows)
                    n += len(rows)
            sizes[split] = n
            relations[split] = dict(db.execute("SELECT r,count(*) FROM edges WHERE s=? GROUP BY r", (split,)))
        if any(n == 0 for n in sizes.values()):
            raise ValueError("Empty KG partition")
        nodes = pd.read_sql_query("SELECT id AS entity_id,name,type FROM nodes ORDER BY id", db)
        nodes.to_csv(directory / "entities.csv", index=False)
        checks = {
            "pair_overlap": db.execute(
                "SELECT count(*) FROM (SELECT p FROM edges GROUP BY p HAVING count(DISTINCT s)>1)"
            ).fetchone()[0],
            "unseen_entities": db.execute(
                "SELECT count(*) FROM nodes WHERE id NOT IN (SELECT h FROM edges WHERE s='train' UNION SELECT t FROM edges WHERE s='train')"
            ).fetchone()[0],
            "unseen_relations": len(set().union(*map(set, relations.values())) - set(relations["train"])),
        }
        if any(checks.values()):
            raise AssertionError(checks)
        manifest = {
            "raw_sha256": source_hash,
            "seed": seed,
            "counts": dict(counts),
            "sizes": sizes,
            "entities": len(nodes),
            "relations": relations,
            "checks": checks,
            "policy": "full eligible types; canonical indication; SHA256 pair split; lexical coverage anchors",
            "allowed_types": sorted(ALLOWED),
            "seconds": time.perf_counter() - start,
            "provenance": "pinned community mirror; official version equivalence unverified",
            "license": "underlying source restrictions apply; no redistribution authorization implied",
        }
        manifest["files"] = {p.name: sha256(p) for p in directory.iterdir() if p.suffix in {".tsv", ".csv"}}
        atomic_json(directory / "manifest.json", manifest)
        return manifest
    finally:
        db.close()
