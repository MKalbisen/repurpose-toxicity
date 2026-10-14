"""Chunked PrimeKG parsing and leakage-resistant splitting."""

from collections import Counter

import numpy as np
import pandas as pd

from repurpose.utils import ROOT, paths, write_json

ALLOWED = {"drug", "disease", "gene/protein", "pathway"}
COLUMNS = ["relation", "x_id", "x_type", "x_name", "y_id", "y_type", "y_name"]


def prepare_kg(config):
    base, _, reports = paths(config)
    source = ROOT / "data/raw/primekg/primekg_raw.csv"
    nodes, pieces, relation_counts, type_counts = {}, [], Counter(), Counter()
    missing = Counter()
    rows = 0
    max_entities = config["kg"]["max_entities"]
    # Full scan first: source statistics and therapeutically connected seed nodes.
    indications = []
    for chunk in pd.read_csv(source, dtype=str, chunksize=150000):
        if not set(COLUMNS) <= set(chunk):
            raise ValueError(f"PrimeKG kolonları eksik: {set(COLUMNS) - set(chunk)}")
        rows += len(chunk)
        relation_counts.update(chunk.relation.value_counts().to_dict())
        type_counts.update(chunk.x_type.value_counts().to_dict())
        type_counts.update(chunk.y_type.value_counts().to_dict())
        missing.update(chunk[COLUMNS].isna().sum().to_dict())
        target = chunk[
            (chunk.relation == "indication")
            & (
                ((chunk.x_type == "drug") & (chunk.y_type == "disease"))
                | ((chunk.y_type == "drug") & (chunk.x_type == "disease"))
            )
        ]
        indications.append(target[COLUMNS])
    targets = pd.concat(indications, ignore_index=True)
    allowed_ids = None
    if max_entities:
        # Choose drugs by therapeutic degree, then diseases and shared neighborhood.
        drugs = pd.concat(
            [targets.loc[targets.x_type == "drug", "x_id"], targets.loc[targets.y_type == "drug", "y_id"]]
        ).value_counts()
        selected_drugs = set(drugs.head(max(12, max_entities // 5)).index)
        chosen = targets[targets.x_id.isin(selected_drugs) | targets.y_id.isin(selected_drugs)]
        diseases = pd.concat(
            [chosen.loc[chosen.x_type == "disease", "x_id"], chosen.loc[chosen.y_type == "disease", "y_id"]]
        ).value_counts()
        allowed_ids = {"drug:" + x for x in selected_drugs}
        allowed_ids |= {"disease:" + x for x in diseases.head(max_entities // 3).index}
    for chunk in pd.read_csv(source, dtype=str, chunksize=150000, usecols=COLUMNS):
        chunk = chunk[chunk.x_type.isin(ALLOWED) & chunk.y_type.isin(ALLOWED)].dropna()
        chunk["head"] = chunk.x_type + ":" + chunk.x_id
        chunk["tail"] = chunk.y_type + ":" + chunk.y_id
        if allowed_ids is not None:
            # Expand once to bounded gene/pathway nodes; never add arbitrary diseases.
            adjacent = chunk[chunk["head"].isin(allowed_ids) | chunk["tail"].isin(allowed_ids)]
            for side in ["x", "y"]:
                for row in adjacent[[f"{side}_type", f"{side}_id"]].itertuples(index=False, name=None):
                    key = ":".join(row)
                    if (
                        row[0] in {"gene/protein", "pathway"}
                        and len(allowed_ids) < max_entities
                        and (
                            row[0] != "gene/protein"
                            or sum(x.startswith("gene/protein:") for x in allowed_ids) < max_entities // 3
                        )
                    ):
                        allowed_ids.add(key)
            chunk = chunk[chunk["head"].isin(allowed_ids) & chunk["tail"].isin(allowed_ids)]
        for side in ["x", "y"]:
            for typ, ident, name in chunk[[f"{side}_type", f"{side}_id", f"{side}_name"]].itertuples(
                index=False, name=None
            ):
                key = f"{typ}:{ident}"
                nodes[key] = (key, ident, name, typ)
        pieces.append(chunk[["head", "relation", "tail"]])
    triples = pd.concat(pieces).drop_duplicates().sort_values(["relation", "head", "tail"])
    cap = config["kg"]["max_triples"]
    if cap and len(triples) > cap:
        # Reserve room for therapeutic edges and gene/pathway/context relations.
        indication = triples[triples.relation == "indication"]
        others = triples[triples.relation != "indication"]
        n_indication = min(len(indication), cap // 2 if len(others) else cap)
        triples = pd.concat(
            [
                indication.sample(n=n_indication, random_state=config["seed"]),
                others.sample(n=min(len(others), cap - n_indication), random_state=config["seed"]),
            ]
        )
    triples = triples.reset_index(drop=True)
    used = set(triples["head"]) | set(triples["tail"])
    entities = pd.DataFrame(
        [v for k, v in nodes.items() if k in used], columns=["entity_id", "source_id", "name", "type"]
    ).sort_values("entity_id")
    entities.to_csv(base / "entities.csv", index=False)
    triples.to_csv(base / "triples.tsv", sep="\t", index=False, header=False)
    triples.relation.value_counts().rename_axis("relation").reset_index(name="count").to_csv(
        base / "relations.csv", index=False
    )
    write_json(
        reports / "kg_data_report.json",
        {
            "source_rows": rows,
            "source_relations": relation_counts,
            "source_endpoint_types": type_counts,
            "missing": missing,
            "entities": len(entities),
            "triples": len(triples),
            "profile": config["profile"],
        },
    )
    (reports / "data_dictionary.md").write_text(
        "# Veri sözlüğü\n\nPrimeKG kaynak kimliği `type:source_id` ile namespace edilir. "
        "DrugBank DB kimlikleri drug, MONDO disease, gen/protein ve pathway kaynak adları "
        "PrimeKG CSV alanlarında korunur. head/relation/tail TSV başlıksızdır. "
        "entities.csv: entity_id, source_id, name, type. "
        "Kaynak eksikleri kg_data_report.json içinde; split kenar grupları split_report.json içinde.\n",
        encoding="utf-8",
    )
    train, val, test, report = split_kg(triples, config["seed"])
    for name, frame in [("train", train), ("validation", val), ("test", test)]:
        frame.to_csv(base / f"{name}.tsv", sep="\t", index=False, header=False)
    write_json(reports / "split_report.json", report)
    return entities, triples


def edge_group(h, r, t):
    # All relations between the same endpoint pair stay together, including reverse
    # relations, contraindication/indication duplicates and symmetric copies.
    return tuple(sorted((h, t)))


def split_kg(triples, seed=42):
    groups = {}
    for i, row in enumerate(triples.itertuples(index=False, name=None)):
        groups.setdefault(edge_group(*row), []).append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    entity_counts = Counter(list(triples["head"]) + list(triples["tail"]))
    relation_counts = Counter(triples.relation)
    held = [[], []]
    targets = [int(len(triples) * 0.1), int(len(triples) * 0.1)]
    removed = set()
    for key in keys:
        idx = groups[key]
        part = triples.iloc[idx]
        ec = Counter(list(part["head"]) + list(part["tail"]))
        rc = Counter(part.relation)
        if any(entity_counts[k] <= n for k, n in ec.items()):
            continue
        if any(relation_counts[k] <= n for k, n in rc.items()):
            continue
        slot = 0 if len(held[0]) < targets[0] else 1
        if len(held[slot]) >= targets[slot]:
            continue
        held[slot].extend(idx)
        removed.update(idx)
        entity_counts.subtract(ec)
        relation_counts.subtract(rc)
    train = triples.iloc[[i for i in range(len(triples)) if i not in removed]].copy()
    val, test = [triples.iloc[sorted(i)].copy() for i in held]
    if val.empty or test.empty:
        raise ValueError("Coverage koruyan split için yeterli KG yok")
    parts = [train, val, test]
    gsets = [{edge_group(*row) for row in p.itertuples(index=False, name=None)} for p in parts]
    overlaps = [len(gsets[i] & gsets[j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
    if any(overlaps):
        raise AssertionError("KG ters ilişki sızıntısı")
    report = {
        "seed": seed,
        "sizes": dict(zip(["train", "validation", "test"], map(len, parts))),
        "requested_ratios": [0.8, 0.1, 0.1],
        "actual_ratios": [len(p) / len(triples) for p in parts],
        "pair_group_overlap": overlaps,
        "entity_coverage": True,
        "relation_coverage": True,
        "policy": "All relations and directions for an unordered endpoint pair stay in one split.",
    }
    return train, val, test, report
