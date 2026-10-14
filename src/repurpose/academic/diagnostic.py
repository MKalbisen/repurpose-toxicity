"""Legacy validation-only adapter check. It never loads final test triples."""

import json

import pandas as pd
import torch

from repurpose.academic.metrics import canonical_indication_pairs, drug_query_metrics, pykeen_drug_scorer
from repurpose.academic.state import atomic_json, now
from repurpose.utils import ROOT, seed_all


def validation_diagnostic():
    seed_all(42)
    base, output = ROOT / "data/processed/smoke", ROOT / "reports/academic"
    entities = pd.read_csv(base / "entities.csv", dtype=str)

    def read(name):
        return pd.read_csv(base / f"{name}.tsv", sep="\t", header=None, names=["head", "relation", "tail"])

    train, validation = read("train"), read("validation")
    model_dir = ROOT / "models/smoke/kg"
    metadata = json.loads((model_dir / "best_kge_config.json").read_text())
    model = torch.load(model_dir / metadata["model_path"], map_location="cpu", weights_only=False)
    target_pairs = canonical_indication_pairs(validation)
    known_pairs = canonical_indication_pairs(pd.concat([train, validation], ignore_index=True))
    drugs = entities[entities.type == "drug"].entity_id.tolist()
    frame, recall, summary = drug_query_metrics(
        pykeen_drug_scorer(model, metadata), drugs, target_pairs, known_pairs
    )
    frame.to_csv(output / "validation_drug_ranks.csv", index=False)
    recall.to_csv(output / "validation_drug_recall.csv", index=False)
    # Train popularity comparator uses training indication pairs only.
    popularity = pd.Series([d for d, _ in canonical_indication_pairs(train)]).value_counts().to_dict()
    _, _, pop_summary = drug_query_metrics(
        lambda disease, candidates: [popularity.get(drug, 0) for drug in candidates],
        drugs,
        target_pairs,
        known_pairs,
    )
    atomic_json(
        output / "validation_diagnostic.json",
        {
            "time": now(),
            "scope": "retrospective legacy smoke validation diagnostic; NOT academic final test",
            "model": metadata["model"],
            "filter_scope": "train + validation; test triples not loaded",
            "model_metrics": summary,
            "train_popularity_metrics": pop_summary,
            "limitations": [
                "Smoke model was selected using legacy all-relation validation metrics.",
                "This diagnostic does not retrospectively change training or model selection.",
                "Previous smoke test exposure exists; no independent-test claim.",
            ],
        },
    )
    return summary
