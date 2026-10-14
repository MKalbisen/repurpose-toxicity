import json
import time

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from repurpose.data.chemistry import fingerprint
from repurpose.integration.ranking import rank_candidates, sensitivity
from repurpose.kg.candidate_generator import generate_candidates
from repurpose.toxicity.models import MultiTaskMLP
from repurpose.utils import TASKS, paths, write_json


class InferenceService:
    def __init__(self, config):
        self.config = config
        self.base, self.models, self.reports = paths(config)
        self.entities = pd.read_csv(self.base / "entities.csv", dtype=str)
        self.triples = pd.read_csv(
            self.base / "triples.tsv", sep="\t", header=None, names=["head", "relation", "tail"], dtype=str
        )
        self.mapping = pd.read_csv(self.base / "drug_chemical_mapping.csv")
        self.metadata = json.loads((self.models / "kg/best_kge_config.json").read_text())
        # Only load artifacts generated locally by the offline training command.
        self.kg_model = torch.load(
            self.models / "kg" / self.metadata["model_path"], map_location="cpu", weights_only=False
        )
        self.xgb = {}
        for task in TASKS:
            path = self.models / "toxicity/xgb_models" / f"assay_{task}.json"
            if path.exists():
                clf = XGBClassifier()
                clf.load_model(path)
                self.xgb[task] = clf
        checkpoint = torch.load(self.models / "toxicity/mlp_best.pt", map_location="cpu", weights_only=True)
        self.mlp = MultiTaskMLP()
        self.mlp.load_state_dict(checkpoint["state_dict"])
        self.mlp.eval()
        self.valid_tasks = np.array(checkpoint["valid_tasks"])

    def analyze(self, disease_id, top_n=30, threshold=0.5, weight=0.7, strategy="weighted", model="XGBoost"):
        start = time.perf_counter()
        candidates = generate_candidates(
            self.kg_model, self.metadata, self.entities, self.triples, disease_id, top_n
        )
        mapping = self.mapping[self.mapping.status == "matched"][["drug_id", "canonical_smiles"]]
        if mapping.drug_id.duplicated().any():
            raise ValueError("Çoklu kimyasal eşleme; çıkarım durduruldu")
        candidates = candidates.merge(mapping, on="drug_id", how="left")
        probabilities = np.full((len(candidates), 12), np.nan)
        valid_indices, features = [], []
        for i, smi in enumerate(candidates.canonical_smiles):
            if isinstance(smi, str):
                try:
                    features.append(fingerprint(smi))
                    valid_indices.append(i)
                except ValueError:
                    pass
        if features:
            x = np.stack(features).astype(np.float32)
            if model == "XGBoost":
                for j, task in enumerate(TASKS):
                    if task in self.xgb:
                        probabilities[valid_indices, j] = self.xgb[task].predict_proba(x)[:, 1]
            elif model == "MLP":
                with torch.no_grad():
                    probabilities[valid_indices] = self.mlp(torch.from_numpy(x)).sigmoid().numpy()
                probabilities[:, ~self.valid_tasks] = np.nan
            else:
                raise ValueError("Model XGBoost veya MLP olmalı")
        for j, task in enumerate(TASKS):
            candidates[f"risk_{task}"] = probabilities[:, j]
        candidates["assay_count"] = np.isfinite(probabilities).sum(axis=1)
        candidates["max_risk"] = pd.DataFrame(probabilities).max(axis=1)
        candidates["mean_risk"] = pd.DataFrame(probabilities).mean(axis=1)
        ranked = rank_candidates(candidates, threshold, weight, strategy)
        return ranked, time.perf_counter() - start

    def export(self, ranked, seconds):
        ranked.to_csv(self.reports / "candidate_results.csv", index=False)
        ranked.to_csv(self.reports / "integrated_candidates.csv", index=False)
        ranked.to_csv(self.reports / "final_candidates.csv", index=False)
        write_json(self.reports / "final_ranking.json", ranked.to_dict("records"))
        sensitivity(ranked, self.config["ranking"]["threshold"]).to_csv(
            self.reports / "sensitivity.csv", index=False
        )
        write_json(
            self.reports / "inference_timing.json",
            {
                "seconds": seconds,
                "candidate_count": len(ranked),
                "complete_risk_candidates": int(ranked.complete_risk.sum()),
                "cold_model_load_excluded": True,
            },
        )
