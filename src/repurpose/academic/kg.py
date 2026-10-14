"""PyKEEN representations with explicit development-only training and exact filtering.

The declared objective is softplus logistic negative sampling for BOTH families;
this is not represented as the original papers' default hyperparameter recipe.
"""

from __future__ import annotations

import itertools
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pykeen.models import RotatE, TransE
from pykeen.triples import CoreTriplesFactory

from repurpose.academic.metrics import canonical_indication_pairs, drug_query_metrics, pykeen_drug_scorer
from repurpose.academic.protocol import final_access, verify_lock
from repurpose.academic.state import atomic_json
from repurpose.academic.toxicity import device_for, torch_save_atomic, verify_complete
from repurpose.utils import ROOT, seed_all, sha256, write_json


def graph_data(config):
    base = ROOT / "data/processed" / config["dataset"] / "kg"
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    entities = pd.read_csv(base / "entities.csv", dtype=str)
    metadata = {
        "entity_to_id": {v: i for i, v in enumerate(entities.entity_id)},
        "relation_to_id": {v: i for i, v in enumerate(sorted(manifest["relations"]["train"]))},
    }
    array = np.empty((manifest["sizes"]["train"], 3), dtype=np.int64)
    start = 0
    for chunk in pd.read_csv(
        base / "train.tsv",
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
        dtype=str,
        chunksize=50000,
    ):
        n = len(chunk)
        array[start : start + n] = np.column_stack(
            [
                chunk["head"].map(metadata["entity_to_id"]),
                chunk.relation.map(metadata["relation_to_id"]),
                chunk["tail"].map(metadata["entity_to_id"]),
            ]
        )
        start += n
    factory = CoreTriplesFactory(
        array, num_entities=len(entities), num_relations=len(metadata["relation_to_id"])
    )
    return base, manifest, metadata, factory


def read_targets(base, split):
    pairs = []
    for frame in pd.read_csv(
        base / f"{split}.tsv",
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
        dtype=str,
        chunksize=50000,
    ):
        pairs.extend(canonical_indication_pairs(frame))
    return sorted(set(pairs))


def triple_codes(triples, entities, relations):
    return (triples[..., 0] * relations + triples[..., 1]) * entities + triples[..., 2]


def negative_batch(positive, count, entities, relations, sorted_known):
    negative = positive[:, None, :].expand(-1, count, -1).clone().reshape(-1, 3)
    sides = torch.randint(0, 2, (len(negative),), device=positive.device) * 2
    negative[torch.arange(len(negative), device=positive.device), sides] = torch.randint(
        entities, (len(negative),), device=positive.device
    )
    code = triple_codes(negative, entities, relations)
    pos = torch.searchsorted(sorted_known, code)
    valid = (pos == len(sorted_known)) | (sorted_known[pos.clamp_max(len(sorted_known) - 1)] != code)
    return negative[valid]


def kg_step(model, optimizer, positive, options, known, entities, relations):
    negative = negative_batch(positive, options["negatives"], entities, relations, known)
    optimizer.zero_grad()
    positive_loss = torch.nn.functional.softplus(-model.score_hrt(positive)).mean()
    negative_loss = torch.nn.functional.softplus(model.score_hrt(negative)).mean() if len(negative) else 0
    loss = positive_loss + negative_loss
    loss.backward()
    optimizer.step()
    model.post_parameter_update()
    return float(loss.detach())


def retrieval(model, metadata, targets, known):
    return drug_query_metrics(
        pykeen_drug_scorer(model, metadata),
        [d for d in metadata["entity_to_id"] if d.startswith("drug:")],
        targets,
        known,
    )


def make_model(family, factory, dim, seed, device):
    return {"TransE": TransE, "RotatE": RotatE}[family](
        triples_factory=factory, embedding_dim=dim, random_seed=seed
    ).to(device)


def train_kg(config, run):
    verify_lock(config, run)
    run = Path(run)
    if (run / "kg_test_access.json").exists():
        raise ValueError("Training cannot run after test access")
    base, _manifest, metadata, factory = graph_data(config)
    validation = read_targets(base, "validation")
    known_pairs = read_targets(base, "train") + validation
    device, options = device_for(config), config["kg"]
    known = (
        torch.sort(triple_codes(factory.mapped_triples, factory.num_entities, factory.num_relations))[0]
        .unique()
        .to(device)
    )
    for seed in config["seeds"]:
        for family in ["TransE", "RotatE"]:
            dest = run / "kg" / str(seed) / family
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "complete.json").exists():
                verify_complete(dest)
                continue
            trials = []
            for dim, lr in itertools.product(options["dimensions"], options["learning_rates"]):
                trial = dest / f"d{dim}_lr{lr}"
                trial.mkdir(exist_ok=True)
                if (trial / "result.json").exists():
                    trials.append(json.loads((trial / "result.json").read_text(encoding="utf-8")))
                    continue
                seed_all(seed)
                torch.set_num_threads(config["threads"])
                model = make_model(family, factory, dim, seed, device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                epoch, best, stale, history = 0, -np.inf, 0, []
                resume = trial / "resume.pt"
                if resume.exists():
                    saved = torch.load(resume, map_location=device, weights_only=False)
                    model.load_state_dict(saved["model"])
                    optimizer.load_state_dict(saved["optimizer"])
                    epoch, best, stale, history = (
                        saved["epoch"],
                        saved["best"],
                        saved["stale"],
                        saved["history"],
                    )
                    torch.set_rng_state(saved["torch_rng"].cpu())
                    np.random.set_state(saved["numpy_rng"])
                    random.setstate(saved["python_rng"])
                    if device == "cuda" and saved["cuda_rng"] is not None:
                        torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
                start = time.perf_counter()
                while epoch < options["epochs"] and stale < options["patience"]:
                    model.train()
                    order = torch.randperm(factory.num_triples)
                    total, batches = 0.0, 0
                    for indices in order.split(options["batch_size"]):
                        total += kg_step(
                            model,
                            optimizer,
                            factory.mapped_triples[indices].to(device),
                            options,
                            known,
                            factory.num_entities,
                            factory.num_relations,
                        )
                        batches += 1
                    epoch += 1
                    row = {"epoch": epoch, "loss": total / batches}
                    if epoch % options["validation_every"] == 0 or epoch == options["epochs"]:
                        _, _, summary = retrieval(model, metadata, validation, known_pairs)
                        row["validation_MRR"] = summary["MRR"]
                        if summary["MRR"] > best:
                            best, stale = summary["MRR"], 0
                            torch_save_atomic(
                                {
                                    "state_dict": model.state_dict(),
                                    "family": family,
                                    "dimension": dim,
                                    "metadata": metadata,
                                    "epoch": epoch,
                                },
                                trial / "best.pt",
                            )
                        else:
                            stale += 1
                    history.append(row)
                    torch_save_atomic(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "best": best,
                            "stale": stale,
                            "history": history,
                            "torch_rng": torch.get_rng_state(),
                            "numpy_rng": np.random.get_state(),
                            "python_rng": random.getstate(),
                            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                        },
                        resume,
                    )
                    print(f"KG {family} seed={seed} dim={dim} lr={lr} epoch={epoch} {row}", flush=True)
                record = {
                    "dimension": dim,
                    "lr": lr,
                    "validation_MRR": best,
                    "path": f"{trial.name}/best.pt",
                    "seconds": time.perf_counter() - start,
                    "history": history,
                }
                write_json(trial / "result.json", record)
                trials.append(record)
                del model, optimizer
                if device == "cuda":
                    torch.cuda.empty_cache()
            best_trial = max(trials, key=lambda row: row["validation_MRR"])
            atomic_json(
                dest / "selection.json", {"selected": best_trial, "trials": trials, "test_accessed": False}
            )
            atomic_json(
                dest / "complete.json",
                {
                    "seed": seed,
                    "family": family,
                    "files": {p: sha256(dest / p) for p in ["selection.json", best_trial["path"]]},
                },
            )


def baseline_scores(base):
    popularity = Counter(d for d, _ in read_targets(base, "train"))
    neighbors = defaultdict(set)
    for chunk in pd.read_csv(
        base / "train.tsv",
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
        dtype=str,
        chunksize=50000,
    ):
        for h, _, t in chunk.itertuples(index=False, name=None):
            if h.startswith("gene/protein:"):
                neighbors[t].add(h)
            if t.startswith("gene/protein:"):
                neighbors[h].add(t)
    return {
        "popularity": lambda disease, drugs: [popularity[d] for d in drugs],
        "shared_genes": lambda disease, drugs: [len(neighbors[disease] & neighbors[d]) for d in drugs],
    }


def evaluate_kg(config, run):
    run = Path(run)
    artifacts, selected = [], []
    for seed in config["seeds"]:
        for family in ["TransE", "RotatE"]:
            dest = run / "kg" / str(seed) / family
            record = verify_complete(dest)
            artifacts.extend(dest / p for p in ["complete.json", *record["files"]])
            selected.append((seed, family, dest))
    final_access(config, run, "kg", artifacts)
    base, _manifest, metadata, factory = graph_data(config)
    targets = read_targets(base, "test")
    known = read_targets(base, "train") + read_targets(base, "validation") + targets
    output = run / "evaluation/kg"
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    def record_result(family, seed, result):
        ranks, recall, summary = result
        ranks.to_csv(output / f"{family}_{seed}_ranks.csv", index=False)
        recall.to_csv(output / f"{family}_{seed}_recall.csv", index=False)
        # Cluster by query disease. Micro MRR keeps unequal disease target counts.
        sums = ranks.groupby("disease_id").RR.agg(["sum", "count"]).to_numpy()
        rng = np.random.default_rng(config["split_seed"])
        bootstrap = []
        for _ in range(config["uncertainty"]["bootstrap_repeats"]):
            sample = sums[rng.integers(0, len(sums), len(sums))].sum(axis=0)
            bootstrap.append(sample[0] / sample[1])
        summary.update(
            family=family,
            seed=seed,
            MRR_CI_low=float(np.quantile(bootstrap, 0.025)),
            MRR_CI_high=float(np.quantile(bootstrap, 0.975)),
            bootstrap_group="disease",
        )
        rows.append(summary)

    drugs = [d for d in metadata["entity_to_id"] if d.startswith("drug:")]
    for family, scorer in baseline_scores(base).items():
        record_result(family, "deterministic", drug_query_metrics(scorer, drugs, targets, known))
    for seed, family, dest in selected:
        selection = json.loads((dest / "selection.json").read_text(encoding="utf-8"))["selected"]
        saved = torch.load(dest / selection["path"], map_location=device_for(config), weights_only=True)
        model = make_model(family, factory, saved["dimension"], seed, device_for(config))
        model.load_state_dict(saved["state_dict"])
        record_result(family, seed, retrieval(model, metadata, targets, known))
        del model
    write_json(output / "metrics.json", rows)
    # Paired disease-cluster differences: same test pairs and same sampled diseases.
    comparisons = []
    for seed in config["seeds"]:
        first = pd.read_csv(output / f"RotatE_{seed}_ranks.csv")
        second = pd.read_csv(output / f"TransE_{seed}_ranks.csv")
        paired = first.merge(
            second, on=["drug_id", "disease_id"], suffixes=("_rotate", "_transe"), validate="one_to_one"
        )
        if len(paired) != len(first) or len(paired) != len(second):
            raise ValueError("Paired KG comparison requires identical test targets")
        paired["difference"] = paired.RR_rotate - paired.RR_transe
        grouped = paired.groupby("disease_id").difference.agg(["sum", "count"]).to_numpy()
        rng = np.random.default_rng(config["split_seed"])
        differences = []
        for _ in range(config["uncertainty"]["bootstrap_repeats"]):
            sample = grouped[rng.integers(0, len(grouped), len(grouped))].sum(axis=0)
            differences.append(sample[0] / sample[1])
        comparisons.append(
            {
                "seed": seed,
                "comparison": "RotatE minus TransE micro MRR",
                "point": float(paired.difference.mean()),
                "CI_low": float(np.quantile(differences, 0.025)),
                "CI_high": float(np.quantile(differences, 0.975)),
                "group": "disease",
            }
        )
    write_json(output / "paired_comparisons.json", comparisons)
    return rows
