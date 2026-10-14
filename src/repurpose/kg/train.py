"""Validation-selected PyKEEN TransE/RotatE experiments with filtered evaluation."""

import itertools
import time

import pandas as pd
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from repurpose.utils import paths, seed_all, write_json


def factories(base):
    def read(name):
        return pd.read_csv(base / f"{name}.tsv", sep="\t", header=None, dtype=str).to_numpy()

    training = TriplesFactory.from_labeled_triples(read("train"), create_inverse_triples=False)
    others = [
        TriplesFactory.from_labeled_triples(
            read(name),
            entity_to_id=training.entity_to_id,
            relation_to_id=training.relation_to_id,
            create_inverse_triples=False,
        )
        for name in ["validation", "test"]
    ]
    return training, *others


def metric_row(result):
    return {
        "MRR": result.get_metric("both.realistic.inverse_harmonic_mean_rank"),
        **{f"Hits@{k}": result.get_metric(f"both.realistic.hits_at_{k}") for k in [1, 3, 10]},
    }


def evaluate(model, target, filters):
    return RankBasedEvaluator(filtered=True).evaluate(
        model=model,
        mapped_triples=target.mapped_triples,
        additional_filter_triples=[f.mapped_triples for f in filters],
        batch_size=128,
        use_tqdm=False,
    )


def train_kg(config):
    seed_all(config["seed"])
    base, models, reports = paths(config)
    training, validation, testing = factories(base)
    rows, histories, selected = [], [], []
    for family in ["TransE", "RotatE"]:
        best = None
        for dim, lr in itertools.product(config["kg"]["dimensions"], config["kg"]["learning_rates"]):
            start = time.perf_counter()
            # use_testing_data=False means every trial is evaluated on validation only.
            result = pipeline(
                training=training,
                validation=validation,
                testing=testing,
                model=family,
                model_kwargs={"embedding_dim": dim},
                optimizer_kwargs={"lr": lr},
                random_seed=config["seed"],
                device=config["device"],
                training_kwargs={
                    "num_epochs": config["kg"]["epochs"],
                    "batch_size": min(config["kg"]["batch_size"], training.num_triples),
                },
                stopper="early",
                stopper_kwargs={
                    "frequency": min(5, config["kg"]["epochs"]),
                    "patience": 3,
                    "metric": "inverse_harmonic_mean_rank",
                    "relative_delta": 0.001,
                },
                evaluator_kwargs={"filtered": True},
                evaluation_kwargs={"batch_size": 128},
                use_testing_data=False,
                use_tqdm=False,
            )
            metrics = metric_row(result.metric_results)
            rows.append(
                {
                    "model": family,
                    "dimension": dim,
                    "lr": lr,
                    "split": "validation",
                    "seconds": time.perf_counter() - start,
                    **metrics,
                }
            )
            histories.append(
                {"model": family, "dimension": dim, "lr": lr, "losses": result.losses, "seed": config["seed"]}
            )
            if best is None or metrics["MRR"] > best[0]:
                best = (metrics["MRR"], result, dim, lr)
        score, result, dim, lr = best
        dest = models / "kg" / f"{family.lower()}_model"
        dest.mkdir(parents=True, exist_ok=True)
        result.save_to_directory(dest)
        # Checkpoint is local trusted output; never deserialize user-supplied models.
        torch.save(result.model.cpu(), dest / "inference.pt")
        test_metrics = metric_row(evaluate(result.model, testing, [training, validation]))
        row = {
            "model": family,
            "dimension": dim,
            "lr": lr,
            "split": "test",
            **test_metrics,
            "validation_MRR": score,
        }
        selected.append(row)
        pd.DataFrame([row]).to_csv(reports / f"kg_metrics_{family.lower()}.csv", index=False)
    winner = max(selected, key=lambda r: r["validation_MRR"])
    write_json(
        models / "kg" / "best_kge_config.json",
        {
            **winner,
            "selection": "validation MRR",
            "seed": config["seed"],
            "model_path": f"{winner['model'].lower()}_model/inference.pt",
            "entity_to_id": training.entity_to_id,
            "relation_to_id": training.relation_to_id,
        },
    )
    pd.DataFrame(rows).to_csv(reports / "kge_trials.csv", index=False)
    pd.DataFrame(selected).to_csv(reports / "kge_comparison.csv", index=False)
    write_json(reports / "training_log.json", histories)
