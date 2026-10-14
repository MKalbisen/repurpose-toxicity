import argparse
import time

import pandas as pd

from repurpose.utils import ROOT, load_config, paths, write_json


def main():
    parser = argparse.ArgumentParser(description="PDF yol haritası: çevrimdışı araştırma pipeline")
    parser.add_argument("--config", default=str(ROOT / "configs/smoke.yaml"))
    parser.add_argument(
        "command",
        choices=[
            "download",
            "prepare-kg",
            "map-drugs",
            "train-kg",
            "prepare-tox21",
            "train-toxicity",
            "infer",
            "report",
            "run",
        ],
    )
    parser.add_argument("--disease", default=None)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["profile"] != "smoke" and args.command in {"run", "train-kg", "train-toxicity"}:
        raise ValueError(
            "Legacy full training lacks the academic test gate. Use: python -m repurpose.academic.experiment --help"
        )
    _, _, reports = paths(config)
    from importlib import import_module

    def invoke(module, function, with_config=True):
        fn = getattr(import_module(module), function)
        return fn(config) if with_config else fn()

    commands = {
        "download": lambda: invoke("repurpose.data.download", "download", False),
        "prepare-kg": lambda: invoke("repurpose.data.kg_data", "prepare_kg"),
        "map-drugs": lambda: invoke("repurpose.data.mapping", "map_drugs"),
        "train-kg": lambda: invoke("repurpose.kg.train", "train_kg"),
        "prepare-tox21": lambda: invoke("repurpose.data.tox21", "prepare_tox21"),
        "train-toxicity": lambda: invoke("repurpose.toxicity.train", "train_toxicity"),
        "infer": lambda: infer(config, args.disease),
        "report": lambda: invoke("repurpose.reporting", "generate_report"),
    }
    write_json(reports / "run_config.json", config)
    steps = list(commands) if args.command == "run" else [args.command]
    if args.skip_download and "download" in steps:
        steps.remove("download")
    for step in steps:
        print(f"[{config['profile']}] {step} başladı", flush=True)
        start = time.perf_counter()
        commands[step]()
        elapsed = time.perf_counter() - start
        path = reports / "experiment_log.csv"
        pd.DataFrame(
            [
                {
                    "step": step,
                    "seconds": elapsed,
                    "seed": config["seed"],
                    "profile": config["profile"],
                    "status": "completed",
                    "unix_time": time.time(),
                }
            ]
        ).to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"[{config['profile']}] {step} tamamlandı: {elapsed:.2f} sn", flush=True)


def infer(config, disease=None):
    from repurpose.integration.inference_pipeline import InferenceService

    service = InferenceService(config)
    if disease is None:
        # Pick most represented target disease, independent of test scores.
        indication = service.triples[service.triples.relation == "indication"]
        diseases = pd.concat([indication["head"], indication["tail"]])
        disease = diseases[diseases.str.startswith("disease:")].value_counts().index[0]
    ranked, seconds = service.analyze(
        disease, config["ranking"]["top_n"], config["ranking"]["threshold"], config["ranking"]["weight"]
    )
    service.export(ranked, seconds)
    write_json(
        service.reports / "inference_query.json", {"disease_id": disease, "profile": config["profile"]}
    )
    print(ranked[["drug_name", "kge_score", "assay_count", "max_risk", "final_rank"]].to_string(index=False))
