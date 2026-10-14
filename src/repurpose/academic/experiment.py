"""Academic runner. Preparation, pilot, lock, training and final evaluation are explicit."""

import argparse
import json
import math
import platform
import subprocess
import time
import zipfile
from pathlib import Path

import psutil
import torch

from repurpose.academic.locking import file_lock
from repurpose.academic.prepare import prepare_chemistry, prepare_graph
from repurpose.academic.protocol import freeze, read_config
from repurpose.academic.state import atomic_json
from repurpose.utils import ROOT, seed_all


def pilot(config, output):
    from repurpose.academic.kg import graph_data, kg_step, make_model, read_targets, retrieval, triple_codes
    from repurpose.academic.toxicity import device_for, load_partition, train_mlp

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config["seeds"][0])
    torch.set_num_threads(config["threads"])
    device = device_for(config)
    base, _manifest, metadata, factory = graph_data(config)
    known = (
        torch.sort(triple_codes(factory.mapped_triples, factory.num_entities, factory.num_relations))[0]
        .unique()
        .to(device)
    )
    targets = read_targets(base, "validation")
    # Pilot validates the scoring adapter on <=100 pairs; never used for HPO/model choice.
    known_pairs = read_targets(base, "train") + targets
    rows = []
    for family in ["TransE", "RotatE"]:
        for dim in config["kg"]["dimensions"]:
            seed_all(config["seeds"][0])
            model = make_model(family, factory, dim, config["seeds"][0], device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["kg"]["learning_rates"][0])
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            order = torch.randperm(factory.num_triples)[: config["kg"]["batch_size"] * 60]
            batches = list(order.split(config["kg"]["batch_size"]))
            # Warm-up excluded from extrapolation.
            for batch in batches[:5]:
                kg_step(
                    model,
                    optimizer,
                    factory.mapped_triples[batch].to(device),
                    config["kg"],
                    known,
                    factory.num_entities,
                    factory.num_relations,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for batch in batches[5:]:
                kg_step(
                    model,
                    optimizer,
                    factory.mapped_triples[batch].to(device),
                    config["kg"],
                    known,
                    factory.num_entities,
                    factory.num_relations,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            per_batch = (time.perf_counter() - start) / (len(batches) - 5)
            eval_start = time.perf_counter()
            retrieval(model, metadata, targets[:100], known_pairs)
            evaluation_seconds = time.perf_counter() - eval_start
            epoch_seconds = per_batch * math.ceil(factory.num_triples / config["kg"]["batch_size"])
            all_validation_estimate = (
                evaluation_seconds * len({t for _, t in targets}) / len({t for _, t in targets[:100]})
            )
            rows.append(
                {
                    "family": family,
                    "dimension": dim,
                    "batch_seconds": per_batch,
                    "estimated_epoch_seconds": epoch_seconds,
                    "estimated_full_validation_seconds": all_validation_estimate,
                    "max_epochs_trial_hours": (
                        epoch_seconds * config["kg"]["epochs"]
                        + all_validation_estimate
                        * math.ceil(config["kg"]["epochs"] / config["kg"]["validation_every"])
                    )
                    / 3600,
                    "peak_cuda_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20
                    if device == "cuda"
                    else None,
                }
            )
            print(f"Pilot {family} dim={dim}: epoch ~{epoch_seconds:.1f}s", flush=True)
            del optimizer, model
            if device == "cuda":
                torch.cuda.empty_cache()
    # Two real epochs measure MLP costs on the final development partition.
    short = {**config, "toxicity": {**config["toxicity"], "mlp_epochs": 2}}
    mlp_dir = output / "mlp"
    mlp_dir.mkdir(exist_ok=True)
    start = time.perf_counter()
    train_mlp(
        short,
        load_partition(config, "train"),
        load_partition(config, "validation"),
        config["seeds"][0],
        mlp_dir,
    )
    mlp_seconds = time.perf_counter() - start
    total = (
        sum(r["max_epochs_trial_hours"] for r in rows)
        * len(config["kg"]["learning_rates"])
        * len(config["seeds"])
    )
    gpu = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        if device == "cuda"
        else "unavailable"
    )
    report = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device,
        "gpu": gpu,
        "RAM_GiB": psutil.virtual_memory().total / 2**30,
        "available_RAM_GiB": psutil.virtual_memory().available / 2**30,
        "train_triples": factory.num_triples,
        "entities": factory.num_entities,
        "trials": rows,
        "maximum_grid_estimated_hours": total,
        "MLP_two_epoch_wall_seconds": mlp_seconds,
        "recommendation": "external_gpu" if total > 8 else "local_feasible",
        "caveats": "Linear extrapolation from 55 batches, fixed hardware load; not a promise. Early stopping may shorten; I/O/checkpoint/thermal costs may lengthen. Validation query subsample extrapolation.",
        "test_accessed": False,
    }
    atomic_json(output / "hardware_pilot.json", report)
    return report


def export_package(config, run, destination):
    """Private compute transfer; raw data omitted, license metadata retained."""
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "requirements-academic-lock.txt",
        ROOT / "configs/academic_run.yaml",
        ROOT / "reports/data_manifest.json",
    ]
    paths += list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").glob("*.py"))
    paths += list((ROOT / "scripts").glob("*academic*"))
    paths += list((ROOT / "docs/academic").glob("*.md"))
    for domain in ["toxicity", "kg"]:
        base = ROOT / "data/processed" / config["dataset"] / domain
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        paths += [base / "manifest.json", *[base / name for name in manifest["files"]]]
    paths += [p for p in Path(run).rglob("*") if p.is_file() and p.suffix not in {".tmp", ".lock"}]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for p in sorted(set(paths)):
            archive.write(p, p.relative_to(ROOT).as_posix())
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "pilot",
            "freeze",
            "train-toxicity",
            "train-kg",
            "evaluate-toxicity",
            "evaluate-kg",
            "export",
        ],
    )
    parser.add_argument("--config", default=str(ROOT / "configs/academic_run.yaml"))
    parser.add_argument("--run", default=str(ROOT / "reports/academic/runs/academic-v2"))
    args = parser.parse_args()
    config, run = read_config(args.config), Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    with file_lock(run / ".execution.lock", blocking=False):
        if args.command == "prepare":
            base = ROOT / "data/processed" / config["dataset"]
            prepare_chemistry(base / "toxicity", seed=config["split_seed"])
            prepare_graph(base / "kg", seed=config["split_seed"])
        elif args.command == "pilot":
            pilot(config, run / "pilot")
        elif args.command == "freeze":
            freeze(config, run)
        elif args.command == "train-toxicity":
            from repurpose.academic.toxicity import train_toxicity

            train_toxicity(config, run)
        elif args.command == "train-kg":
            from repurpose.academic.kg import train_kg

            train_kg(config, run)
        elif args.command == "evaluate-toxicity":
            from repurpose.academic.toxicity import evaluate_toxicity

            evaluate_toxicity(config, run)
        elif args.command == "evaluate-kg":
            from repurpose.academic.kg import evaluate_kg

            evaluate_kg(config, run)
        else:
            print(export_package(config, run, ROOT / "exports/academic-v2-compute.zip"))


if __name__ == "__main__":
    main()
