"""Development-only baselines/MLP training; separate locked final evaluation."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from repurpose.academic.metrics import grouped_bootstrap, macro_grouped_bootstrap
from repurpose.academic.protocol import final_access, verify_lock
from repurpose.academic.state import atomic_json
from repurpose.toxicity.evaluate import binary_metrics
from repurpose.toxicity.models import MultiTaskMLP, masked_bce
from repurpose.utils import ROOT, TASKS, seed_all, sha256, write_json

FAMILIES = ["prevalence", "logistic", "xgboost", "mlp"]


def device_for(config):
    return (
        ("cuda" if torch.cuda.is_available() else "cpu") if config["device"] == "auto" else config["device"]
    )


def load_partition(config, name):
    with np.load(ROOT / "data/processed" / config["dataset"] / "toxicity" / f"{name}.npz") as data:
        return {k: data[k] for k in data.files}


def metric(y, p, kind="auc"):
    mask = np.isfinite(y) & np.isfinite(p)
    if len(np.unique(y[mask])) != 2:
        return np.nan
    return float((roc_auc_score if kind == "auc" else average_precision_score)(y[mask], p[mask]))


def macro(y, p):
    values = np.array([metric(y[:, j], p[:, j]) for j in range(y.shape[1])])
    return float(np.nanmean(values)) if np.isfinite(values).any() else -np.inf


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_calibration(y, p):
    params = []
    for j in range(y.shape[1]):
        mask = np.isfinite(y[:, j]) & np.isfinite(p[:, j])
        counts = np.bincount(y[mask, j].astype(int), minlength=2)
        row = {"negative": int(counts[0]), "positive": int(counts[1]), "status": "insufficient_classes"}
        if min(counts) >= 5:
            fit = LogisticRegression(C=1.0, max_iter=1000).fit(logit(p[mask, j])[:, None], y[mask, j])
            row.update(status="fitted", slope=float(fit.coef_[0, 0]), intercept=float(fit.intercept_[0]))
        params.append(row)
    return params


def calibrate(p, params):
    out = p.copy()
    for j, param in enumerate(params):
        if param["status"] == "fitted":
            z = np.clip(param["slope"] * logit(p[:, j]) + param["intercept"], -40, 40)
            out[:, j] = 1 / (1 + np.exp(-z))
    return out


def torch_save_atomic(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train_mlp(config, train, validation, seed, dest):
    options = config["toxicity"]
    device = device_for(config)
    seed_all(seed)
    torch.set_num_threads(config["threads"])
    model = MultiTaskMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    y = train["Y"].astype(np.float32).copy()
    positive = np.nansum(y, axis=0)
    observed = np.isfinite(y).sum(axis=0)
    valid = (positive > 0) & (positive < observed)
    y[:, ~valid] = np.nan
    weights = torch.tensor(
        (observed - positive) / np.maximum(positive, 1), dtype=torch.float32, device=device
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(train["X"], dtype=torch.float32), torch.from_numpy(y)),
        batch_size=options["batch_size"],
        shuffle=True,
        generator=generator,
    )
    history, epoch, best, stale, best_state = [], 0, -np.inf, 0, None
    checkpoint = dest / "mlp_resume.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, best, stale, best_state = saved["epoch"], saved["best"], saved["stale"], saved["best_state"]
        history = saved["history"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        generator.set_state(saved["loader_rng"].cpu())
        if device == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
    start = time.perf_counter()
    while epoch < options["mlp_epochs"] and stale < options["patience"]:
        model.train()
        losses = []
        for x, yb in loader:
            if len(x) == 1:
                continue  # BatchNorm singleton; never interpreted as missing labels.
            optimizer.zero_grad()
            loss = masked_bce(model(x.to(device)), yb.to(device), weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            p = (
                model(torch.tensor(validation["X"], dtype=torch.float32, device=device))
                .sigmoid()
                .cpu()
                .numpy()
            )
        p[:, ~valid] = np.nan
        score = macro(validation["Y"], p)
        epoch += 1
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_macro_auc": score})
        if score > best:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        torch_save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best": best,
                "stale": stale,
                "best_state": best_state,
                "history": history,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "loader_rng": generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            },
            checkpoint,
        )
        print(f"MLP seed={seed} epoch={epoch} validation AUC={score:.4f}", flush=True)
    if best_state is None:
        raise ValueError("MLP has no evaluable validation task")
    torch_save_atomic({"state_dict": best_state, "valid_tasks": valid.tolist()}, dest / "mlp.pt")
    write_json(
        dest / "mlp_history.json",
        {"history": history, "seconds_this_invocation": time.perf_counter() - start},
    )
    return {"kind": "mlp", "path": "mlp.pt", "validation_auc": best}


def predict(bundle, x, directory):
    if bundle["kind"] == "mlp":
        saved = torch.load(Path(directory) / bundle["path"], map_location="cpu", weights_only=True)
        model = MultiTaskMLP()
        model.load_state_dict(saved["state_dict"])
        model.eval()
        with torch.no_grad():
            p = model(torch.tensor(x, dtype=torch.float32)).sigmoid().numpy()
        p[:, ~np.array(saved["valid_tasks"])] = np.nan
        return p
    p = np.full((len(x), len(TASKS)), np.nan)
    for j, estimator in enumerate(bundle["estimators"]):
        if estimator is None:
            continue
        p[:, j] = estimator if isinstance(estimator, float) else estimator.predict_proba(x)[:, 1]
    return p


def train_toxicity(config, run):
    verify_lock(config, run)
    if (Path(run) / "toxicity_test_access.json").exists():
        raise ValueError("Training cannot run after final test access")
    train, val, calibration = [load_partition(config, n) for n in ["train", "validation", "calibration"]]
    options = config["toxicity"]
    for seed in config["seeds"]:
        seed_all(seed)
        torch.set_num_threads(config["threads"])
        dest = Path(run) / "toxicity" / str(seed)
        dest.mkdir(parents=True, exist_ok=True)
        complete = dest / "complete.json"
        if complete.exists():
            verify_complete(dest)
            continue
        start = time.perf_counter()
        for family in FAMILIES:
            bundle_path = dest / f"{family}.joblib"
            if bundle_path.exists():
                continue
            if family == "mlp":
                bundle = train_mlp(config, train, val, seed, dest)
            else:
                estimators, trials = [], []
                for j, task in enumerate(TASKS):
                    mask = np.isfinite(train["Y"][:, j])
                    y, x = train["Y"][mask, j], train["X"][mask]
                    if not len(y):
                        estimators.append(None)
                        continue
                    if family == "prevalence" or len(np.unique(y)) < 2:
                        estimators.append(float(y.mean()))
                        continue
                    best_score, best_model = -np.inf, None
                    choices = options["logistic_C"] if family == "logistic" else options["xgb_estimators"]
                    for setting in choices:
                        if family == "logistic":
                            estimator = LogisticRegression(C=setting, max_iter=2000, random_state=seed)
                        else:
                            estimator = XGBClassifier(
                                n_estimators=setting,
                                max_depth=options["xgb_depth"],
                                learning_rate=0.05,
                                tree_method="hist",
                                n_jobs=config["threads"],
                                subsample=0.9,
                                colsample_bytree=0.9,
                                random_state=seed,
                                scale_pos_weight=float((y == 0).sum() / (y == 1).sum()),
                            )
                        estimator.fit(x, y)
                        p = estimator.predict_proba(val["X"])[:, 1]
                        score = metric(val["Y"][:, j], p)
                        trials.append({"assay": task, "setting": setting, "validation_auc": score})
                        # Missing validation class: deterministic first preset, explicitly recorded.
                        if best_model is None or (np.isfinite(score) and score > best_score):
                            best_model = estimator
                            best_score = score if np.isfinite(score) else -np.inf
                    estimators.append(best_model)
                bundle = {"kind": family, "estimators": estimators, "trials": trials}
            raw = predict(bundle, calibration["X"], dest)
            bundle["calibration"] = (
                fit_calibration(calibration["Y"], raw)
                if family != "prevalence"
                else [{"status": "train_prevalence_baseline"} for _ in TASKS]
            )
            # Atomic family completion; an interrupted family is retrained, never read partially.
            temporary = bundle_path.with_suffix(".tmp")
            joblib.dump(bundle, temporary)
            temporary.replace(bundle_path)
            print(f"Toxicity seed={seed} family={family} development complete", flush=True)
        files = {p.name: sha256(p) for p in dest.iterdir() if p.suffix == ".joblib" or p.name == "mlp.pt"}
        atomic_json(
            complete,
            {
                "seed": seed,
                "files": files,
                "seconds": time.perf_counter() - start,
                "selection": "validation ROC-AUC; calibration held-out groups; threshold 0.5",
                "test_accessed": False,
            },
        )


def verify_complete(directory):
    directory = Path(directory)
    record = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    for name, expected in record["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Trained artifact changed: {name}")
    return record


def evaluate_toxicity(config, run):
    run = Path(run)
    artifacts = []
    for seed in config["seeds"]:
        directory = run / "toxicity" / str(seed)
        record = verify_complete(directory)
        artifacts.extend(directory / f for f in ["complete.json", *record["files"]])
    final_access(config, run, "toxicity", artifacts)
    test = load_partition(config, "test")
    output = run / "evaluation/toxicity"
    output.mkdir(parents=True, exist_ok=True)
    rows, predictions, reliability = [], {}, []
    for family in FAMILIES:
        predictions[family] = []
        for seed in config["seeds"]:
            directory = run / "toxicity" / str(seed)
            bundle = joblib.load(directory / f"{family}.joblib")  # local trusted artifact, hash checked above
            raw = predict(bundle, test["X"], directory)
            p = calibrate(raw, bundle["calibration"])
            predictions[family].append(p)
            np.savez_compressed(output / f"{family}_{seed}.npz", raw=raw, calibrated=p, rows=test["rows"])
            for j, task in enumerate(TASKS):
                mask = np.isfinite(test["Y"][:, j]) & np.isfinite(p[:, j])
                y, pred = test["Y"][mask, j], p[mask, j]
                classification = binary_metrics(y, pred, threshold=config["toxicity"].get("threshold", 0.5))
                for lower in np.arange(0, 1, 0.1):
                    in_bin = (pred >= lower) & (pred < lower + 0.1 if lower < 0.89 else pred <= 1)
                    reliability.append(
                        {
                            "family": family,
                            "seed": seed,
                            "assay": task,
                            "lower": float(lower),
                            "n": int(in_bin.sum()),
                            "mean_prediction": float(pred[in_bin].mean()) if in_bin.any() else np.nan,
                            "observed_fraction": float(y[in_bin].mean()) if in_bin.any() else np.nan,
                        }
                    )
                rows.append(
                    {
                        **{
                            k: classification[k]
                            for k in ["Precision", "Recall", "F1", "Balanced Accuracy", "status"]
                        },
                        "family": family,
                        "seed": seed,
                        "assay": task,
                        "n": len(y),
                        "positive": int(y.sum()),
                        "ROC-AUC": metric(y, pred),
                        "AP": metric(y, pred, "ap"),
                        "raw_ROC-AUC": metric(test["Y"][:, j], raw[:, j]),
                        "raw_Brier": float(brier_score_loss(y, raw[mask, j])) if len(y) else np.nan,
                        "Brier": float(brier_score_loss(y, pred)) if len(y) else np.nan,
                        "log_loss": float(log_loss(y, pred, labels=[0, 1])) if len(y) else np.nan,
                        "calibration_status": bundle["calibration"][j]["status"],
                    }
                )
    frame = pd.DataFrame(rows)
    pd.DataFrame(reliability).to_csv(output / "reliability_bins.csv", index=False)
    frame.to_csv(output / "assay_metrics.csv", index=False)
    summary = frame.groupby(["family", "seed"])[["ROC-AUC", "AP", "Brier", "log_loss"]].mean()
    summary["evaluable_assays"] = frame.groupby(["family", "seed"])["ROC-AUC"].count()
    summary["macro_12_ROC-AUC"] = summary["ROC-AUC"].where(summary.evaluable_assays == 12)
    summary.to_csv(output / "seed_metrics.csv")
    summary.groupby("family").agg(["mean", "std"]).to_csv(output / "seed_distribution.csv")
    # A predeclared seed-ensemble is a single fixed predictor; do not count seeds as independent data.
    ensembles = {f: np.mean(ps, axis=0) for f, ps in predictions.items()}
    uncertainty = []
    for family, p in ensembles.items():
        for j, task in enumerate(TASKS):
            for name in ["ROC-AUC", "PR-AUC"]:
                ci = grouped_bootstrap(
                    test["Y"][:, j],
                    p[:, j],
                    test["groups"],
                    metric=name,
                    repeats=config["uncertainty"]["bootstrap_repeats"],
                )
                uncertainty.append({"family": family, "assay": task, "comparison": "ensemble", **ci})
                if family in {"xgboost", "mlp"}:
                    ci = grouped_bootstrap(
                        test["Y"][:, j],
                        p[:, j],
                        test["groups"],
                        metric=name,
                        repeats=config["uncertainty"]["bootstrap_repeats"],
                        paired_predictions=ensembles["logistic"][:, j],
                    )
                    uncertainty.append(
                        {"family": family, "assay": task, "comparison": "minus_logistic", **ci}
                    )
    write_json(output / "cluster_uncertainty.json", uncertainty)
    macro_uncertainty = []
    for family, p in ensembles.items():
        for paired in [False, True] if family in {"xgboost", "mlp"} else [False]:
            ci = macro_grouped_bootstrap(
                test["Y"],
                p,
                test["groups"],
                repeats=config["uncertainty"]["bootstrap_repeats"],
                paired_predictions=ensembles["logistic"] if paired else None,
            )
            macro_uncertainty.append(
                {"family": family, "comparison": "minus_logistic" if paired else "ensemble", **ci}
            )
    write_json(output / "macro_cluster_uncertainty.json", macro_uncertainty)
    return summary
