import itertools
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from repurpose.toxicity.evaluate import binary_metrics, evaluate_tasks, macro_auc
from repurpose.toxicity.models import MultiTaskMLP, masked_bce
from repurpose.utils import TASKS, paths, seed_all, write_json


def train_toxicity(config):
    seed_all(config["seed"])
    base, models, reports = paths(config)
    data = np.load(base / "tox21_features.npz")
    x, y = data["X"].astype(np.float32), data["Y"]
    train, val, test = data["train"], data["val"], data["test"]
    output = models / "toxicity"
    xgb_dir = output / "xgb_models"
    xgb_dir.mkdir(parents=True, exist_ok=True)
    predictions = np.full((len(test), 12), np.nan)
    trials, manifest = [], {}
    for j, task in enumerate(TASKS):
        ti = train[np.isfinite(y[train, j])]
        vi = val[np.isfinite(y[val, j])]
        if len(np.unique(y[ti, j])) < 2 or len(np.unique(y[vi, j])) < 2:
            manifest[task] = {"status": "insufficient_train_or_validation_classes"}
            continue
        negatives = int((y[ti, j] == 0).sum())
        positives = int((y[ti, j] == 1).sum())
        best = None
        for estimators, depth in itertools.product(
            config["toxicity"]["xgb_estimators"], config["toxicity"]["xgb_depths"]
        ):
            clf = XGBClassifier(
                n_estimators=estimators,
                max_depth=depth,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=negatives / max(1, positives),
                random_state=config["seed"],
                tree_method="hist",
                n_jobs=4,
                eval_metric="aucpr",
            )
            clf.fit(x[ti], y[ti, j], eval_set=[(x[vi], y[vi, j])], verbose=False)
            ap = binary_metrics(y[vi, j], clf.predict_proba(x[vi])[:, 1])["PR-AUC"]
            trials.append({"assay": task, "estimators": estimators, "depth": depth, "validation_PR-AUC": ap})
            if best is None or ap > best[0]:
                best = (ap, clf, estimators, depth)
        ap, clf, estimators, depth = best
        clf.save_model(xgb_dir / f"assay_{task}.json")
        predictions[:, j] = clf.predict_proba(x[test])[:, 1]
        manifest[task] = {
            "status": "trained",
            "estimators": estimators,
            "depth": depth,
            "validation_PR-AUC": ap,
            "scale_pos_weight": negatives / positives,
        }
    write_json(output / "xgb_manifest.json", manifest)
    pd.DataFrame(trials).to_csv(reports / "xgb_trials.csv", index=False)
    xm = evaluate_tasks(y[test], predictions, "XGBoost", config["toxicity"]["bootstrap"], config["seed"])
    xm.to_csv(reports / "xgb_metrics.csv", index=False)
    _save_predictions(base, reports / "xgb_predictions.csv", test, predictions)
    # Multi-task MLP: class weights computed from train labels only.
    model = MultiTaskMLP().to(config["device"])
    positives = np.nansum(y[train], axis=0)
    observed = np.isfinite(y[train]).sum(axis=0)
    valid_tasks = (positives > 0) & (positives < observed)
    pos_weight = torch.tensor(
        (observed - positives) / np.maximum(positives, 1), dtype=torch.float32, device=config["device"]
    )
    mlp_y = y.copy()
    mlp_y[:, ~valid_tasks] = np.nan
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(mlp_y[train])),
        batch_size=config["toxicity"]["batch_size"],
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history, best_score, patience = [], float("-inf"), 0
    checkpoint = output / "mlp_best.pt"
    start = time.perf_counter()
    for epoch in range(config["toxicity"]["mlp_epochs"]):
        model.train()
        total, batches = 0.0, 0
        for xb, yb in loader:
            if len(xb) < 2:  # BatchNorm requires >1 training example.
                continue
            xb, yb = xb.to(config["device"]), yb.to(config["device"])
            optimizer.zero_grad()
            loss = masked_bce(model(xb), yb, pos_weight)
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        model.eval()
        with torch.no_grad():
            p = model(torch.tensor(x[val], device=config["device"])).sigmoid().cpu().numpy()
        p[:, ~valid_tasks] = np.nan
        score = macro_auc(y[val], p)
        history.append(
            {"epoch": epoch + 1, "train_loss": total / max(1, batches), "validation_ROC-AUC": score}
        )
        if score > best_score:
            best_score, patience = score, 0
            torch.save(
                {
                    "state_dict": model.cpu().state_dict(),
                    "valid_tasks": valid_tasks.tolist(),
                    "seed": config["seed"],
                    "validation_ROC-AUC": score,
                    "epoch": epoch + 1,
                },
                checkpoint,
            )
            model.to(config["device"])
        else:
            patience += 1
        if patience >= 15:
            break
    if not checkpoint.exists():
        raise ValueError("MLP validation için yeterli etiket yok; model seçimi yapılamadı")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = MultiTaskMLP()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with torch.no_grad():
        mp = model(torch.from_numpy(x[test])).sigmoid().numpy()
    mp[:, ~valid_tasks] = np.nan
    mm = evaluate_tasks(y[test], mp, "MLP", config["toxicity"]["bootstrap"], config["seed"])
    mm.to_csv(reports / "mlp_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(reports / "training_history.csv", index=False)
    _save_predictions(base, reports / "mlp_predictions.csv", test, mp)
    metrics = pd.concat([xm, mm], ignore_index=True)
    metrics.to_csv(reports / "toxicity_final_metrics.csv", index=False)
    comparison = metrics.groupby("model")[["ROC-AUC", "PR-AUC", "F1", "Balanced Accuracy"]].mean()
    comparison["evaluable_assays"] = metrics.groupby("model")["ROC-AUC"].count()
    comparison.to_csv(reports / "model_comparison.csv")
    write_json(
        reports / "toxicity_training.json",
        {
            "mlp_seconds": time.perf_counter() - start,
            "mlp_best_epoch": saved["epoch"],
            "mlp_validation_ROC-AUC": best_score,
            "selection": "XGBoost per-assay validation AP; MLP validation macro ROC-AUC",
            "test_used_for_selection": False,
        },
    )
    (reports / "evaluation_report.md").write_text(
        "# Toksisite değerlendirmesi\n\nPR-AUC burada average precision olarak hesaplanır. "
        "Eksik etiketler maskelenir. ROC/PR-AUC tek sınıflı görevlerde tanımsız (NaN); "
        "bootstrap %95 percentile, tek sınıflı çekimler atlanır. Eşik 0.5; "
        "olasılık kalibrasyonu uygulanmadı. CI molekül bazlıdır, scaffold-kümeli CI değildir.\n\n"
        + comparison.to_string()
        + "\n\nEn düşük ve en yüksek test ROC-AUC:\n"
        + metrics.sort_values("ROC-AUC").iloc[[0, -1]][["model", "assay", "ROC-AUC"]].to_string(index=False),
        encoding="utf-8",
    )


def _save_predictions(base, dest, indices, p):
    frame = pd.DataFrame(p, columns=TASKS)
    clean = pd.read_csv(base / "tox21_clean.csv")
    frame.insert(0, "smiles", clean.iloc[indices].smiles.to_numpy())
    frame.to_csv(dest, index=False)
