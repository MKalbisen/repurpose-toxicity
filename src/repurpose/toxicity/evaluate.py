import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from repurpose.utils import TASKS


def binary_metrics(y, p, threshold=0.5, bootstrap=0, seed=42):
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = np.asarray(y)[mask], np.asarray(p)[mask]
    out = {"n": len(y), "positive": int(y.sum()), "status": "ok"}
    names = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Balanced Accuracy"]
    if len(np.unique(y)) < 2:
        out.update({name: np.nan for name in names})
        out.update(
            status="insufficient_classes",
            ROC_CI_low=np.nan,
            ROC_CI_high=np.nan,
            PR_CI_low=np.nan,
            PR_CI_high=np.nan,
        )
        return out
    predicted = p >= threshold
    out.update(
        {
            "ROC-AUC": roc_auc_score(y, p),
            "PR-AUC": average_precision_score(y, p),
            "Precision": precision_score(y, predicted, zero_division=0),
            "Recall": recall_score(y, predicted, zero_division=0),
            "F1": f1_score(y, predicted, zero_division=0),
            "Balanced Accuracy": balanced_accuracy_score(y, predicted),
        }
    )
    rng = np.random.default_rng(seed)
    aucs, aps = [], []
    # Ordinary molecule bootstrap; single-class draws are skipped and counted.
    for _ in range(bootstrap):
        ids = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ids])) == 2:
            aucs.append(roc_auc_score(y[ids], p[ids]))
            aps.append(average_precision_score(y[ids], p[ids]))
    out["bootstrap_valid"] = len(aucs)
    for prefix, samples in [("ROC", aucs), ("PR", aps)]:
        low, high = np.quantile(samples, [0.025, 0.975]) if samples else [np.nan, np.nan]
        out.update({f"{prefix}_CI_low": low, f"{prefix}_CI_high": high})
    return out


def evaluate_tasks(y, p, model, bootstrap=0, seed=42):
    return pd.DataFrame(
        [
            {
                "model": model,
                "assay": task,
                **binary_metrics(y[:, j], p[:, j], bootstrap=bootstrap, seed=seed + j),
            }
            for j, task in enumerate(TASKS)
        ]
    )


def macro_auc(y, p):
    scores = [binary_metrics(y[:, j], p[:, j])["ROC-AUC"] for j in range(len(TASKS))]
    return float(np.nanmean(scores)) if np.isfinite(scores).any() else float("-inf")
