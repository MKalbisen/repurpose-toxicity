import numpy as np
import pandas as pd


def pareto_mask(benefit, risk):
    benefit, risk = np.asarray(benefit), np.asarray(risk)
    valid = np.isfinite(benefit) & np.isfinite(risk)
    keep = np.zeros(len(benefit), dtype=bool)
    for i in np.flatnonzero(valid):
        dominated = (
            valid & (benefit >= benefit[i]) & (risk <= risk[i]) & ((benefit > benefit[i]) | (risk < risk[i]))
        )
        keep[i] = not dominated.any()
    return keep


def rank_candidates(frame, threshold=0.5, weight=0.7, strategy="weighted"):
    if not 0 <= weight <= 1 or not 0 <= threshold <= 1:
        raise ValueError("Eşik ve ağırlık [0, 1] aralığında olmalıdır")
    out = frame.copy()
    values = out.kge_score.to_numpy(dtype=float)
    span = values.max() - values.min() if len(values) else 0
    out["normalized_kge"] = (values - values.min()) / span if span > 0 else np.full(len(values), 0.5)
    out["complete_risk"] = out["assay_count"] == 12
    out["high_risk_count"] = out.filter(regex="^risk_").ge(threshold).sum(axis=1)
    for column in out.filter(regex="^risk_").columns:
        out["flag_" + column.removeprefix("risk_")] = np.where(
            out[column].notna(), (out[column] >= threshold).astype(float), np.nan
        )
    out["eligible"] = out.complete_risk & (out.max_risk < threshold)
    out["exclusion_reason"] = np.where(
        ~out.complete_risk,
        "unknown_or_incomplete_assay_risk",
        np.where(out.max_risk >= threshold, "predicted_assay_risk_above_threshold", ""),
    )
    out["combined_score"] = weight * out.normalized_kge - (1 - weight) * out.max_risk
    out.loc[~out.complete_risk, "combined_score"] = np.nan
    out["pareto"] = pareto_mask(out.normalized_kge, out.max_risk.where(out.complete_risk))
    if strategy == "hard_filter":
        out = out.sort_values(["eligible", "kge_score", "drug_id"], ascending=[False, False, True])
    elif strategy == "pareto":
        out = out.sort_values(
            ["complete_risk", "pareto", "combined_score", "drug_id"],
            ascending=[False, False, False, True],
            na_position="last",
        )
    elif strategy == "weighted":
        out = out.sort_values(
            ["complete_risk", "combined_score", "drug_id"], ascending=[False, False, True], na_position="last"
        )
    else:
        raise ValueError(f"Bilinmeyen strateji: {strategy}")
    out["final_rank"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)


def sensitivity(frame, threshold):
    rows = []
    for weight in [0.0, 0.25, 0.5, 0.7, 0.75, 1.0]:
        ranked = rank_candidates(frame, threshold, weight)
        for r in ranked.head(10).itertuples():
            rows.append(
                {
                    "weight": weight,
                    "drug_id": r.drug_id,
                    "rank": r.final_rank,
                    "complete_risk": r.complete_risk,
                    "combined_score": r.combined_score,
                }
            )
    return pd.DataFrame(rows)
