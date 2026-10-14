"""Disease-query drug retrieval metrics and scaffold-aware paired uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def filtered_rank(scores, target, known_positives=()):
    """Expected rank under uniform random ordering of exact score ties."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Skorlar sonlu bir vektör olmalıdır")
    if not 0 <= target < len(values):
        raise ValueError("Hedef aday listesinde yok")
    keep = np.ones(len(values), dtype=bool)
    for i in known_positives:
        if not 0 <= i < len(values):
            raise ValueError("Pozitif indeks aday listesinde yok")
        if i != target:
            keep[i] = False
    better = np.count_nonzero((values > values[target]) & keep)
    tied = np.count_nonzero((values == values[target]) & keep)
    return float(1 + better + (tied - 1) / 2), int(keep.sum())


def random_expected_metrics(candidate_count, cutoffs=(1, 3, 10)):
    """Analytical random-ranking expectation, not a ties-as-random approximation."""
    if candidate_count < 1:
        raise ValueError("Aday sayısı pozitif olmalıdır")
    harmonic = np.sum(1 / np.arange(1, candidate_count + 1))
    return {
        "MRR": float(harmonic / candidate_count),
        **{f"Hits@{k}": min(k, candidate_count) / candidate_count for k in cutoffs},
    }


def drug_query_metrics(score_function, drug_ids, target_pairs, known_pairs, cutoffs=(1, 3, 10)):
    """Evaluate canonical (drug, disease) positives among drug-only candidates.

    score_function(disease_id, sorted_drug_ids) must return scores in exactly that order.
    known_pairs is an explicit filter scope chosen by the caller: train+validation
    for development, all known positives only after final-test protocol lock.
    Recall@K is unfiltered, query-level known-target recall, and is labeled separately.
    """
    candidates = sorted(set(drug_ids))
    if not candidates or any(not d.startswith("drug:") for d in candidates):
        raise ValueError("Aday havuzu yalnız drug kimliklerinden oluşmalıdır")
    pairs = sorted({tuple(p) for p in target_pairs})
    if not pairs:
        raise ValueError("Değerlendirilecek indication hedefi yok")
    if any(d not in candidates or not disease.startswith("disease:") for d, disease in pairs):
        raise ValueError("Hedef drug/disease kimlikleri geçersiz")
    lookup = {d: i for i, d in enumerate(candidates)}
    targets, known = {}, {}
    for drug, disease in pairs:
        targets.setdefault(disease, set()).add(drug)
    for drug, disease in known_pairs:
        if drug in lookup:
            known.setdefault(disease, set()).add(lookup[drug])
    rows, recalls = [], []
    for disease in sorted(targets):
        scores = np.asarray(score_function(disease, candidates), dtype=float)
        if scores.shape != (len(candidates),):
            raise ValueError("Skor sırası/şekli aday havuzuyla uyumsuz")
        for drug in sorted(targets[disease]):
            rank, remaining = filtered_rank(scores, lookup[drug], known.get(disease, ()))
            random = random_expected_metrics(remaining, cutoffs)
            rows.append(
                {
                    "drug_id": drug,
                    "disease_id": disease,
                    "rank": rank,
                    "candidate_count_after_filter": remaining,
                    "RR": 1 / rank,
                    "random_expected_RR": random["MRR"],
                    **{f"Hits@{k}": float(rank <= k) for k in cutoffs},
                    **{f"random_expected_Hits@{k}": random[f"Hits@{k}"] for k in cutoffs},
                }
            )
        for k in cutoffs:
            # Expected inclusion under uniform random tie breaking. Lexical drug IDs
            # must not decide recall for constant/popularity/shared-neighbor baselines.
            inclusion = []
            for drug in targets[disease]:
                value = scores[lookup[drug]]
                better = int(np.count_nonzero(scores > value))
                tied = int(np.count_nonzero(scores == value))
                inclusion.append(float(np.clip((k - better) / tied, 0, 1)))
            recalls.append(
                {
                    "disease_id": disease,
                    "k": k,
                    "unfiltered_target_Recall@K": float(np.mean(inclusion)),
                }
            )
    frame, recall = pd.DataFrame(rows), pd.DataFrame(recalls)
    macro_rr = frame.groupby("disease_id").RR.mean()
    summary = {
        "MRR": float(frame.RR.mean()),
        "disease_macro_MRR": float(macro_rr.mean()),
        "target_pair_count": len(frame),
        "disease_count": len(targets),
        "candidate_type": "drug_only",
        "filtered": True,
        "tie_policy": "midrank MRR/Hits (not expected reciprocal rank); expected random-tie unfiltered Recall@K",
        "random_expected_MRR": float(frame.random_expected_RR.mean()),
    }
    for k in cutoffs:
        summary[f"Hits@{k}"] = float(frame[f"Hits@{k}"].mean())
        summary[f"unfiltered_target_Recall@{k}"] = float(
            recall.loc[recall.k == k, "unfiltered_target_Recall@K"].mean()
        )
    return frame, recall, summary


def pykeen_drug_scorer(model, metadata, relation="indication", batch_size=4096):
    import torch

    if relation not in metadata["relation_to_id"]:
        raise ValueError("indication ilişkisi modelde yok")

    def score(disease, candidates):
        if disease not in metadata["entity_to_id"]:
            raise ValueError("Hastalık eğitimde yok")
        if any(d not in metadata["entity_to_id"] for d in candidates):
            raise ValueError("İlaç eğitimde yok")
        triples = [
            [
                metadata["entity_to_id"][d],
                metadata["relation_to_id"][relation],
                metadata["entity_to_id"][disease],
            ]
            for d in candidates
        ]
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(triples), batch_size):
                batch = torch.tensor(
                    triples[start : start + batch_size], dtype=torch.long, device=model.device
                )
                chunks.append(model.score_hrt(batch).detach().cpu().numpy().ravel())
        return np.concatenate(chunks)

    return score


def canonical_indication_pairs(triples):
    pairs = set()
    for h, relation, t in triples.itertuples(index=False, name=None):
        if relation != "indication":
            continue
        if h.startswith("drug:") and t.startswith("disease:"):
            pairs.add((h, t))
        elif t.startswith("drug:") and h.startswith("disease:"):
            pairs.add((t, h))
        else:
            raise ValueError("indication ilişkisinde beklenmeyen entity türü")
    return sorted(pairs)


def _metric(y, p, metric):
    if metric == "ROC-AUC":
        return float(roc_auc_score(y, p))
    if metric == "PR-AUC":
        return float(average_precision_score(y, p))
    raise ValueError("Metric ROC-AUC veya PR-AUC olmalıdır")


def grouped_bootstrap(y, p, groups, metric="ROC-AUC", repeats=2000, seed=2026, paired_predictions=None):
    """Sample scaffold clusters, preserving within-group observations and model pairing.

    Single-class resamples are skipped, never silently counted as zero. This is a
    conditional model CI for a fixed trained model, not a CI over training seeds.
    Paired comparisons use the common finite-label/prediction rows of both models.
    """
    y, p, groups = np.asarray(y, dtype=float), np.asarray(p, dtype=float), np.asarray(groups, dtype=object)
    if y.ndim != 1 or p.shape != y.shape or groups.shape != y.shape:
        raise ValueError("Etiket/tahmin/grup vektörleri aynı boyutta olmalıdır")
    if repeats < 1:
        raise ValueError("Bootstrap sayısı pozitif olmalıdır")
    if pd.isna(groups).any():
        raise ValueError("Grup kimliği eksik olamaz")
    if metric not in {"ROC-AUC", "PR-AUC"}:
        raise ValueError("Metric ROC-AUC veya PR-AUC olmalıdır")
    paired = None if paired_predictions is None else np.asarray(paired_predictions, dtype=float)
    if paired is not None and paired.shape != y.shape:
        raise ValueError("Eşlenmiş tahminler aynı boyutta olmalıdır")
    mask = np.isfinite(y) & np.isfinite(p)
    if paired is not None:
        mask &= np.isfinite(paired)
    y, p, groups = y[mask], p[mask], groups[mask]
    paired = paired[mask] if paired is not None else None
    if not np.isin(y, [0, 1]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("İkili etiket ve [0,1] olasılık gerekli")
    if paired is not None and ((paired < 0) | (paired > 1)).any():
        raise ValueError("Eşlenmiş olasılıklar [0,1] aralığında olmalıdır")
    keys = sorted(set(groups), key=str)
    group_indices = [np.flatnonzero(groups == key) for key in keys]
    out = {
        "metric": metric,
        "n": len(y),
        "group_count": len(keys),
        "requested_repeats": repeats,
        "valid_repeats": 0,
        "skipped_single_class": 0,
        "seed": seed,
        "paired": paired is not None,
        "point": None,
        "CI_low": None,
        "CI_high": None,
        "status": "ok",
        "scope": "fixed-model cluster percentile CI; paired effect = first model minus second",
    }
    if len(np.unique(y)) < 2:
        out["status"] = "insufficient_classes"
        return out
    point = _metric(y, p, metric)
    out["point"] = point - _metric(y, paired, metric) if paired is not None else point
    if len(keys) < 2:
        out["status"] = "insufficient_groups"
        return out
    rng, samples = np.random.default_rng(seed), []
    for _ in range(repeats):
        sampled = rng.integers(0, len(keys), len(keys))
        ids = np.concatenate([group_indices[i] for i in sampled])
        if len(np.unique(y[ids])) < 2:
            out["skipped_single_class"] += 1
            continue
        value = _metric(y[ids], p[ids], metric)
        if paired is not None:
            value -= _metric(y[ids], paired[ids], metric)
        samples.append(value)
    out["valid_repeats"] = len(samples)
    if samples:
        out["CI_low"], out["CI_high"] = map(float, np.quantile(samples, [0.025, 0.975]))
        if out["valid_repeats"] < 0.8 * repeats:
            out["status"] = "unstable_resampling"
    else:
        out["status"] = "no_valid_resamples"
    return out


def macro_grouped_bootstrap(y, p, groups, repeats=2000, seed=2026, paired_predictions=None):
    """Fixed task set, shared cluster draws; never silently vary macro denominator."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    paired = None if paired_predictions is None else np.asarray(paired_predictions)
    if y.ndim != 2 or y.shape != p.shape or len(groups) != len(y):
        raise ValueError("Macro label/prediction/group shapes differ")
    if paired is not None and paired.shape != y.shape:
        raise ValueError("Paired prediction shape differs")
    masks = np.isfinite(y) & np.isfinite(p)
    if paired is not None:
        masks &= np.isfinite(paired)
    tasks = [j for j in range(y.shape[1]) if len(np.unique(y[masks[:, j], j])) == 2]
    result = {
        "task_indices": tasks,
        "evaluable_tasks": len(tasks),
        "requested_tasks": y.shape[1],
        "requested_repeats": repeats,
        "valid_repeats": 0,
        "skipped_single_class": 0,
        "point": None,
        "CI_low": None,
        "CI_high": None,
        "paired": paired is not None,
        "status": "ok" if len(tasks) == y.shape[1] else "incomplete_task_set",
    }
    if not tasks:
        return result

    def score(ids):
        values = []
        for j in tasks:
            selected = ids[masks[ids, j]]
            if len(np.unique(y[selected, j])) < 2:
                return None
            value = roc_auc_score(y[selected, j], p[selected, j])
            if paired is not None:
                value -= roc_auc_score(y[selected, j], paired[selected, j])
            values.append(value)
        return float(np.mean(values))

    result["point"] = score(np.arange(len(y)))
    keys = np.unique(groups)
    if len(keys) < 2:
        result["status"] = "insufficient_groups"
        return result
    indices = [np.flatnonzero(groups == key) for key in keys]
    rng, samples = np.random.default_rng(seed), []
    for _ in range(repeats):
        ids = np.concatenate([indices[i] for i in rng.integers(0, len(keys), len(keys))])
        value = score(ids)
        if value is None:
            result["skipped_single_class"] += 1
        else:
            samples.append(value)
    result["valid_repeats"] = len(samples)
    if samples:
        result["CI_low"], result["CI_high"] = map(float, np.quantile(samples, [0.025, 0.975]))
    if len(samples) < 0.8 * repeats:
        result["status"] = "unstable_resampling"
    return result
