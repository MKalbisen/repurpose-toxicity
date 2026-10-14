import numpy as np
import pandas as pd
import pytest

from repurpose.academic.metrics import (
    canonical_indication_pairs,
    drug_query_metrics,
    filtered_rank,
    grouped_bootstrap,
    macro_grouped_bootstrap,
    random_expected_metrics,
)


def test_filter_keeps_target_and_removes_other_true_links():
    rank, n = filtered_rank([0.9, 0.8, 0.7], 1, known_positives=[0, 1])
    assert rank == 1 and n == 2
    tied_rank, _ = filtered_rank([1.0, 1.0, 0.0], 0)
    assert tied_rank == 1.5


def test_canonical_inverse_targets_count_once():
    triples = pd.DataFrame(
        [
            ("drug:a", "indication", "disease:x"),
            ("disease:x", "indication", "drug:a"),
            ("drug:b", "contraindication", "disease:x"),
        ],
        columns=["head", "relation", "tail"],
    )
    assert canonical_indication_pairs(triples) == [("drug:a", "disease:x")]


def test_drug_pool_filter_and_disease_macro_weighting():
    scores = {"disease:x": [0.9, 0.8, 0.7], "disease:y": [0.1, 0.2, 0.3]}
    targets = [("drug:b", "disease:x"), ("drug:c", "disease:x"), ("drug:a", "disease:y")]
    known = targets + [("drug:a", "disease:x")]
    ranks, _, summary = drug_query_metrics(
        lambda disease, drugs: scores[disease], ["drug:c", "drug:a", "drug:b"], targets, known
    )
    # Canonical sorted order a,b,c. For x all other positives are filtered.
    assert ranks.loc[ranks.disease_id == "disease:x", "rank"].tolist() == [1.0, 1.0]
    assert summary["MRR"] == pytest.approx(7 / 9)
    assert summary["disease_macro_MRR"] == pytest.approx(2 / 3)
    assert summary["candidate_type"] == "drug_only"
    with pytest.raises(ValueError, match="yalnız drug"):
        drug_query_metrics(lambda d, c: [1, 2], ["gene:1", "drug:a"], targets, known)


def test_random_rank_expectation_is_harmonic_not_rank_of_ties():
    result = random_expected_metrics(3)
    assert result["MRR"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
    assert result["Hits@1"] == pytest.approx(1 / 3)
    assert result["Hits@10"] == 1


def test_grouped_resamples_preserve_clusters_and_report_single_class_skips():
    # Two pure-class clusters: selecting the same one twice must be skipped.
    result = grouped_bootstrap([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["A", "A", "B", "B"], repeats=100, seed=3)
    assert 0 < result["skipped_single_class"] < 100
    assert result["valid_repeats"] + result["skipped_single_class"] == 100
    assert result["CI_low"] == result["CI_high"] == 1
    assert result["status"] == "unstable_resampling"


def test_paired_identical_predictions_have_zero_difference_and_shared_missing_mask():
    y = [0.0, 1.0, 0.0, 1.0, np.nan]
    p = [0.1, 0.9, 0.2, 0.8, 0.6]
    result = grouped_bootstrap(y, p, ["A", "A", "B", "B", "C"], paired_predictions=p, repeats=40)
    assert result["n"] == 4 and result["group_count"] == 2
    assert result["point"] == result["CI_low"] == result["CI_high"] == 0
    assert result["valid_repeats"] == 40


def test_paired_effect_direction_and_reproducibility():
    y = [0, 1, 0, 1]
    a, b, groups = [0.1, 0.9, 0.2, 0.8], [0.9, 0.1, 0.8, 0.2], ["A", "A", "B", "B"]
    result = grouped_bootstrap(y, a, groups, paired_predictions=b, repeats=20, seed=5)
    assert result["point"] == result["CI_low"] == result["CI_high"] == 1
    assert result == grouped_bootstrap(y, a, groups, paired_predictions=b, repeats=20, seed=5)


def test_invalid_classes_groups_and_predictions_are_explicit():
    single = grouped_bootstrap([1, 1], [0.8, 0.9], ["A", "B"], repeats=10)
    assert single["status"] == "insufficient_classes" and single["point"] is None
    one_group = grouped_bootstrap([0, 1], [0.1, 0.9], ["A", "A"], repeats=10)
    assert one_group["status"] == "insufficient_groups" and one_group["CI_low"] is None
    with pytest.raises(ValueError, match="Grup"):
        grouped_bootstrap([0, 1], [0.1, 0.9], ["A", None], repeats=10)
    with pytest.raises(ValueError, match="olasılık"):
        grouped_bootstrap([0, 1], [0.1, 1.2], ["A", "B"], repeats=10)


def test_macro_bootstrap_fixed_task_set_and_paired_draws():
    y = np.array([[0, 0], [1, 1], [0, 0], [1, 1]])
    p = np.array([[0.1, 0.2], [0.9, 0.8], [0.2, 0.3], [0.8, 0.7]])
    result = macro_grouped_bootstrap(y, p, ["A", "A", "B", "B"], repeats=20, paired_predictions=p)
    assert result["evaluable_tasks"] == 2
    assert result["point"] == result["CI_low"] == result["CI_high"] == 0
    y[:, 1] = 1
    result = macro_grouped_bootstrap(y, p, ["A", "A", "B", "B"], repeats=20)
    assert result["evaluable_tasks"] == 1 and result["status"] == "incomplete_task_set"


def test_recall_ties_do_not_depend_on_drug_identifier():
    _, _, summary = drug_query_metrics(
        lambda d, c: [0, 0, 0],
        ["drug:a", "drug:b", "drug:c"],
        [("drug:c", "disease:x")],
        [("drug:c", "disease:x")],
    )
    assert summary["unfiltered_target_Recall@1"] == pytest.approx(1 / 3)
