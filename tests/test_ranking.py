import numpy as np
import pandas as pd

from repurpose.integration.ranking import pareto_mask, rank_candidates


def frame():
    return pd.DataFrame(
        {
            "drug_id": ["a", "b", "c", "d"],
            "kge_score": [10.0, 8.0, 7.0, 20.0],
            "max_risk": [0.3, 0.1, 0.9, np.nan],
            "mean_risk": [0.1, 0.08, 0.5, np.nan],
            "assay_count": [12, 12, 12, 0],
            "risk_test": [0.3, 0.1, 0.9, np.nan],
        }
    )


def test_unknown_risk_is_never_safe_or_pareto():
    result = rank_candidates(frame())
    unknown = result[result.drug_id == "d"].iloc[0]
    assert not unknown.eligible and not unknown.pareto
    assert np.isnan(unknown.flag_test)
    assert result[result.drug_id == "c"].iloc[0].flag_test == 1
    assert np.isnan(unknown.combined_score)
    assert unknown.exclusion_reason == "unknown_or_incomplete_assay_risk"
    assert result.iloc[-1].drug_id == "d"


def test_hard_filter_and_pareto_dominance():
    result = rank_candidates(frame(), strategy="hard_filter")
    assert set(result[result.eligible].drug_id) == {"a", "b"}
    assert pareto_mask([5, 4, 3, 5], [0.1, 0.3, 0.5, 0.1]).tolist() == [True, False, False, True]
    assert rank_candidates(frame()).equals(rank_candidates(frame()))
