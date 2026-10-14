import numpy as np
import pytest

from repurpose.utils import ROOT, load_config

ARTIFACT = ROOT / "models/smoke/kg/best_kge_config.json"


@pytest.mark.skipif(not ARTIFACT.exists(), reason="Run real-data smoke pipeline first")
def test_disease_candidates_toxicity_ranking_and_reload():
    from repurpose.integration.inference_pipeline import InferenceService

    service = InferenceService(load_config(ROOT / "configs/smoke.yaml"))
    query = service.triples[service.triples.relation == "indication"]["tail"]
    disease = next(x for x in query if x.startswith("disease:"))
    result, seconds = service.analyze(disease, top_n=20)
    repeated, _ = service.analyze(disease, top_n=20)
    assert result.drug_id.tolist() == repeated.drug_id.tolist()
    assert np.allclose(result.kge_score, repeated.kge_score)
    assert len(result) == 20 and seconds >= 0
    assert result.assay_count.max() == 12
    valid = result.filter(regex="^risk_").to_numpy()
    finite = valid[np.isfinite(valid)]
    assert ((finite >= 0) & (finite <= 1)).all()
    for name in ["XGBoost", "MLP"]:
        ranked, _ = service.analyze(disease, model=name)
        assert ranked.complete_risk.any()


def test_streamlit_initial_and_analysis():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=60).run()
    assert not at.exception
    if ARTIFACT.exists():
        at.button[0].click().run(timeout=60)
        assert not at.exception
        assert len(at.metric) == 4


@pytest.mark.skipif(
    not (ROOT / "data/processed/smoke/tox21_features.npz").exists(), reason="Prepare Tox21 first"
)
def test_real_data_scaffold_and_missing_label_artifacts():
    import pandas as pd

    base = ROOT / "data/processed/smoke"
    data = np.load(base / "tox21_features.npz")
    clean = pd.read_csv(base / "tox21_clean.csv")
    assert data["X"].shape == (len(clean), 1024)
    assert data["Y"].shape == (len(clean), 12)
    assert np.isnan(data["Y"]).any()
    sets = [set(clean.iloc[data[name]].scaffold) for name in ["train", "val", "test"]]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert sum(len(data[name]) for name in ["train", "val", "test"]) == len(clean)
