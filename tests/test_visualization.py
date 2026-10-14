import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from repurpose.data.chemistry import fingerprint
from repurpose.visualization.graph_builder import neighborhood
from repurpose.visualization.molecule_view import molecule_3d, molecule_svg
from repurpose.visualization.xai import bit_contributions


def test_bounded_graph_and_3d_fallback():
    triples = pd.DataFrame(
        [("disease:1", "r", f"drug:{i}") for i in range(200)], columns=["head", "relation", "tail"]
    )
    nodes, edges = neighborhood(triples, "disease:1", max_nodes=12)
    assert len(nodes) == 12 and len(edges) == 11
    assert "<svg" in molecule_svg("CCO")
    assert molecule_3d("invalid")[0] is None
    block, error = molecule_3d("CCO")
    assert "V2000" in block and error is None


def test_treeshap_contributions_reconstruct_logit():
    x = np.stack([fingerprint(s) for s in ["CCO", "CCCC", "c1ccccc1", "c1ccncc1"]])
    model = XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1, min_child_weight=0)
    model.fit(x, [0, 0, 1, 1])
    rows, meta = bit_contributions(model, "c1ccccc1")
    assert len(rows) == 12
    assert abs(meta["sum_log_odds"] - meta["raw_margin"]) < 1e-5
