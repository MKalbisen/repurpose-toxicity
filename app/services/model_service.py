import streamlit as st

from repurpose.integration.inference_pipeline import InferenceService
from repurpose.utils import ROOT, load_config, seed_all


@st.cache_resource
def load_service(profile, artifact_version):
    seed_all(42)
    return InferenceService(load_config(ROOT / "configs" / f"{profile}.yaml"))
