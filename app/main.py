import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from stmol import showmol

from app.components.graph_view import show_graph
from app.services.model_service import load_service
from repurpose.utils import TASKS
from repurpose.visualization.graph_builder import local_paths
from repurpose.visualization.molecule_view import molecule_3d, molecule_svg
from repurpose.visualization.xai import bit_contributions

st.set_page_config(page_title="İlaç Araştırma Kokpiti", page_icon="🧬", layout="wide")
st.markdown(
    """<style>.block-container{padding-top:2rem;max-width:1450px}
[data-testid="stMetric"]{background:#13243a;border-radius:12px;padding:18px}
</style>""",
    unsafe_allow_html=True,
)
st.caption("PRIMEKG · TOX21 · ARAŞTIRMA PLATFORMU")
st.title("İlaç yeniden konumlandırma & toksisite")
st.write(
    "Hastalık bağlantılarını araştırın, ilaç adaylarını karşılaştırın ve tahmini assay risklerini inceleyin."
)
with st.sidebar:
    st.header("Analiz ayarları")
    profile = st.selectbox("Deney profili", ["smoke", "full"])
    model = st.selectbox("Toksisite modeli", ["XGBoost", "MLP"])
    strategy_label = st.selectbox("Sıralama", ["Ağırlıklı skor", "Sert risk filtresi", "Pareto sınırı"])
    strategy = {"Ağırlıklı skor": "weighted", "Sert risk filtresi": "hard_filter", "Pareto sınırı": "pareto"}[
        strategy_label
    ]
    top_n = st.slider("Aday sayısı", 5, 100, 30)
    threshold = st.slider("Assay risk eşiği", 0.05, 0.95, 0.5, 0.05)
    weight = st.slider("KG skor ağırlığı", 0.0, 1.0, 0.7, 0.05)
    st.caption("Eşikler ve ağırlık araştırma seçimleridir. Bilinmeyen riskler düşük risk kabul edilmez.")
artifact = ROOT / "models" / profile / "kg/best_kge_config.json"
if not artifact.exists():
    st.info(
        "Bu profil için modeller henüz hazırlanmadı. README içindeki çevrimdışı eğitim komutunu çalıştırın."
    )
    st.code(f"repurpose --config configs/{profile}.yaml run")
    st.stop()
try:
    stamps = [p.stat().st_mtime_ns for p in (ROOT / "models" / profile).rglob("*") if p.is_file()]
    mapping_stamp = (ROOT / "data/processed" / profile / "drug_chemical_mapping.csv").stat().st_mtime_ns
    service = load_service(profile, (max(stamps), mapping_stamp))
except (FileNotFoundError, ValueError, RuntimeError) as e:
    st.error(f"Model veya veri dosyaları hazır değil: {e}")
    st.stop()
if profile == "smoke":
    st.info(
        "Kısa gerçek-veri doğrulaması: sınırlı çizge ve kısa eğitim. Sonuçlar akademik performans ölçümü değildir."
    )
st.caption(
    "KG skoru aday önceliğini, toksisite çıktıları predicted assay risk değerini gösterir. "
    "Bu arayüz tedavi önerisi veya klinik güvenlilik değerlendirmesi üretmez."
)
diseases = service.entities[service.entities.type == "disease"].sort_values("name")
disease = st.selectbox(
    "Hedef hastalık",
    diseases.entity_id.tolist(),
    format_func=lambda ident: diseases.set_index("entity_id").loc[ident, "name"] + f" ({ident})",
)
if st.button("Analiz Et", type="primary"):
    with st.spinner("Adaylar skorlanıyor ve assay riskleri hesaplanıyor…"):
        try:
            result, seconds = service.analyze(disease, top_n, threshold, weight, strategy, model)
            st.session_state["analysis"] = (result, seconds, disease, model, threshold, weight, strategy)
        except (ValueError, RuntimeError) as e:
            st.error(str(e))
if "analysis" not in st.session_state:
    st.write("Bir hastalık seçip **Analiz Et** düğmesine basın.")
    st.stop()
result, seconds, analyzed_disease, analyzed_model, analyzed_threshold, analyzed_weight, analyzed_strategy = (
    st.session_state.analysis
)
if (disease, model, threshold, weight, strategy) != (
    analyzed_disease,
    analyzed_model,
    analyzed_threshold,
    analyzed_weight,
    analyzed_strategy,
):
    st.warning(
        "Ayarlar değişti. Aşağıdaki sonuçlar önceki analize ait; güncellemek için Analiz Et düğmesine basın."
    )
a, b, c, d = st.columns(4)
a.metric("İlaç adayı", len(result))
b.metric("12 assay tamamlanan", int(result.complete_risk.sum()))
c.metric("Eşik altında", int(result.eligible.sum()))
d.metric("Çıkarım süresi", f"{seconds:.2f} sn")
table, graph, experiment = st.tabs(["Aday analizi", "Biyomedikal çizge", "Deney sonuçları"])
with table:
    shown = result if analyzed_strategy != "hard_filter" else result[result.eligible]
    if shown.empty:
        st.info("Bu eşikte ve tam assay bilgisiyle kalan aday yok. Tüm adaylar indirme dosyasında yer alır.")
    st.dataframe(
        shown[
            [
                "final_rank",
                "drug_name",
                "known_link",
                "kge_score",
                "max_risk",
                "mean_risk",
                "high_risk_count",
                "assay_count",
                "combined_score",
                "pareto",
                "exclusion_reason",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Tüm sonuçları CSV indir",
        result.to_csv(index=False).encode("utf-8-sig"),
        "ilac_adaylari.csv",
        "text/csv",
    )
    if not result.empty:
        chosen = st.selectbox(
            "Aday ayrıntısı",
            result.drug_id.tolist(),
            format_func=lambda x: result.set_index("drug_id").loc[x, "drug_name"],
        )
        row = result[result.drug_id == chosen].iloc[0]
        left, right = st.columns(2)
        with left:
            st.subheader(row.drug_name)
            if isinstance(row.canonical_smiles, str):
                st.code(row.canonical_smiles)
                components.html(molecule_svg(row.canonical_smiles), height=330)
                with st.expander("Etkileşimli 3D yapı"):
                    block, error = molecule_3d(row.canonical_smiles)
                    if error:
                        st.info(error)
                    else:
                        view = py3Dmol.view(width=550, height=350)
                        view.addModel(block, "mol")
                        view.setStyle({"stick": {}})
                        view.zoomTo()
                        showmol(view, height=350, width=550)
            else:
                st.info("Doğrulanmış kimyasal eşleme yok; assay riski bilinmiyor.")
        with right:
            st.subheader("Tahmini assay riskleri")
            risks = pd.DataFrame({"assay": TASKS, "risk": [row[f"risk_{t}"] for t in TASKS]}).set_index(
                "assay"
            )
            st.bar_chart(risks, horizontal=True)
            st.caption("Olasılıklar kalibre edilmemiştir; boş değerler bilinmeyen assay riskidir.")
            st.write(f"Normalize KG: {row.normalized_kge:.3f} · Ağırlık: {analyzed_weight:.2f}")
            if pd.notna(row.combined_score):
                st.write(
                    f"Birleşik skor: {analyzed_weight:.2f} × {row.normalized_kge:.3f} "
                    f"− {1 - analyzed_weight:.2f} × {row.max_risk:.3f} = {row.combined_score:.3f}"
                )
            else:
                st.write("Birleşik skor: assay bilgisi eksik olduğu için bilinmiyor.")
        with st.expander("Model açıklamaları ve yerel çizge patikaları"):
            evidence = local_paths(service.triples, analyzed_disease, chosen)
            for path in evidence:
                st.text(path)
            if not evidence:
                st.write("Sınırlı alt çizgede 1–2 hop patika bulunamadı.")
            if analyzed_model == "XGBoost" and isinstance(row.canonical_smiles, str):
                assay = st.selectbox("Açıklanacak assay", list(service.xgb))
                bits, meta = bit_contributions(service.xgb[assay], row.canonical_smiles)
                st.dataframe(bits, hide_index=True)
                st.caption(
                    f"TreeSHAP log-odds katkıları. Bias: {meta['bias']:.3f}. "
                    "Bit çakışmaları olabilir; bu açıklamalar biyolojik nedensellik kanıtı değildir."
                )
            elif analyzed_model == "MLP":
                st.caption(
                    "MLP için kimyasal katkı açıklaması uygulanmadı; çizge patikaları ve skor bileşenleri gösterilir."
                )
with graph:
    st.caption("Hastalık: sarı · İlaç: yeşil · Gen/protein: mor · Pathway: pembe. En fazla 100 düğüm.")
    show_graph(service, analyzed_disease)
with experiment:
    for name in ["kge_comparison.csv", "model_comparison.csv", "toxicity_final_metrics.csv"]:
        path = ROOT / "reports" / profile / name
        if path.exists():
            st.caption(name)
            st.dataframe(pd.read_csv(path), hide_index=True)
