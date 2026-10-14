"""Generate an honest report from existing experiment artifacts, including PDF."""

import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from repurpose.utils import ROOT, paths


def architecture(dest):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis("off")
    labels = [
        "PrimeKG\nClean + coverage split",
        "PyKEEN\nTransE / RotatE",
        "Drug identity\nVerified SMILES",
        "RDKit\n1024-bit ECFP4",
        "Tox21 models\nXGBoost / MLP",
        "Rank + explain\nStreamlit",
    ]
    for i, label in enumerate(labels):
        ax.text(
            i * 0.17 + 0.07,
            0.65,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=.7", "fc": "#dcf4ef", "ec": "#309987"},
        )
        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=(i * 0.17 + 0.18, 0.65),
                xytext=(i * 0.17 + 0.14, 0.65),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#1c4263"},
            )
    ax.text(
        0.5,
        0.23,
        "Offline training: validation selection / held-out test / scaffold groups / fixed seed",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
    )
    fig.savefig(dest, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_report(config):
    _, _, reports = paths(config)
    lines = [
        "# İlaç Yeniden Konumlandırma ve Toksisite — Teknik Rapor",
        "",
        f"Profil: {config['profile']}; seed: {config['seed']}.",
        "Amaç: bilgi çizgesi bağlantı skorunu moleküler assay riskiyle birlikte incelemek.",
        "Kısa profil sonuçları çalışma doğrulamasıdır; tam deney sonucu olarak yorumlanamaz.",
        "",
        "## Veri ve lisans",
        "PrimeKG DOI: 10.7910/DVN/IXA7BM. Kaynak manifesti reports/data_manifest.json içindedir.",
        "PrimeKG yazılımının MIT lisansı, DrugBank dahil kaynak verilerin lisansını kapsamaz.",
        "Tox21: DeepChem MoleculeNet dağıtımı; 12 in vitro assay. Kaynak URL ve hash manifesttedir.",
        "",
        "## Yöntem",
        "KG: type:source_id kimlikleri; aynı çift arasındaki tüm yön/ilişkiler tek splitte.",
        "TransE/RotatE validation MRR ile seçilir; filtered test tüm bilinen pozitifleri filtreler.",
        "Tox21: canonical isomeric SMILES, duplicate çelişkiler NaN; eksikler negatif sayılmaz.",
        "Morgan radius=2, 1024 bit. Fragment/tautomer standardizasyonu scaffold grup kimliğinde kullanılır.",
        "Scaffold grupları bölünmez; acyclic moleküller tek grup. Oranlar yaklaşık %80/10/10.",
        "XGBoost: görev başına class weight, validation AP seçimi. MLP: 1024/512/256/12, maskeli BCE.",
        "MLP validation macro ROC-AUC ile erken durdurulur. Test model seçiminde kullanılmaz.",
        "PR-AUC average precision; %95 bootstrap CI yalnızca iki sınıflı çekimlerde hesaplanır.",
        "",
        "## Gerçekleşen deneyler",
    ]
    manifest = ROOT / "reports/data_manifest.json"
    if manifest.exists():
        lines += ["", "### Kaynak manifesti", manifest.read_text(encoding="utf-8")]
    lines += [
        "",
        "Resmî PrimeKG deposu OptimusKG güncellemesini bildirir; bu deney sabit PrimeKG MVP baseline kullanır.",
        "Kaynak: https://github.com/mims-harvard/PrimeKG ; OptimusKG şema desteği uygulanmadı.",
    ]
    tests = reports / "test_results.xml"
    if tests.exists():
        import xml.etree.ElementTree as ET

        suite = ET.parse(tests).getroot().find("testsuite")
        lines += ["", "### Otomatik testler", json.dumps(suite.attrib, ensure_ascii=False)]
    for name in [
        "kg_data_report.json",
        "split_report.json",
        "fingerprint_metadata.json",
        "scaffold_split_report.json",
        "mapping_report.json",
        "inference_timing.json",
    ]:
        path = reports / name
        lines += ["", f"### {name}", path.read_text() if path.exists() else "Bu adım henüz çalıştırılmadı."]
    for name in [
        "kge_comparison.csv",
        "model_comparison.csv",
        "toxicity_final_metrics.csv",
        "experiment_log.csv",
    ]:
        path = reports / name
        lines += [
            "",
            f"### {name}",
            pd.read_csv(path).to_string(index=False) if path.exists() else "Bu deney henüz çalıştırılmadı.",
        ]
    lines += [
        "",
        "## Sıralama ve açıklanabilirlik",
        "Hard filter: 12 assay tamamlanmış ve maksimum risk eşikten düşük. Bilinmeyenler elenir.",
        "Ağırlıklı skor: w × min-max KG − (1−w) × maksimum assay riski; eksikler skorlanmaz.",
        "Pareto: KG yüksek ve risk düşük hedefleri. Eşit noktalar birlikte korunur.",
        "sensitivity.csv ağırlık 0/.25/.5/.7/.75/1 sonuçlarını kaydeder; varsayılan .7 klinik optimum değildir.",
        "XGBoost native TreeSHAP bit katkıları log-odds birimindedir; hash çakışmaları açıklamayı sınırlar.",
        "Yerel 1–2 hop çizge patikaları gözlenen ilişkileri gösterir; nedensellik kanıtı değildir.",
        "",
        "## Sınırlılıklar ve tamamlanmamış deneyler",
        "Tam PrimeKG ve uzun full profil bu kısa doğrulamanın yerine geçmez; ayrı yürütülmelidir.",
        "Assay riskleri kalibre edilmemiştir. Tox21 assay aktivitesi klinik toksisite ile eşdeğer değildir.",
        "KG hedeflerinin test metrikleri tüm ilişkilere aittir; drug-disease özel prospektif doğrulama yapılmadı.",
        "RDKit tautomer enumeration sınırları geçerlidir; karmaşık moleküllerde tüm tautomerler tüketilmemiş olabilir.",
        "Kimyasal eşleme DrugBank kimliği doğrulanamayan adaylarda bilinmeyendir; kapsam mapping_report içindedir.",
        "Salt ve stereokimya retained; fingerprint chirality kapalı. Scaffold kimliği achiral tautomer parenttır.",
        "Ablasyon/çoklu seed/prospektif klinik doğrulama yapılmadı. Hedef metrikler başarı garantisi değildir.",
        "UI 3–5 sn hedefi gözlenen çıkarım süresiyle karşılaştırılır; cold load ve 3D süreleri ayrıca değişebilir.",
        "",
        "## Tekrar üretme",
        "README komutları, configs/*.yaml, requirements.txt ve manifest SHA256 kayıtlarını kullanın.",
    ]
    text = "\n".join(lines)
    (reports / "final_report.md").write_text(text, encoding="utf-8")
    architecture(reports / "architecture.png")
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", font))
    pdf = canvas.Canvas(str(reports / "final_report.pdf"))
    pdf.setTitle("İlaç Yeniden Konumlandırma ve Toksisite — Teknik Rapor")
    pdf.setFont("DejaVu", 8)
    y = 800
    for line in text.splitlines():
        for wrapped in textwrap.wrap(line, width=108, replace_whitespace=False) or [""]:
            if y < 45:
                pdf.showPage()
                pdf.setFont("DejaVu", 8)
                y = 800
            pdf.drawString(35, y, wrapped)
            y -= 12
    pdf.save()
    (reports / "ranking_strategy.md").write_text(
        "\n".join(lines[lines.index("## Sıralama ve açıklanabilirlik") :]), encoding="utf-8"
    )
