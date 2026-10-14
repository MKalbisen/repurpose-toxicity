"""Record only measured academic-v2 stages after historical foundation validation."""

import json

import torch  # noqa: F401 - initialize Windows CUDA DLLs before RDKit/scikit-learn

from repurpose.academic.prepare import verify_bundle
from repurpose.academic.protocol import read_config, verify_lock
from repurpose.academic.state import AcademicState
from repurpose.academic.toxicity import verify_complete
from repurpose.utils import ROOT


def main():
    state, config = AcademicState(), read_config()
    run = ROOT / "reports/academic/runs/academic-v2"
    with state.lock():
        record = state.read()
        record["preferences"]["hardware"] = (
            "RTX 3050 Ti Laptop GPU 4 GB; uzun eğitim başka makinede yapılabilir"
        )
        state._persist(record)
    data_evidence = []
    for domain in ["toxicity", "kg"]:
        directory = ROOT / "data/processed/academic-v2" / domain
        manifest = verify_bundle(directory)
        data_evidence += [directory / "manifest.json", *[directory / p for p in manifest["files"]]]
    state.start("A03", "v2 kimyasal bileşenler ve tam uygun KG split kanıtları doğrulanıyor.")
    state.finish(
        "A03",
        data_evidence + ["src/repurpose/academic/prepare.py", "tests/test_academic_prepare.py"],
        "7729 molekül, altı kimlikte overlap=0; 47130 entity/3757160 triple; coverage tam. Ayna sürümü/lisans sınırlılığı açık.",
    )
    pilot = run / "pilot/hardware_pilot.json"
    evidence = json.loads(pilot.read_text(encoding="utf-8"))
    assert evidence["test_accessed"] is False and len(evidence["trials"]) == 4
    state.start("A04", "Gerçek CUDA pilotu ve eğitim/test ayrımı testleri kaydediliyor.")
    state.finish(
        "A04",
        [
            pilot,
            "configs/academic_run.yaml",
            "src/repurpose/academic/experiment.py",
            "tests/test_academic_experiment.py",
            "reports/academic/windows_tests.xml",
        ],
        "3050 Ti CUDA pilotu başarılı; KGE üst bütçe yaklaşık 464 saat. MLP/KGE checkpoint ve test kapısı test edildi.",
    )
    verify_lock(config, run)
    state.start("A05", "Kod/veri/ayar/bilimsel paket kilidi doğrulanıyor.")
    state.finish(
        "A05",
        [
            run / "protocol_lock.json",
            "src/repurpose/academic/protocol.py",
            "configs/academic_run.yaml",
            "requirements-academic-lock.txt",
        ],
        "v2 protokol kilitli. Ayrı final komutu ve seçilmiş ağırlık hash kaydı etkin; akademik test henüz açılmadı.",
    )
    evidence = []
    for seed in config["seeds"]:
        directory = run / "toxicity" / str(seed)
        manifest = verify_complete(directory)
        evidence += [directory / "complete.json", *[directory / p for p in manifest["files"]]]
    state.start("A06", "Beş seed toksisite model ve kalibrasyon kayıtları doğrulanıyor.")
    state.finish(
        "A06",
        evidence + ["src/repurpose/academic/toxicity.py"],
        "17/29/43/71/101 seed; prevalence, Logistic Regression, XGBoost, MLP ve ayrı grup kalibrasyonu tamam. Final test açılmadı.",
    )
    state.note(
        "11 Ekim 2026: Kullanıcı 3050 Ti sağladı; uzun KGE eğitimi için başka makine kullanabilecek. Tam arama yerelde başlatılmadı. Paket/komutlar docs/academic/REMOTE_TRAINING.md içinde. A07–A10 ve tam kimyasal eşleme/entegrasyon henüz tamamlanmadı."
    )


if __name__ == "__main__":
    main()
