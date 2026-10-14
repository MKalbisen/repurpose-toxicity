# Akademik çalışma ilerlemesi

Son güncelleme (UTC): 2026-10-11T10:11:02.889726+00:00
Ana plan: ACADEMIC_PLAN.md. Bu dosya reports/academic/state.json içinden üretilir.

## Kullanıcı tercihleri

- Donanım: RTX 3050 Ti Laptop GPU 4 GB; uzun eğitim başka makinede yapılabilir
- Akademik hedef türü henüz belirtilmedi; geçici kapsam retrospektif tez/makale çalışması.

## Aşama durumu

| ID | İş | Durum | Kanıt kontrolü | Sonuç |
|---|---|---|---|---|
| A00 | Deney ve devam kayıtları | validated | geçerli | Plan/protokol taslağı, atomik JSON, olay günlüğü, artefakt hash ve kesinti testleri geçti. |
| A01 | Bilimsel veri/kod denetimi | validated | geçerli | Read-only bilimsel denetim tamamlandı; bulgular audit.json/md içinde. Kritik açıklar hâlâ A03–A09 iş listesinde. |
| A02 | Hedef değerlendirme altyapısı | validated | geçerli | Drug-only filtered MRR, canonical hedef tekilleştirme, analytic random baseline, scaffold/paired bootstrap testleri geçti; legacy validation adapter çalıştı. |
| A03 | Akademik veri ve split üretimi | validated | geçerli | 7729 molekül, altı kimlikte overlap=0; 47130 entity/3757160 triple; coverage tam. Ayna sürümü/lisans sınırlılığı açık. |
| A04 | CPU/GPU pilot ve development protokolü | validated | geçerli | 3050 Ti CUDA pilotu başarılı; KGE üst bütçe yaklaşık 464 saat. MLP/KGE checkpoint ve test kapısı test edildi. |
| A05 | Nihai protokol/test kilidi | validated | geçerli | v2 protokol kilitli. Ayrı final komutu ve seçilmiş ağırlık hash kaydı etkin; akademik test henüz açılmadı. |
| A06 | Toksisite eğitimleri ve baselines | validated | geçerli | 17/29/43/71/101 seed; prevalence, Logistic Regression, XGBoost, MLP ve ayrı grup kalibrasyonu tamam. Final test açılmadı. |
| A07 | KGE eğitimleri ve baselines | planned | bekliyor/değişmiş |  |
| A08 | Final test ve istatistik | planned | bekliyor/değişmiş |  |
| A09 | Entegrasyon/ablation/dış doğrulama | planned | bekliyor/değişmiş |  |
| A10 | Akademik rapor ve yeniden üretme | planned | bekliyor/değişmiş |  |

## Kaldığımız nokta

İlk tamamlanmamış veya yeniden doğrulanması gereken aşama: **A07**.
`source .venv/bin/activate`
`python -m repurpose.academic.cli status`
`python -m repurpose.academic.cli resume`

Resume yalnız uygulanmış hazırlık/denetim adımlarını yürütür; henüz kodlanmamış uzun eğitimleri başlatmaz.
Güncel çalışma ve komutlar: reports/academic/REVIEW.md ve docs/academic/REMOTE_TRAINING.md.
Akademik veri/pilot/eğitim: python -m repurpose.academic.experiment --help. Smoke sonuçları final sonuç değildir.

## Değişiklik ve kesinti kaydı

Her başlangıç/başarısızlık/doğrulama reports/academic/events.jsonl içine UTC zamanı ile eklenir.
State evidence listesinde dosya yolları, byte sayıları ve SHA256 vardır; dosya değişirse eski doğrulama geçersiz olur.
Yeni MLP/KGE eğitimlerinde epoch sonunda optimizer/RNG checkpoint vardır. Klasik model aileleri tamamlanınca saklanır.

## Kayıtlı dosyalar

- `ACADEMIC_PLAN.md`
- `configs/academic_protocol.yaml`
- `configs/academic_run.yaml`
- `data/processed/academic-v2/kg/entities.csv`
- `data/processed/academic-v2/kg/manifest.json`
- `data/processed/academic-v2/kg/test.tsv`
- `data/processed/academic-v2/kg/train.tsv`
- `data/processed/academic-v2/kg/validation.tsv`
- `data/processed/academic-v2/toxicity/calibration.npz`
- `data/processed/academic-v2/toxicity/chemical_identity.csv`
- `data/processed/academic-v2/toxicity/development_labels.csv`
- `data/processed/academic-v2/toxicity/excluded_molecules.csv`
- `data/processed/academic-v2/toxicity/label_conflicts.csv`
- `data/processed/academic-v2/toxicity/manifest.json`
- `data/processed/academic-v2/toxicity/molecules.csv`
- `data/processed/academic-v2/toxicity/nearest_train_similarity.csv`
- `data/processed/academic-v2/toxicity/test.npz`
- `data/processed/academic-v2/toxicity/train.npz`
- `data/processed/academic-v2/toxicity/validation.npz`
- `data/processed/smoke/drug_chemical_mapping.csv`
- `data/processed/smoke/entities.csv`
- `data/processed/smoke/test.tsv`
- `data/processed/smoke/tox21_clean.csv`
- `data/processed/smoke/tox21_features.npz`
- `data/processed/smoke/train.tsv`
- `data/processed/smoke/triples.tsv`
- `data/processed/smoke/validation.tsv`
- `reports/academic/assay_label_audit.csv`
- `reports/academic/audit.json`
- `reports/academic/audit.md`
- `reports/academic/foundation_tests.xml`
- `reports/academic/foundation_validation.json`
- `reports/academic/metrics_tests.xml`
- `reports/academic/metrics_validation.json`
- `reports/academic/runs/academic-v2/pilot/hardware_pilot.json`
- `reports/academic/runs/academic-v2/protocol_lock.json`
- `reports/academic/runs/academic-v2/toxicity/101/complete.json`
- `reports/academic/runs/academic-v2/toxicity/101/logistic.joblib`
- `reports/academic/runs/academic-v2/toxicity/101/mlp.joblib`
- `reports/academic/runs/academic-v2/toxicity/101/mlp.pt`
- `reports/academic/runs/academic-v2/toxicity/101/prevalence.joblib`
- `reports/academic/runs/academic-v2/toxicity/101/xgboost.joblib`
- `reports/academic/runs/academic-v2/toxicity/17/complete.json`
- `reports/academic/runs/academic-v2/toxicity/17/logistic.joblib`
- `reports/academic/runs/academic-v2/toxicity/17/mlp.joblib`
- `reports/academic/runs/academic-v2/toxicity/17/mlp.pt`
- `reports/academic/runs/academic-v2/toxicity/17/prevalence.joblib`
- `reports/academic/runs/academic-v2/toxicity/17/xgboost.joblib`
- `reports/academic/runs/academic-v2/toxicity/29/complete.json`
- `reports/academic/runs/academic-v2/toxicity/29/logistic.joblib`
- `reports/academic/runs/academic-v2/toxicity/29/mlp.joblib`
- `reports/academic/runs/academic-v2/toxicity/29/mlp.pt`
- `reports/academic/runs/academic-v2/toxicity/29/prevalence.joblib`
- `reports/academic/runs/academic-v2/toxicity/29/xgboost.joblib`
- `reports/academic/runs/academic-v2/toxicity/43/complete.json`
- `reports/academic/runs/academic-v2/toxicity/43/logistic.joblib`
- `reports/academic/runs/academic-v2/toxicity/43/mlp.joblib`
- `reports/academic/runs/academic-v2/toxicity/43/mlp.pt`
- `reports/academic/runs/academic-v2/toxicity/43/prevalence.joblib`
- `reports/academic/runs/academic-v2/toxicity/43/xgboost.joblib`
- `reports/academic/runs/academic-v2/toxicity/71/complete.json`
- `reports/academic/runs/academic-v2/toxicity/71/logistic.joblib`
- `reports/academic/runs/academic-v2/toxicity/71/mlp.joblib`
- `reports/academic/runs/academic-v2/toxicity/71/mlp.pt`
- `reports/academic/runs/academic-v2/toxicity/71/prevalence.joblib`
- `reports/academic/runs/academic-v2/toxicity/71/xgboost.joblib`
- `reports/academic/validation_diagnostic.json`
- `reports/academic/validation_drug_ranks.csv`
- `reports/academic/windows_tests.xml`
- `reports/data_manifest.json`
- `requirements-academic-lock.txt`
- `scripts/resume_academic.sh`
- `src/repurpose/academic/audit.py`
- `src/repurpose/academic/cli.py`
- `src/repurpose/academic/diagnostic.py`
- `src/repurpose/academic/experiment.py`
- `src/repurpose/academic/metrics.py`
- `src/repurpose/academic/prepare.py`
- `src/repurpose/academic/protocol.py`
- `src/repurpose/academic/state.py`
- `src/repurpose/academic/toxicity.py`
- `src/repurpose/data/chemistry.py`
- `src/repurpose/data/kg_data.py`
- `src/repurpose/data/tox21.py`
- `src/repurpose/kg/train.py`
- `src/repurpose/toxicity/evaluate.py`
- `src/repurpose/toxicity/models.py`
- `src/repurpose/toxicity/train.py`
- `tests/test_academic_experiment.py`
- `tests/test_academic_metrics.py`
- `tests/test_academic_prepare.py`
- `tests/test_academic_state.py`

## Son çalışma notları

- 2026-10-08T12:24:57.388890+00:00: Kullanıcı: şu an GPU yok; gerektiğinde T4/L4 sağlanabilir. Akademik hedef türü belirtilmedi. Ücretli compute başlatılmadı.
- 2026-10-08T12:24:58.110924+00:00: A01 gerçek bulgu: scaffold/canonical overlap=0; 11 fingerprint eşdeğerlik grubu split sınırlarını aşıyor. A03 bileşen gruplaması gerektirir; bu tek başına aynı molekül/etiket sızıntısı kanıtı değildir.
- 2026-10-08T12:24:58.845407+00:00: A02 validation-only karşılaştırma: RotatE drug-only MRR=0.173939; train-popularity MRR=0.221466; random expected=0.101415. Nihai test/akademik başarı sonucu değildir.
- 2026-10-08T12:24:59.824092+00:00: Bu oturumda eklenenler: ACADEMIC_PLAN.md, ACADEMIC_PROGRESS.md, configs/academic_protocol.yaml, src/repurpose/academic/{state,audit,metrics,diagnostic,cli}.py, tests/test_academic_state.py, tests/test_academic_metrics.py, scripts/resume_academic.sh; README ve PLAN yönlendirmeleri güncellendi.
- 2026-10-08T12:25:00.938907+00:00: Sonraki somut iş emri docs/academic/A03_WORK_ORDER.md içine kaydedildi: A03.1 versioned chemical identity/cache ve tautomer completion; A03.2 transitive scaffold/parent/fingerprint split; A03.3 KG ana deney verisi. A03 henüz uygulanmadı.
- 2026-10-08T12:25:49.738060+00:00: Son bütünlük kontrolü: 26 test geçti; ruff temiz. reports/academic/all_tests.xml ve all_tests.log kayıtlı. Resume A00/A01/A02 için geçerli kanıtı tanıyıp adımları atladı; sıradaki iş A03.
- 2026-10-11T10:09:18.824722+00:00: 11 Ekim 2026: Kullanıcı 3050 Ti sağladı; uzun KGE eğitimi için başka makine kullanabilecek. Tam arama yerelde başlatılmadı. Paket/komutlar docs/academic/REMOTE_TRAINING.md içinde. A07–A10 ve tam kimyasal eşleme/entegrasyon henüz tamamlanmadı.
