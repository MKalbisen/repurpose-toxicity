# Uygulama ve deney durumu

PDF planındaki yazılım modülleri oluşturuldu ve gerçek veride kısa uçtan uca çalışma doğrulandı.
Tam veri üzerinde uzun akademik deneyler ayrıca yürütülmelidir; aşağıdaki sonuçlar smoke profilidir.

| PDF adımı | Bölüm | Gerçek doğrulama |
|---|---|---|
| 1 | Ortam ve proje iskeleti | Python 3.11.17; tüm paket importları geçti; CPU doğrulandı, CUDA yok. |
| 2 | PrimeKG indirme ve inceleme | 8.100.498 ham yönlü satır tarandı; 231 entity, 1.800 triple, 9 relation kısa alt çizge. |
| 3 | Kimyasal kimlik | 50 ilaçtan 36 doğrulanmış SMILES; 14 eşleşmemiş/belirsiz. |
| 4 | KG split | 1440/180/180; entity/relation coverage tam, unordered pair overlap 0. |
| 5–6 | TransE/RotatE | 5 epoch; 32 boyut; validation MRR ile RotatE seçildi. Uzun full arama yapılmadı. |
| 7 | Hastalıktan aday | disease:4638 sorgusunda deterministik 30 aday ve bilinen bağlantı bayrakları. |
| 8–10 | Tox21/ECFP4/scaffold | 7831 ham, 7822 temiz molekül; 8 geçersiz SMILES, 1 başarısız scaffold; 6258/782/782; overlap 0. |
| 11 | XGBoost | 12 assay eğitildi; 35/60 ağaç validation AP ile seçildi; görev bazlı test metrikleri ve CI var. |
| 12 | Çok görevli MLP | 5 epoch; validation ile 2. epoch checkpoint seçildi; 12 assay, maskeli loss testi geçti. |
| 13 | Değerlendirme | 12 görev ROC/PR-AUC/precision/recall/F1/balanced accuracy; 40 çekimli kısa bootstrap. |
| 14–15 | Birleşik çıkarım ve sıralama | 30 adayın 24 tanesinde 12 assay tamamlandı; 0.122 sn sıcak çıkarım; hard/weighted/Pareto ve duyarlılık tablosu. |
| 16 | Streamlit | Modeller cache edilir; çevrimdışı eğitim; AppTest hastalık seçimi/analiz akışını geçti. |
| 17 | PyVis | En fazla 100 düğümlü 2 hop graph; 12 düğüm sınır testi geçti. |
| 18 | 2D/3D | RDKit SVG, ETKDGv3 ve Stmol/Py3Dmol; geçersiz/başarısız 3D fallback testi geçti. |
| 19 | XAI | XGBoost TreeSHAP bit katkıları raw logit toplamını yeniden oluşturdu; local paths ve skor bileşenleri UI'da. MLP kimyasal XAI yok. |
| 20 | Test/performans | 12 test geçti; failure=0, error=0, skipped=0; ruff temiz. |
| 21 | Rapor | Gerçek CSV/JSON/test çıktılarından final_report.md/pdf ve architecture.png üretildi. |

## Ölçülen model sonuçları

 model  dimension    lr split      MRR   Hits@1   Hits@3  Hits@10  validation_MRR
TransE         32 0.005  test 0.019659 0.000000 0.005556 0.030556        0.020319
RotatE         32 0.005  test 0.084608 0.027778 0.058333 0.177778        0.081808

  model  ROC-AUC   PR-AUC       F1  Balanced Accuracy  evaluable_assays
    MLP 0.744105 0.346993 0.321205           0.683230                12
XGBoost 0.706299 0.301203 0.278990           0.639067                12

Kısa eğitim Hits@10≥.70 ve mean ROC-AUC>.80 hedeflerini sağlamadı. Sonuçlar değiştirilmeden kaydedildi.

## Kaynak ve kapsam farklılıkları

Dataverse bu ortamda 403 verdi. Hash/commit sabitli topluluk aynası kullanıldı; resmî güncel sürüm eşdeğerliği doğrulanmadı. reports/data_manifest.json kaynakları içerir.
Resmî PrimeKG deposu OptimusKG güncellemesini bildirir (https://github.com/mims-harvard/PrimeKG). PDF PrimeKG CSV MVP baseline burada uygulandı; OptimusKG Parquet şema adaptörü yok.
RDKit tautomer enumerator üst sınırına ulaşan karmaşık moleküller olabilir; sınırlılık teknik raporda belirtilir.

## Henüz gerçekleştirilmemiş deneyler

- Full profil / 100–200 embedding boyutu ve 200 epoch validation araması.
- Çoklu seed, ablasyon, bağımsız dış veri doğrulaması ve prospektif deney.
- Olasılık kalibrasyonu, scaffold-cluster bootstrap ve MLP kimyasal katkı açıklaması.
- CUDA uyumluluğu/performance: mevcut ortam CPU-only.

## Kullanım

`bash scripts/start_app.sh` → http://localhost:8501
`bash scripts/run_smoke.sh --skip-download` → bütün kısa profil ve testler.
`repurpose --config configs/full.yaml run` → ayrı uzun deney çıktıları.

Ayrıntılı PDF incelemesi: docs/pdf_analysis.md. Adım eşlemesi: PLAN.md. Kurulum/komutlar: README.md.