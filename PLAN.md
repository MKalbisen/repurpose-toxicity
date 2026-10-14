# PDF'ye bağlı uygulama planı

Kaynak: **İlaç Yeniden Konumlandırma ve Toksisite Yol Haritası (1).pdf**, 6 sayfa, 21 uygulama adımı ve 12 aşamalı özet matris. Son sayfadaki tablolar görsel olarak da incelendi.

## Gereksinimler ve kabul ölçütleri

| PDF adımı | Uygulama | Kabul kanıtı |
|---|---|---|
| 1 | Python 3.11, izole ortam, sabit bağımlılıklar, modüler dizinler | environment_check.json, requirements.txt |
| 2 | Resmi PrimeKG verisini indir, şema/lisans kaydı, alt çizge | entities.csv, relations.csv, triples.tsv, data_dictionary.md |
| 3 | Kimyasal kimlik, canonical stereokimyalı SMILES, belirsiz eşleşmeleri ayır | drug_chemical_mapping.csv, unmatched_drugs.csv |
| 4 | %80/10/10 coverage ve ters kenar grubu temelli KG split | split_report.json, sızıntı testleri |
| 5 | TransE, seed, erken durdurma, filtered MRR/Hits | transe_model/, kg_metrics_transe.csv |
| 6 | RotatE, aynı split, validation ile model/parametre seçimi | kge_comparison.csv, best_kge_config.json |
| 7 | Disease → indication ilişkisiyle deterministik ilaç adayları | candidate_results.csv |
| 8 | Tox21 12 görev, geçersiz SMILES ve yinelenen/çelişen etiketler | tox21_clean.csv, invalid_smiles.csv, label_report.csv |
| 9 | Radius=2, 1024 bit Morgan, sabit standardizasyon | tox21_features.npz, fingerprint_metadata.json |
| 10 | Bemis–Murcko scaffold grupları, sıfır overlap | tox21_train/val/test.csv, scaffold_split_report.json |
| 11 | Her assay için XGBoost, mask, class weight, validation seçim | xgb_models/, xgb_metrics.csv, xgb_predictions.csv |
| 12 | 1024→512→256→12 MLP, BN/Dropout, maskeli BCE logits | mlp_best.pt, mlp_metrics.csv, training_history.csv |
| 13 | Görev bazlı ROC/PR-AUC, precision/recall/F1/balanced accuracy, bootstrap | toxicity_final_metrics.csv, evaluation_report.md |
| 14 | KG + kimya + 12 risk, eşleşmeyen adaylarda bilinmeyen risk | integrated_candidates.csv, final_ranking.json |
| 15 | Hard filter, ağırlıklı sıralama, Pareto, ağırlık duyarlılığı | final_candidates.csv, ranking_strategy.md, sensitivity.csv |
| 16 | Eğitim yapmayan, modelleri cache eden Streamlit kokpit | app/main.py, Streamlit AppTest |
| 17 | Sınırlı 1–2 hop PyVis alt çizge | graph_builder.py, graph_view.py |
| 18 | RDKit SVG ve ETKDG/Py3Dmol 3D, hata halinde mesaj | molecule_view.py |
| 19 | XGBoost bit katkıları, yerel çizge patikaları, skor ayrıştırma | xai.py, açıklama paneli |
| 20 | Birim/entegrasyon testleri, seed ve gecikme kayıtları | tests/, experiment_log.csv |
| 21 | Gerçek yürütme çıktılarından teknik rapor ve mimari | final_report.md/pdf, architecture.png |

## Uygulama sırası

1. Ortam ve bu plan; ardından veri kaynağı/lisans manifesti.
2. Veri indirme, şema inceleme, kimlik eşleme ve sızıntısız KG split.
3. TransE ve RotatE eğitim/değerlendirme, validation seçimi.
4. Tox21 temizlik, fingerprint ve scaffold split.
5. XGBoost ve MLP eğitim, değerlendirme ve kıyaslama.
6. Birleşik çıkarım, sıralama ve duyarlılık analizi.
7. Streamlit, alt çizge, molekül ve açıklanabilirlik bileşenleri.
8. Testler, kısa gerçek-veri çalıştırması, deney kayıtları ve rapor.

## Deney kapsamı

- `smoke`: **gerçek veri** üzerinde sınırlı KG, kısa eğitim ve düşük bootstrap sayısı. Yazılımın çalışmasını doğrular; akademik performans sonucu değildir.
- `full`: bütün uygun PrimeKG alt çizgesi, 100–500 epoch aralığı, daha geniş validation parametre araması ve bootstrap. Uzun deney ayrı komutla tekrarlanabilir.
- Aynı model ailesinin tüm ayarları validation ile seçilir. Test, seçim tamamlandıktan sonra kullanılır.
- Hits@10 ≥ .70 ve ortalama ROC-AUC > .80 yalnızca PDF'nin deneysel hedefleridir.
- Toksisite olasılıkları `predicted assay risk` olarak sunulur; etkinlik veya klinik güvenlilik iddiası yapılmaz.
- PrimeKG yazılım MIT lisansı veri kaynaklarına uygulanmaz. DrugBank verisi yeniden dağıtılmaz.
- Eksik SMILES veya başarısız assay modeli düşük risk sayılmaz; sıralamada bilinmeyen olarak işaretlenir.

Güncel yürütme ve tamamlanma durumu: `reports/implementation_status.md`.

## Akademik devam planı

Kullanıcının akademik geliştirme talebiyle ayrı [ACADEMIC_PLAN.md](ACADEMIC_PLAN.md) açıldı.
Yeni eklemeler ve bir sonraki iş [ACADEMIC_PROGRESS.md](ACADEMIC_PROGRESS.md) içinde takip edilir.
Mevcut MVP deney raporu korunur; akademik aşamalar bunun tamamlandığı anlamına gelmez.
