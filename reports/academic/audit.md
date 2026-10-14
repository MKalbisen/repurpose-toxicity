# Akademik hazırlık denetimi

Tarih: 2026-10-08T12:23:43.248188+00:00
Mevcut smoke verileri/kodu incelendi; yeni final model test skoru üretilmedi.

KG: 231 entity, 1800 triple. Kaynak: pinned_community_mirror.
Tox21: 7822 molekül; 16012 eksik assay etiketi.
Kimyasal eşleme: 36/50.

## Ölçülen sızıntı/coverage kontrolleri

KG: {"train_validation_pair_overlap": 0, "train_validation_exact_overlap": 0, "train_test_pair_overlap": 0, "train_test_exact_overlap": 0, "validation_test_pair_overlap": 0, "validation_test_exact_overlap": 0, "validation_unseen_entities": 0, "validation_unseen_relations": 0, "test_unseen_entities": 0, "test_unseen_relations": 0, "partition_unique_triples_complete": true}
Tox21: {"feature_shape_matches": true, "label_shape_matches": true, "binary_features": true, "binary_or_missing_labels": true, "split_covers_every_row_once": true, "npz_labels_match_csv": true}
Split kesişen kimlik grupları: {"canonical_smiles": 0, "scaffold": 0, "fingerprint": 11}
Scaffold dağılımı: {"train": {"n": 6258, "groups": 947, "largest_group": 1780, "acyclic_count": 1780}, "val": {"n": 782, "groups": 674, "largest_group": 3, "acyclic_count": 0}, "test": {"n": 782, "groups": 674, "largest_group": 3, "acyclic_count": 0}}

## Açıklar ve giderecek aşama

| ID | Öncelik | Aşama | Bulgu |
|---|---|---|---|
| F01 | critical | A02 | Legacy KGE selection/evaluation uses all relations and all entity types; primary drug-only indication metric missing. |
| F02 | critical | A04 | Legacy training automatically evaluates test after each run; separate train/select/final-evaluate gate required. |
| F03 | high | A03 | Smoke subgraph selection favors high-degree drugs and input file order; cannot be the general academic cohort. |
| F04 | high | A02 | Legacy confidence intervals resample individual molecules; group and paired uncertainty missing. |
| F05 | high | A03 | Tautomer enumeration completion/identity audit absent; rejected molecule and limit handling must be versioned. |
| F06 | high | A03 | Mirror is hash-verified but official version/provenance/license equivalence remains unverified. |
| F07 | high | A04 | Legacy paths key on profile only, model artifacts overwrite, optimizer/RNG epoch resume absent. |
| F08 | high | A06 | Only one short training seed; classical baselines, grouped calibration and fixed task denominator comparison absent. |
| F09 | high | A09 | No independent external validation. Previous smoke test exposure must be disclosed. |
| F10 | medium | A03 | No nearest-training chemistry similarity audit; zero scaffold overlap alone does not demonstrate independence. |
| F11 | medium | A04 | Full KG splitting constructs Python groups and loops over dataframe groups; scalability must be measured before full run. |
| F12 | critical | A03 | Equivalent chemistry/model inputs cross split boundaries; fix before academic training. |

Kod/veri hash envanteri audit.json içinde. Ayrıntılı assay dağılımı assay_label_audit.csv içinde.
A01 denetimin tamamlanmasıdır; kritik açıkların giderildiği veya akademik sonuçların hazır olduğu anlamına gelmez.
