# İlaç Yeniden Konumlandırma ve Toksisite — Teknik Rapor

Profil: smoke; seed: 42.
Amaç: bilgi çizgesi bağlantı skorunu moleküler assay riskiyle birlikte incelemek.
Kısa profil sonuçları çalışma doğrulamasıdır; tam deney sonucu olarak yorumlanamaz.

## Veri ve lisans
PrimeKG DOI: 10.7910/DVN/IXA7BM. Kaynak manifesti reports/data_manifest.json içindedir.
PrimeKG yazılımının MIT lisansı, DrugBank dahil kaynak verilerin lisansını kapsamaz.
Tox21: DeepChem MoleculeNet dağıtımı; 12 in vitro assay. Kaynak URL ve hash manifesttedir.

## Yöntem
KG: type:source_id kimlikleri; aynı çift arasındaki tüm yön/ilişkiler tek splitte.
TransE/RotatE validation MRR ile seçilir; filtered test tüm bilinen pozitifleri filtreler.
Tox21: canonical isomeric SMILES, duplicate çelişkiler NaN; eksikler negatif sayılmaz.
Morgan radius=2, 1024 bit. Fragment/tautomer standardizasyonu scaffold grup kimliğinde kullanılır.
Scaffold grupları bölünmez; acyclic moleküller tek grup. Oranlar yaklaşık %80/10/10.
XGBoost: görev başına class weight, validation AP seçimi. MLP: 1024/512/256/12, maskeli BCE.
MLP validation macro ROC-AUC ile erken durdurulur. Test model seçiminde kullanılmaz.
PR-AUC average precision; %95 bootstrap CI yalnızca iki sınıflı çekimlerde hesaplanır.

## Gerçekleşen deneyler

### Kaynak manifesti
{
  "retrieved_unix": 1791460308.431595,
  "primekg_doi": "doi:10.7910/DVN/IXA7BM",
  "origin": "pinned_community_mirror",
  "official_access_error": "403 Client Error: Forbidden for url: https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi%3A10.7910%2FDVN%2FIXA7BM",
  "primekg_version": "mirror commit 03de2c6613831c66203fb8af8ac7f3fdae6b1887; official version unverified",
  "license_metadata": "See PrimeKG original source licenses; mirror does not grant new rights.",
  "sources": [
    {
      "url": "https://huggingface.co/datasets/hieupth/primekg/resolve/03de2c6613831c66203fb8af8ac7f3fdae6b1887/org/kg.csv",
      "file": "data/raw/primekg/primekg_raw.csv",
      "sha256": "0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69"
    },
    {
      "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
      "file": "data/raw/tox21/tox21.csv.gz",
      "sha256": "45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360"
    }
  ],
  "license_note": "PrimeKG software MIT != underlying source licenses; DrugBank is restricted."
}

Resmî PrimeKG deposu OptimusKG güncellemesini bildirir; bu deney sabit PrimeKG MVP baseline kullanır.
Kaynak: https://github.com/mims-harvard/PrimeKG ; OptimusKG şema desteği uygulanmadı.

### Otomatik testler
{"name": "pytest", "errors": "0", "failures": "0", "skipped": "0", "tests": "12", "time": "12.506", "timestamp": "2026-10-08T12:02:56.981587+00:00", "hostname": "TOFAVDDEV-0"}

### kg_data_report.json
{
  "source_rows": 8100498,
  "source_relations": {
    "protein_protein": 642150,
    "drug_drug": 2672628,
    "contraindication": 61350,
    "drug_protein": 51306,
    "indication": 18776,
    "off-label use": 5136,
    "disease_phenotype_positive": 300634,
    "phenotype_phenotype": 37472,
    "phenotype_protein": 6660,
    "disease_phenotype_negative": 2386,
    "disease_protein": 160822,
    "drug_effect": 129568,
    "bioprocess_bioprocess": 105772,
    "disease_disease": 64388,
    "molfunc_protein": 139060,
    "cellcomp_protein": 166804,
    "molfunc_molfunc": 27148,
    "cellcomp_cellcomp": 9690,
    "bioprocess_protein": 289610,
    "anatomy_protein_present": 3036406,
    "pathway_protein": 85292,
    "anatomy_anatomy": 28064,
    "pathway_pathway": 5070,
    "exposure_disease": 4608,
    "exposure_exposure": 4140,
    "exposure_bioprocess": 3250,
    "exposure_protein": 2424,
    "exposure_molfunc": 90,
    "exposure_cellcomp": 20,
    "anatomy_protein_absent": 39774
  },
  "source_endpoint_types": {
    "gene/protein": 5262458,
    "drug": 5611392,
    "disease": 682488,
    "effect/phenotype": 514192,
    "biological_process": 504404,
    "molecular_function": 193446,
    "cellular_component": 186204,
    "anatomy": 3132308,
    "exposure": 18672,
    "pathway": 95432
  },
  "missing": {
    "relation": 0,
    "x_id": 0,
    "x_type": 0,
    "x_name": 0,
    "y_id": 0,
    "y_type": 0,
    "y_name": 0
  },
  "entities": 231,
  "triples": 1800,
  "profile": "smoke"
}

### split_report.json
{
  "seed": 42,
  "sizes": {
    "train": 1440,
    "validation": 180,
    "test": 180
  },
  "requested_ratios": [
    0.8,
    0.1,
    0.1
  ],
  "actual_ratios": [
    0.8,
    0.1,
    0.1
  ],
  "pair_group_overlap": [
    0,
    0,
    0
  ],
  "entity_coverage": true,
  "relation_coverage": true,
  "policy": "All relations and directions for an unordered endpoint pair stay in one split."
}

### fingerprint_metadata.json
{
  "rdkit_version": "2026.03.6",
  "radius": 2,
  "n_bits": 1024,
  "binary": true,
  "include_chirality": false,
  "canonical_smiles": "isomeric, charge/stereo/salt retained",
  "split_identity": "fragment parent + canonical tautomer + achiral Murcko; acyclic together",
  "shape": [
    7822,
    1024
  ],
  "seconds": 115.52872743199987,
  "raw_rows": 7831,
  "clean_rows": 7822,
  "invalid_rows": 8,
  "collapsed_duplicates": 0,
  "invalid_scaffold_rows": 1,
  "conflicting_labels": 0
}

### scaffold_split_report.json
{
  "seed": 42,
  "sizes": {
    "train": 6258,
    "validation": 782,
    "test": 782
  },
  "scaffold_overlap": [
    0,
    0,
    0
  ],
  "requested_ratios": [
    0.8,
    0.1,
    0.1
  ],
  "actual_ratios": [
    0.8000511378164152,
    0.09997443109179238,
    0.09997443109179238
  ],
  "acyclic_policy": "all empty scaffolds stay together",
  "group_count": 2295
}

### mapping_report.json
{
  "total_drugs": 50,
  "queried_this_run": 32,
  "verified_structures": 36,
  "unmatched": 14,
  "policy": "DrugBank ID must appear in PubChem CID synonyms or explicit curated verification."
}

### inference_timing.json
{
  "seconds": 0.10616749900009381,
  "candidate_count": 30,
  "complete_risk_candidates": 24,
  "cold_model_load_excluded": true
}

### kge_comparison.csv
 model  dimension    lr split      MRR   Hits@1   Hits@3  Hits@10  validation_MRR
TransE         32 0.005  test 0.019659 0.000000 0.005556 0.030556        0.020319
RotatE         32 0.005  test 0.084608 0.027778 0.058333 0.177778        0.081808

### model_comparison.csv
  model  ROC-AUC   PR-AUC       F1  Balanced Accuracy  evaluable_assays
    MLP 0.744105 0.346993 0.321205           0.683230                12
XGBoost 0.706299 0.301203 0.278990           0.639067                12

### toxicity_final_metrics.csv
  model         assay   n  positive status  ROC-AUC   PR-AUC  Precision   Recall       F1  Balanced Accuracy  bootstrap_valid  ROC_CI_low  ROC_CI_high  PR_CI_low  PR_CI_high
XGBoost         NR-AR 720        32     ok 0.840298 0.520615   0.164948 0.500000 0.248062           0.691134               40    0.772489     0.910291   0.380878    0.701803
XGBoost     NR-AR-LBD 638        23     ok 0.842983 0.383724   0.206897 0.521739 0.296296           0.723471               40    0.784081     0.941172   0.253596    0.617637
XGBoost        NR-AhR 626        95     ok 0.780375 0.408206   0.240122 0.831579 0.372642           0.680385               40    0.749103     0.816167   0.364892    0.489483
XGBoost  NR-Aromatase 521        51     ok 0.647184 0.248074   0.156627 0.509804 0.239631           0.605966               40    0.587445     0.727264   0.186149    0.397261
XGBoost         NR-ER 571        76     ok 0.603495 0.276819   0.182320 0.434211 0.256809           0.567610               40    0.543113     0.658403   0.201115    0.363037
XGBoost     NR-ER-LBD 676        32     ok 0.649384 0.347745   0.144737 0.343750 0.203704           0.621409               40    0.555147     0.743339   0.193334    0.449045
XGBoost NR-PPAR-gamma 590        30     ok 0.666994 0.121058   0.104651 0.300000 0.155172           0.581250               40    0.627056     0.747031   0.070303    0.212113
XGBoost        SR-ARE 489       107     ok 0.682524 0.390018   0.295455 0.728972 0.420485           0.621030               40    0.641166     0.736122   0.333141    0.471774
XGBoost      SR-ATAD5 679        33     ok 0.732456 0.136241   0.128834 0.636364 0.214286           0.708275               40    0.619028     0.804979   0.094831    0.262416
XGBoost        SR-HSE 597        54     ok 0.696815 0.235446   0.200000 0.425926 0.272189           0.628248               40    0.645173     0.780920   0.142496    0.337171
XGBoost        SR-MMP 514       114     ok 0.687478 0.375166   0.346535 0.614035 0.443038           0.642018               40    0.626233     0.741556   0.328165    0.479166
XGBoost        SR-p53 639        60     ok 0.645596 0.171323   0.145631 0.500000 0.225564           0.598014               40    0.561432     0.729921   0.120072    0.231722
    MLP         NR-AR 720        32     ok 0.752316 0.507835   0.132867 0.593750 0.217143           0.706759               40    0.676364     0.869708   0.361707    0.694147
    MLP     NR-AR-LBD 638        23     ok 0.865465 0.414236   0.250000 0.695652 0.367816           0.808802               40    0.799994     0.939445   0.283938    0.597239
    MLP        NR-AhR 626        95     ok 0.822500 0.521748   0.336364 0.778947 0.469841           0.751997               40    0.794313     0.864737   0.453240    0.592369
    MLP  NR-Aromatase 521        51     ok 0.741239 0.306515   0.250000 0.627451 0.357542           0.711598               40    0.672036     0.812503   0.226668    0.434522
    MLP         NR-ER 571        76     ok 0.641414 0.300764   0.211957 0.513158 0.300000           0.610114               40    0.589505     0.703665   0.236727    0.384307
    MLP     NR-ER-LBD 676        32     ok 0.665906 0.259890   0.125000 0.375000 0.187500           0.622283               40    0.563198     0.754681   0.132682    0.432269
    MLP NR-PPAR-gamma 590        30     ok 0.757143 0.170005   0.113402 0.366667 0.173228           0.606548               40    0.710714     0.829808   0.093755    0.315664
    MLP        SR-ARE 489       107     ok 0.749180 0.457681   0.378947 0.672897 0.484848           0.681998               40    0.701500     0.795480   0.377823    0.533173
    MLP      SR-ATAD5 679        33     ok 0.800990 0.259508   0.160839 0.696970 0.261364           0.755606               40    0.727746     0.871868   0.170000    0.450798
    MLP        SR-HSE 597        54     ok 0.689414 0.273247   0.204082 0.370370 0.263158           0.613362               40    0.627207     0.748075   0.181828    0.376435
    MLP        SR-MMP 514       114     ok 0.762840 0.444405   0.432927 0.622807 0.510791           0.695154               40    0.705464     0.810375   0.390527    0.541029
    MLP        SR-p53 639        60     ok 0.680858 0.248080   0.172973 0.533333 0.261224           0.634542               40    0.618026     0.749424   0.159460    0.319165

### experiment_log.csv
          step    seconds  seed profile    status    unix_time
      download  24.603144    42   smoke completed 1.791460e+09
    prepare-kg  49.805778    42   smoke completed 1.791460e+09
     map-drugs  31.640207    42   smoke completed 1.791461e+09
    prepare-kg  55.156626    42   smoke completed 1.791461e+09
 prepare-tox21 115.557885    42   smoke completed 1.791461e+09
      train-kg  58.190115    42   smoke completed 1.791461e+09
     map-drugs  45.186992    42   smoke completed 1.791461e+09
train-toxicity  49.901899    42   smoke completed 1.791461e+09
         infer   4.701651    42   smoke completed 1.791461e+09
        report   1.003565    42   smoke completed 1.791461e+09
         infer   6.294498    42   smoke completed 1.791461e+09

## Sıralama ve açıklanabilirlik
Hard filter: 12 assay tamamlanmış ve maksimum risk eşikten düşük. Bilinmeyenler elenir.
Ağırlıklı skor: w × min-max KG − (1−w) × maksimum assay riski; eksikler skorlanmaz.
Pareto: KG yüksek ve risk düşük hedefleri. Eşit noktalar birlikte korunur.
sensitivity.csv ağırlık 0/.25/.5/.7/.75/1 sonuçlarını kaydeder; varsayılan .7 klinik optimum değildir.
XGBoost native TreeSHAP bit katkıları log-odds birimindedir; hash çakışmaları açıklamayı sınırlar.
Yerel 1–2 hop çizge patikaları gözlenen ilişkileri gösterir; nedensellik kanıtı değildir.

## Sınırlılıklar ve tamamlanmamış deneyler
Tam PrimeKG ve uzun full profil bu kısa doğrulamanın yerine geçmez; ayrı yürütülmelidir.
Assay riskleri kalibre edilmemiştir. Tox21 assay aktivitesi klinik toksisite ile eşdeğer değildir.
KG hedeflerinin test metrikleri tüm ilişkilere aittir; drug-disease özel prospektif doğrulama yapılmadı.
RDKit tautomer enumeration sınırları geçerlidir; karmaşık moleküllerde tüm tautomerler tüketilmemiş olabilir.
Kimyasal eşleme DrugBank kimliği doğrulanamayan adaylarda bilinmeyendir; kapsam mapping_report içindedir.
Salt ve stereokimya retained; fingerprint chirality kapalı. Scaffold kimliği achiral tautomer parenttır.
Ablasyon/çoklu seed/prospektif klinik doğrulama yapılmadı. Hedef metrikler başarı garantisi değildir.
UI 3–5 sn hedefi gözlenen çıkarım süresiyle karşılaştırılır; cold load ve 3D süreleri ayrıca değişebilir.

## Tekrar üretme
README komutları, configs/*.yaml, requirements.txt ve manifest SHA256 kayıtlarını kullanın.