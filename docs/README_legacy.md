# İlaç Yeniden Konumlandırma ve Toksisite

Klasördeki 6 sayfalık PDF'nin 21 adımını uygulayan Python 3.11 araştırma projesi. PrimeKG → sızıntısız KG split → PyKEEN TransE/RotatE → doğrulanmış SMILES → RDKit ECFP4 → Tox21 XGBoost/çok görevli MLP → risk/sıralama → Streamlit.

Gereksinim–çıktı eşlemesi [PLAN.md](PLAN.md), ayrıntılı deney durumu [reports/implementation_status.md](reports/implementation_status.md).

## Çalıştırma — mevcut WSL ortamı

Proje klasöründe:

```bash
source .venv/bin/activate
python environment_check.py
pytest -q
streamlit run app/main.py
```

Tarayıcı: http://localhost:8501. `smoke` profili hazır kısa gerçek-veri deneyini gösterir. Hastalık seçin, **Analiz Et** düğmesine basın. Arayüz eğitim çalıştırmaz; aday tablosu, CSV indirme, 12 assay grafiği, PyVis alt çizge, RDKit 2D/3D yapı ve TreeSHAP bit katkıları sunar.

WSL'de paket kurulum hızı için `.venv` Linux diskindeki `/home/kalbisen/.local/share/venvs/repurpose-toxicity` ortamına bağlıdır. Başka bilgisayara taşırken aşağıdaki kurulumla yeni ortam oluşturun.

## Sıfırdan kurulum

Python **3.11** gereklidir. CPU deneyleri için:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps
python environment_check.py
```

GPU kullanımı için PyTorch'u donanımınıza uygun resmi dağıtımdan kurun ve YAML'de `device: cuda` seçin. requirements CPU kurulumu bu ortamda doğrulanmıştır; CUDA bu ortamda yoksa GPU uyumluluğu doğrulanmış sayılmaz.

## Pipeline — sırayla

```bash
repurpose --config configs/smoke.yaml download
repurpose --config configs/smoke.yaml prepare-kg
repurpose --config configs/smoke.yaml map-drugs
repurpose --config configs/smoke.yaml train-kg
repurpose --config configs/smoke.yaml prepare-tox21
repurpose --config configs/smoke.yaml train-toxicity
repurpose --config configs/smoke.yaml infer
repurpose --config configs/smoke.yaml report
```

Hepsi: `repurpose --config configs/smoke.yaml run`. Veri zaten varsa `--skip-download` kullanılabilir. Özel sorgu: `repurpose --config configs/smoke.yaml infer --disease 'disease:SOURCE_ID'` (ID için entities.csv).

Uzun deney: `repurpose --config configs/full.yaml run`. Bu profil bütün uygun çizgeyi kullanır, 100/200 boyut ve .001/.01 öğrenme oranlarını aynı split üzerinde arar, maksimum 200 epoch ve validation erken durdurma uygular. CPU'da çok uzun sürebilir; çizge boyutuna göre RAM/GPU gereksinimi değerlendirilmelidir. Full deney, kısa çalışma sonuçlarından ayrı dizinlerde saklanır.

## Veri ve lisans kaydı

- Güncellik: [Resmî PrimeKG deposu](https://github.com/mims-harvard/PrimeKG) OptimusKG güncellemesini bildirir. Bu proje PDF'nin PrimeKG CSV MVP'sini sabit baseline olarak uygular; OptimusKG şeması desteklenmez.
- PrimeKG resmî kaynak: https://github.com/mims-harvard/PrimeKG ve DOI https://doi.org/10.7910/DVN/IXA7BM. Resmî metadata ile checksum kontrol edilir.
- Resmî Dataverse erişilemezse hash ve commit sabitli `hieupth/primekg` topluluk aynası kullanılır. Bu sürüm **en güncel resmî sürüm olarak sunulmaz**. Kaynak ve erişim hatası `reports/data_manifest.json` içine kaydedilir.
- Tox21 dağıtımı: https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/tox21_datasets.py. İndirilen dosyanın SHA256 değeri manifesttedir.
- PrimeKG yazılım lisansı MIT; alttaki kaynak veri lisansları ayrıdır. DrugBank için kendi lisans koşulları geçerlidir. Ham veriler ve ağırlıklar Git'e eklenmez.
- Kimyasal eşleme: PubChem PUG REST ile DrugBank ID → CID/SMILES; CID synonym listesinde aynı DrugBank kimliği bulunmalıdır. İsim eşleşmesi tek başına yeterli değildir.
- Manuel doğrulanmış eşleme gerekiyorsa `data/mappings/verified_drugs.csv` sağlayın: `drug_id,smiles,source,verified`. `source` gerçek kaynak URL'si, `verified` true olmalıdır. Çoklu eşlemeyi birleştirmeyin; insan tarafından tek kimlik seçilmelidir.

## Bilimsel protokol

KG entity kimlikleri `type:source_id`. Ters yönler ve aynı çift üzerindeki bütün ilişkiler bir grupta kalır; eğitim entity/relation coverage korunur. Bölme yaklaşık %80/10/10; gerçekleşen oran raporlanır. TransE/RotatE model ve ayar seçimi validation MRR üzerinden, test filtered MRR ve Hits@1/3/10 yalnızca son seçilmiş modellerde hesaplanır.

Tox21 eksik etiketleri NaN'dır, hiçbir zaman negatif sayılmaz. Aynı canonical SMILES yinelenirse etiketler birleştirilir; çelişkiler NaN'a çevrilip raporlanır. RDKit canonical stereokimyalı SMILES korunur; fingerprint radius=2, 1024 bit ve chirality kapalıdır. Scaffold grupları için salt parent, canonical tautomer ve achiral Bemis–Murcko kullanılır. Acyclic moleküllerin tümü aynı grupta kalır; scaffold overlap sıfır olmalıdır.

XGBoost her assay için ayrı eğitilir, class weight train üzerinden hesaplanır; parametreler validation AP ile seçilir. MLP 1024→512→256→12, ReLU/Dropout/BatchNorm ve maskeli BCEWithLogits uygular. Validation macro ROC-AUC checkpoint seçer. ROC-AUC, AP (PR-AUC), precision, recall, F1, balanced accuracy ve bootstrap CI görev bazında raporlanır. Tek sınıflı görevlerde tanımsız metrikler boş bırakılır.

Sıralamalar: hard filter (tüm 12 assay biliniyor ve max risk < eşik), ağırlıklı normalize KG − max risk ve Pareto. Bilinmeyen risk güvenli sayılmaz. Varsayılan .7 ağırlık bir araştırma tercihidir; 0–1 duyarlılık tablosu kaydedilir. Tahminler kalibre edilmemiş assay aktivitesi olasılığıdır, klinik toksisite/etkinlik kanıtı değildir.

## Dizinler

`src/repurpose/{data,kg,toxicity,integration,visualization}` modülleri; `app/{components,services}` kokpit; `configs` deney ayarları; `data/raw` indirilenler; `data/processed/{profile}` temiz veriler/split; `models/{profile}` ağırlıklar; `reports/{profile}` sayısal çıktılar ve teknik PDF; `tests` bilimsel kuralları ve uçtan uca akışı doğrular.

Rapor hiçbir gerçekleştirilmemiş deneyi tamamlanmış göstermez. Hits@10≥.70 ve mean ROC-AUC>.80 PDF'deki hedeflerdir; garanti değildir. Çoklu seed, ablasyon, dış veri doğrulaması ve full eğitim durumu uygulama durum dosyasında açıkça belirtilir.

## Akademik çalışma ve kaldığı yerden devam

[ACADEMIC_PLAN.md](ACADEMIC_PLAN.md) ana akademik protokol/aşama planıdır.
[ACADEMIC_PROGRESS.md](ACADEMIC_PROGRESS.md) tamamlanan işler, eklenen dosyalar ve sıradaki adımı gösterir.
Ayrıntılı aşama/hash kayıtları `reports/academic/state.json`, olay günlüğü `reports/academic/events.jsonl` içindedir.

```bash
source .venv/bin/activate
python -m repurpose.academic.cli status
python -m repurpose.academic.cli resume
```

`bash scripts/resume_academic.sh` aynı hazırlık akışını çalıştırır. Geçerli kanıtı olan adımları atlar; değişmiş/eksik artefaktları yeniden doğrular. Şu anda A00–A02 (kayıt, bilimsel audit, hedef metrik altyapısı) uygulanmıştır. Bu komut henüz uygulanmamış A03–A10 eğitimlerini başlatmaz. A03'ün somut devam işi `docs/academic/A03_WORK_ORDER.md` içindedir. Aşama kayıtlarından devam vardır; uzun eğitimlerde epoch/checkpoint resume desteği daha sonraki aşamada uygulanacaktır.

Bilimsel denetim `reports/academic/audit.md` içinde. 11 identical fingerprint grubu mevcut smoke split sınırlarını aşar; bunlar model-input eşitliğidir ve yeni akademik gruplamada birlikte tutulmalıdır. Smoke model/test sonuçları akademik final sonuç değildir. Kullanıcı şu an CPU kullanır; T4/L4 gerektiğinde sağlanabilir.
