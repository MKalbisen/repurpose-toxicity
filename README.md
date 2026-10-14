# İlaç yeniden konumlandırma ve Tox21 araştırma projesi

PrimeKG üzerinde ilaç–hastalık bağlantı tahmini ve moleküler Tox21 assay aktivitesi tahminini birleştiren retrospektif araştırma. Çıktılar klinik etkinlik/güvenlilik kanıtı değildir.

**11 Ekim 2026 güncel durum:** akademik veri hazırlığı, Windows/CUDA ortamı, model seçimi/test ayrımı, çoklu seed eğitimleri ve istatistik altyapısı geliştirilmiştir. Tam KGE eğitimi ve final test henüz tamamlanmamıştır. Güncel sayısal durum: [inceleme raporu](reports/academic/REVIEW.md). Eski smoke sonuçları final akademik sonuç sayılmaz.

- [Yöntem, kaynaklar ve sınırlılıklar](docs/academic/METHODS.md)
- [Başka GPU'da eğitim ve geri getirilecek dosyalar](docs/academic/REMOTE_TRAINING.md)
- [Aşama ve kanıt kayıtları](ACADEMIC_PROGRESS.md)
- [Deney ayarları](configs/academic_run.yaml)

## Yerelde çalıştırma — Windows

Proje kökünde PowerShell:

```powershell
.\.venv\Scripts\python.exe environment_check.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Mevcut ortam Python 3.11 ve CUDA destekli PyTorch 2.11.0 kullanır. `.venv`, mevcut CUDA PyTorch'u kullanmak için system-site-packages ile oluşturulmuştur; bağımsız temiz ortam doğrulaması olarak sunulmaz. Eski WSL kurulumu bu bilgisayarda bulunamadı. Platforma bağlı eski paket envanteri `requirements-legacy-wsl.txt` içinde korunmuştur.

Yeni bilgisayarda Python 3.11 ile sanal ortam oluşturun, uygun CUDA PyTorch wheel'ini kurun, ardından:

```bash
python -m pip install -r requirements-academic-lock.txt
python -m pip install -e '.[dev,academic]'
```

`requirements.txt` taşınabilir kurulum girişidir. Kilitli mevcut deneyi sürdürmek için `requirements-academic-lock.txt` sürümleri gereklidir. GPU wheel seçimi sürücüye uygun olmalıdır; `torch.cuda.is_available()` ile kontrol edin.

Windows'ta eski A00–A02 kayıtlarını yeniden doğrulamak için `powershell -ExecutionPolicy Bypass -File scripts/resume_academic.ps1` kullanın. Bu sarmalayıcı yerel CUDA DLL yükleme sırası sorununu giderir. Yeni `academic.experiment` girişi CUDA'yı önce başlatır.

## Akademik deney akışı

```powershell
.\.venv\Scripts\python.exe -m repurpose.academic.experiment prepare
.\.venv\Scripts\python.exe -m repurpose.academic.experiment pilot
.\.venv\Scripts\python.exe -m repurpose.academic.experiment freeze
.\.venv\Scripts\python.exe -m repurpose.academic.experiment train-toxicity
# Tam KGE araması için başka GPU öneriliyor; ölçülen süreyi önce okuyun.
.\.venv\Scripts\python.exe -m repurpose.academic.experiment train-kg
```

Hazırlanmış veri hash'leri doğruysa `prepare` yeniden üretmez. `freeze` kod/veri/ayar/paket sürümlerini kilitler. Eğitim komutları test tahmini veya test metriği üretmez. MLP/KGE epoch checkpoint'inden devam eder; biten klasik model aileleri saklanır. Tam eğitimleri bitirmeden final sonuç raporlamayın.

Final değerlendirme ayrı ve kayıtlıdır:

```powershell
.\.venv\Scripts\python.exe -m repurpose.academic.experiment evaluate-toxicity
.\.venv\Scripts\python.exe -m repurpose.academic.experiment evaluate-kg
```

Veri: `data/processed/academic-v2/{toxicity,kg}`. Eğitim/kilit/çıktı: `reports/academic/runs/academic-v2`. Paket: `python -m repurpose.academic.experiment export` → `exports/academic-v2-compute.zip`.

## Bu sürümün bilimsel önlemleri

- Moleküller parent/tautomer/scaffold/fingerprint ilişkilerinin geçişli bileşenleriyle bölünür; bütün bu kimliklerin train/validation/calibration/test overlap değeri sıfırdır. Tamamlanmamış tautomer üretimi karantinaya alınır.
- KG tüm uygun dört entity türünü kapsar. Endikasyonlar tek yönde tekilleştirilir; ters ilişkiler ve aynı çiftin diğer ilişkileri aynı bölümde tutulur; train coverage ölçülür.
- Toksisite: prevalence, Logistic Regression, XGBoost ve MLP; beş seed, development-only seçim, ayrı calibration grupları.
- KG: TransE/RotatE, ilaç adayları arasında endikasyon filtered MRR; random/popularity/shared-gene baselines.
- Sabit test kapısı, artefakt hash'leri, assay/macro component-bootstrap, eşlenmiş model farkları, hastalık-bootstrap ve seed dağılımı.

Split gruplarının ayrılması tüm kimyasal benzerliği ortadan kaldırmaz. Acyclic grup büyük ve yalnız train'dedir. Bu çalışma MoleculeNet'in varsayılan Tox21 split'iyle aynı değildir. Sonuçlar doğrudan leaderboard üstünlüğü olarak sunulamaz.

## Veri kaynakları ve uygulama

[PrimeKG resmî depo](https://github.com/mims-harvard/PrimeKG), [Tox21 DeepChem loader](https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/tox21_datasets.py). Kaynak hash/commit kayıtları `reports/data_manifest.json` içindedir. PrimeKG topluluk aynası sürümü resmî güncel sürüm olarak sunulmaz; kaynak veri lisansları ayrıca geçerlidir.

Streamlit şu anda eski smoke modellerini gösterir ve bunu açıkça belirtir. Yeni akademik ağırlıkların entegrasyonu, tam KGE eğitimi ve kimyasal eşleme kapsamı tamamlandıktan sonra yapılmalıdır. Eski `full` eğitim girişi akademik test kapısını atladığı için kapatılmıştır. Smoke akışı mühendislik kontrolü için korunur.

Eski kullanım ve deney notları: [önceki README](docs/README_legacy.md), [MVP durum raporu](reports/implementation_status.md). Bunların tarihli sonuçları v2 sonucu olarak yorumlanmamalıdır.
