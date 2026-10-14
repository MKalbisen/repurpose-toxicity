# Akademik inceleme ve yapılan işler

**11 Ekim 2026 — proje artık daha güvenilir bir deney altyapısına sahip; henüz tamamlanmış/yayına hazır bir akademik çalışma değildir.** Veri protokolü, test ayrımı, istatistik, yerel eğitim ve taşınabilir hesaplama paketi hazırlandı. Tam KGE, final değerlendirme ve entegrasyon çalışması henüz tamamlanmadı.

## İnceleme sonucu

| Bulgu | Yapılan işlem / kalan sınır |
|---|---|
| Eski Tox21 split'inde 11 fingerprint eşdeğerlik grubu birden fazla bölümde | Parent/tautomer/scaffold/fingerprint geçişli bileşenleriyle yeni split. Altı kimlik denetiminde overlap=0. |
| Tautomer üretiminin tamamlanması denetlenmiyordu | Completion status kaydı; 93 sınırda molekül karantinaya alındı. Kompleks moleküllerin sistematik dışlanması sınırlılık olarak korunmalı. |
| Smoke KG derece/dosya sırası yanlı küçük alt çizgeydi | Bütün uygun dört entity türü disk üzerinden işlendi; sıralamadan bağımsız split ve gerçek coverage denetimi. |
| KG seçimi genel relation/all-entity MRR kullanıyordu | Yeni eğitimde indication-specific, drug-only filtered MRR ile ayar ve checkpoint seçimi. |
| Eğitim test sonuçlarını otomatik hesaplıyordu | Ayrı train/final-evaluate komutları; veri/kod/ayar/paket kilidi; test erişim kaydı; seçilmiş model hash'leri. Eski full eğitim girişi engellendi. |
| Kalibrasyon ve basit karşılaştırma modelleri eksikti | Prevalence, Logistic Regression, XGBoost, MLP; beş seed; ayrı calibration grupları. |
| Tekil molekül bootstrap'ı belirsizliği yanlış temsil edebilirdi | Component-cluster assay/macro bootstrap ve eşlenmiş model farkları; KG için hastalık-cluster CI ve eşlenmiş RotatE−TransE. |
| Uzun eğitim kesintiden devam etmiyordu | Epoch sonunda optimizer/RNG/model kaydı. CPU MLP resume, kesintisiz eğitime karşı birebir test edildi. |
| Eski kurulum Linux'a ve CPU paketlerine bağlıydı | Windows proje ortamı ve gerçek CUDA doğrulaması; platformlar arası dosya kilidi; taşınabilir bağımlılıklar. |
| Resmî PrimeKG sürümü ve lisans eşdeğerliği belirsiz | 11 Ekim'de Dataverse API tekrar denendi, HTTP 403. Kaynak/commit/hash açıkça korunuyor; belirsizlik çözülmüş sayılmıyor. |
| Bağımsız dış doğrulama yok; smoke test geçmişte görüldü | Retrospektif kapsam açıklandı; yeni split bağımsız dış test sayılmıyor. |

## Gerçek veri çıktıları

**Tox21:** 7831 ham satır → 7729 kabul edilmiş molekül. 93 tamamlanmamış tautomer, 8 geçersiz SMILES, 1 standardizasyon hatası. Train/validation/calibration/test: **5410 / 773 / 773 / 773**. Canonical/parent/tautomer/scaffold/fingerprint/component kesişimlerinin tamamı sıfır. En büyük component 1778 molekül; bütün acyclic yapılar birlikte train'de.

Sıfır kimlik kesişimi uzak kimyasal genelleme garantisi değildir: test→train en yakın Tanimoto medyanı .396, en yüksek değer .964. Benzerlik denetimi yapısaldır, test performansı hesaplanmamıştır.

**PrimeKG:** 8,100,498 ham yönlü satır tarandı. Dört türde 3,766,918 uygun satır, endikasyon tekilleştirmesinden sonra **3,757,160 triple / 47,130 entity**. Train/validation/test: **3,023,797 / 366,872 / 366,491**. Aynı çiftin bölümler arasında kesişimi 0; held-out görülmemiş entity/relation sayısı 0. 7946 drug, 17054 disease, 19614 gene/protein ve 2516 pathway içeriyor.

Veri üretimi bu makinede kimya için yaklaşık 206 saniye, KG için 191 saniye sürdü. `data/processed/academic-v2/*/manifest.json` bütün çıktı hash'lerini ve politikaları içerir.

## Yapılmış eğitimler ve development sonuçları

Toksisite dört model ailesiyle 17/29/43/71/101 seed'lerinde tamamlandı. Toplam ölçülen eğitim süresi yaklaşık **297 saniye**. Sigmoid kalibrasyon ayrı component gruplarında; karar eşiği sabit .5. Logistic/prevalence deterministik olduğundan seed değişimi bunlara yapay varyasyon kazandırmaz.

| Model | Validation macro ROC-AUC, ortalama ± seed SD |
|---|---:|
| Train prevalence | 0.5000 ± 0.0000 |
| Logistic Regression | 0.7347 ± 0.0000 |
| XGBoost | 0.7204 ± 0.0034 |
| MLP | 0.7337 ± 0.0051 |

**Bu tablo final test sonucu değildir.** Raw modellerin seçim yapılan validation bölümündeki değerleridir; seçim iyimserliği içerir. Beş seed SD'si güven aralığı değildir. Bu aşamada karmaşık modellerin Logistic Regression'dan üstün olduğunu söylemek için kanıt yok. Önceden dondurulan deney bu sonuçlara göre değiştirilmedi.

MLP validation performansı erken epoch'larda tepe yapıp sonra düşüyor; checkpoint seçimi/erken durdurma bu nedenle önemlidir. Detaylar `runs/academic-v2/toxicity/*/mlp_history.json`, karşılaştırma `runs/academic-v2/development_summary.csv` içindedir.

**Akademik final test henüz açılmadı:** `toxicity_test_access.json` ve `kg_test_access.json` oluşturulmadı. Tam KGE eğitimi de başlatılmadı. Final metrik/CI komutlarının çalışması küçük sentetik veride test edildi; gerçek veri üzerinde final sonuçları varmış gibi sunulmuyor.

## 3050 Ti ve uzun eğitim kararı

Gerçek cihaz RTX 3050 Ti Laptop GPU, 4096 MiB VRAM; PyTorch 2.11.0+cu130, CUDA çalışıyor. Toplam RAM 15.2 GiB; pilotta yaklaşık 1.65 GiB boş RAM vardı. VRAM kapasitesi bu ayarlarda yeterli; esas kısıt süredir.

| Model | Boyut | Tahmini epoch | 200 epoch + validation için tek deneme |
|---|---:|---:|---:|
| TransE | 100 | 129 sn | 7.3 saat |
| TransE | 200 | 194 sn | 10.9 saat |
| RotatE | 100 | 214 sn | 12.1 saat |
| RotatE | 200 | 288 sn | 16.1 saat |

Beş seed × dört ayar × iki model = 40 deneme; üst epoch bütçesi yaklaşık **464 saat / 19.3 gün**. Bu, 55 ölçümlü batch üzerinden yapılan doğrusal tahmindir. Erken durdurma azaltabilir; termal kısıt, diğer uygulamalar ve checkpoint/disk maliyeti artırabilir. Başka GPU'da aynı süre varsayılmaz. Büyük eğitim kullanıcının belirttiği dış makinede yapılmalı.

Detaylı ölçüm: `runs/academic-v2/pilot/hardware_pilot.json`. Taşıma yönergesi: [REMOTE_TRAINING.md](../../docs/academic/REMOTE_TRAINING.md). Paket: `exports/academic-v2-compute.zip`.

Paket 96,916,368 byte (yaklaşık 97 MB), 120 dosya içerir. ZIP CRC, tüm kilitli kaynak kodu, 14 veri dosyası ve 25 model dosyasının SHA256 değerleri doğrulandı. Ham veri içermez. Paket SHA256: `5aca6e3662a774292514cd1f432b4fd55eeb3d84c3e0cc32b11424687ac8faeb`. Doğrulama kaydı: `package_verification.json`.

## Doğrulama

- **38 test geçti:** legacy veri/ranking/UI akışı, geçişli molekül grupları, KG dosya sırası bağımsızlığı, coverage/ters ilişki sızıntısı, test erişim kapısı, değişmiş model/hash reddi, eğitimde test verisine erişilmemesi, gerçek küçük TransE/RotatE ve toksisite eğitim/final-evaluation akışları, MLP checkpoint resume eşitliği, macro/grup/eşlenmiş bootstrap.
- Ruff temiz. Python 3.11, bütün gerekli importlar ve CUDA kontrolü geçti.
- Test kayıtları: `windows_tests.xml`, `windows_tests.log`; statik kontrol: `ruff_windows.log`; ortam: `environment_windows.log`.
- Windows native DLL yükleme sırası için PyTorch önce başlatılır. Yeni deney CLI'ı, test conftest'i ve ortam kontrolü bunu uygular. Eski A00–A02 devamı için `scripts/resume_academic.ps1` kullanılır.
- Yerel sanal ortam mevcut CUDA PyTorch'u paylaşır. Tamamen temiz dış ortam testi henüz yapılmadı; dış paketin kurulum komutları ve bilimsel paket sürümleri kilitlidir.

## Hâlâ gerekli işler

1. Paketi dış GPU'da çalıştırıp tam KGE modellerini getirmek; sonra kilitli final değerlendirmeyi çalıştırmak.
2. Kimyasal eşleme kapsamını artırmak. Mevcut 36 doğrulanmış eşleme, 7946 ilaçlık tam çizgenin temsilî bir kohortu değildir. Akademik ağırlıklar bu nedenle henüz Streamlit'e bağlanmadı; uygulama açıkça etiketlenmiş smoke sonuçlarını gösteriyor.
3. Aynı aday havuzunda entegrasyon ablasyonu ve eksik kimya/seçim yanlılığı analizi. Klinik toksisite ground truth yokken güvenlilik kazanımı iddiası yapılmamalı.
4. Temsil/loss duyarlılıkları, eşleşen endpoint'lerle dış doğrulama, resmî veri kaynağı/lisans teyidi ve biyomedikal uzman incelemesi.
5. Nihai tablo/şekil/metni gerçek sonuçlarla tamamlamak. Bu aşamalar A07–A10 olarak açık bırakıldı; yapılmamış deney tamamlanmış işaretlenmedi.

Ayrıntılı yöntem ve birincil kaynaklar: [METHODS.md](../../docs/academic/METHODS.md). Resmî [PrimeKG deposu](https://github.com/mims-harvard/PrimeKG) OptimusKG öneriyor; çalışma eski sürümde sabit baseline olarak tanımlanıyor. [RDKit](https://rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html), [PyKEEN](https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html), [scikit-learn](https://scikit-learn.org/stable/modules/calibration.html) belgeleri ve [MoleculeNet](https://pubmed.ncbi.nlm.nih.gov/29629118/) yöntem kararlarına dayanak olarak incelendi.
