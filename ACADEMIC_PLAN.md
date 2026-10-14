# Akademik geliştirme ve deney planı

> 11 Ekim 2026 güncellemesi: aşağıdaki metin ilk planın tarihsel kaydıdır. Uygulanan v2 yöntem `docs/academic/METHODS.md`, gerçek durum `reports/academic/REVIEW.md`, yürütülebilir ayarlar `configs/academic_run.yaml` içindedir. Donanım RTX 3050 Ti 4 GB olarak doğrulandı. Yeni protokol 70/10/10/10 kimyasal bölümleme ve bütün toksisite aileleri için validation ROC-AUC seçimi kullanır. Eski CPU bilgileri ve A00–A02 ile sınırlı oturum ifadeleri güncel durum değildir.

Bu plan, çalışan MVP'yi değerlendirilebilir ve tekrar üretilebilir bir akademik çalışmaya dönüştürür. Önceki 21 adımın mühendislik durumu `reports/implementation_status.md` içinde korunur. Bu planın yürütme kaydı **ACADEMIC_PROGRESS.md**, makine tarafından okunan durumu **reports/academic/state.json**, değişiklik günlüğü **reports/academic/events.jsonl** içindedir.

## Araştırma sorusu ve kapsam

**Soru:** Bilgi çizgesiyle önceliklendirilmiş ilaç adaylarına moleküler assay riski taraması eklemek, aday geri kazanımını ve risk/etkinlik skoru dengesini nasıl değiştiriyor?

İki alt problem ayrı sınanacak: (1) hastalık sorgusunda bilinen endikasyonların geri kazanılması; (2) yeni scaffold'larda 12 Tox21 assay aktivitesinin tahmini. Entegrasyon aday önceliklendirmesidir. İki modelin başarısı birlikte klinik etkinlik veya güvenlilik kanıtı oluşturmaz. Pozitif sonuç çıkmaması da geçerli araştırma sonucudur. Hits@10≥.70 ve ortalama ROC-AUC>.80 hedeflerdir; akademik yeterliliğin tanımı değildir.

Geçici teslim kapsamı: tez/makale düzeninde, retrospektif hesaplamalı çalışma. Kullanıcı henüz belirli bir tez/makale türü bildirmedi. Mevcut donanım CPU; gerektiğinde T4/L4 bulunabilir. Hazırlık/denetim/baseline CPU'da, büyük KGE araması donanım pilotundan sonra yapılır. GPU'ya erişildiği varsayılmayacak; donanım, süre, VRAM ve maliyet yürütme sırasında kaydedilecek.

## Adımlar ve kabul ölçütleri

| ID | Yapılacak iş | Kabul ölçütü | Bağımlılık |
|---|---|---|---|
| A00 | Deney kimliği, kaynak/ayar hash'leri, atomik durum, olay günlüğü ve devam komutları | Kesilen iş tamamlandı sayılmaz; artefakt değişirse eski doğrulama kullanılmaz; testler geçer | — |
| A01 | MVP'nin bilimsel denetimi; veri, split, kimya, kaynak ve donanım envanteri | Hesaplanmış audit.json/md; kritik açıklar, mevcut test erişimi ve veri kapsamı açık | A00 |
| A02 | Hedefe uygun değerlendirme altyapısı | Drug-only filtered head ranking, baseline ve leakage testleri; scaffold cluster/paired bootstrap; metrik geçerlilik testleri | A01 |
| A03 | Akademik veri protokolü ve temiz veri üretimi | Sürüm/lisans, kimyasal kimlik, tautomer/scaffold/fingerprint leakage denetimi; KG hedef yönü tekilleştirilmiş; sabit split hash'leri | A01, A02 |
| A04 | CPU pilotu, train/validation ayrımı, arama uzayı ve kaynak bütçesi | GPU/RAM varsayımı yapmayan ölçüm; her model için eşit karşılaştırma koşulları; test skoru üretilmez | A03 |
| A05 | Nihai deney protokolünü dondurma ve test kapısını etkinleştirme | Veri/split/kod/ayar hash'leri kilitli; seçim ve kalibrasyon yalnız development verisinde; final değerlendirme ayrı komut | A04 |
| A06 | Toksisite modelleri ve baselines | Train-prevalence, Logistic Regression, XGBoost, MLP; 5 seed; aynı split, validation bütçesi; kalibrasyon/threshold development içinde | A05 |
| A07 | KGE modelleri ve baselines | Random beklenti, train popularity ve ortak gen komşuluğu; TransE/RotatE; 5 seed; indication-specific drug-only validation | A05, donanım pilotu |
| A08 | Kilitli test, belirsizlik ve hata analizi | Görev bazlı ve macro metrikler, %95 CI, eşlenmiş model farkları; seed dağılımı; testten sonra ayar değişmez | A06, A07 |
| A09 | Entegrasyon, ablation ve bağımsız doğrulama | KG-only / risk-only / hard filter / weighted / Pareto; mapping kapsamı; dış veri varsa endpoint/overlap denetimi; yoksa iddia sınırı | A08 |
| A10 | Akademik rapor ve yeniden üretme paketi | Kaynaklı yöntem, sayısal tablolar, negatif sonuçlar, limitations; temiz ortam denemesi; kod/config/data manifest ve deney kaydı birlikte | A09 |

A00–A02 bu oturumda uygulanır ve doğrulanır. A03 ile sonraki aşamalar tamamlanmadan uzun eğitim başlatılmaz. Bu sıra teknik önkoşulları ifade eder; yeniden kullanıcı onayı isteme koşulu değildir. Her oturum sonunda kayıt güncellenir ve bir sonraki somut komut belirtilir.

## Değerlendirme protokolü taslağı

### Veri ve test bağımsızlığı

- `smoke` sonuçları geliştirme sonuçlarıdır; akademik başarı tablosuna taşınmaz. Geçmişte Tox21 smoke testine erişildi. Yeni bir split bu geçmiş erişimi ortadan kaldırmaz; sonuçlar retrospektif benchmark olarak raporlanacak. Tam anlamıyla bağımsız genelleme iddiası için erişilmemiş, aynı endpoint'lere sahip dış veri gerekir.
- PrimeKG aynasının hash'i doğrulandı; en güncel resmî sürüm ve lisans eşdeğerliği doğrulanmadı. Yayın için bu veri sürümü açık kaynak/provenansla tanımlanacak veya resmî kaynak ayrıca doğrulanacak. OptimusKG geçişi şema ve deney tasarımını değiştirir; ayrı çalışma sürümüdür.
- KG: drug/disease/gene/pathway kapsamı, relation dağılımları, çift yön kopyaları ve dahil etme kriterleri sabitlenir. Eski smoke örneklemi yüksek derece seçimi ve dosya sırasına bağlıdır; ana deney için tarafsızlık/genelleme sınırlaması taşır.
- KG split'te aynı endpoint çiftinin tüm ilişkileri beraber tutulur. Hedef endikasyonlar değerlendirmede drug→disease yönünde bir kez sayılır. Transductive entity/relation coverage gerçek setlerden ölçülür. Zaman etiketi yoksa temporal veya prospective split iddiası yapılmaz.
- Moleküllerde canonical SMILES, tuz/charge/stereo policy, tautomer kimliği, scaffold ve fingerprint eşitlikleri denetlenir. Tautomer enumerator sınırları kaydedilir; başarısız veya sınırda kimliklerin farklı split'lere sızması önlenir. Boş scaffold grubu bölünmez.
- Test ayrılmadan seçilmiş scaffold/model performansına göre seed araması yapılmaz. Split seed ve eğitim seed'i ayrıdır. Sabit dış split ve development içinde group-aware seçim; bir sınıflı görevler gizlenmez, denominatörler kaydedilir.
- Taslak dış split seed: 2026; final kilit öncesinde yalnız yapı/grup ve development dağılımları denetlenir. Eğitim seed'leri: 17, 29, 43, 71, 101. Split hassasiyeti ayrı ablation'dır; iyi test split'i seçilmez.

### Modeller ve seçim

- Primary KG endpoint: hastalık için **ilaç adayları arasındaki filtered MRR**, Hits@1/3/10 ve Recall@K. Genel KG both/head/tail metrikleri ikincil ve farklı isimle raporlanır. Bilinen pozitifler filtrelenir, hedef korunur. Validation filtreleri train+validation; test filtreleri ancak final kapısında train+validation+test.
- Primary toxicity endpoint: 12 assay macro ROC-AUC; class imbalance için macro ve assay AP (PR-AUC) birlikte. Train prevalence, AP/prevalence bağlamı ve her assay örnek/pozitif sayısı raporlanır. İki sınıfı olmayan görevde AUC boş bırakılır; sessizce aynı 12 görev karşılaştırması gibi sunulmaz.
- KGE modeli validation drug-only MRR; toksisite parametreleri aynı development protokolünde seçilir. XGBoost/MLP başlıca model aileleridir; baselines performans kazanımının neye göre olduğunu gösterir. Arama ve compute bütçesi A04'te dondurulur. Epoch üst sınırı başlangıçta 200; convergence/erken durdurma development üzerinden izlenir.
- Kalibrasyon için development içinde ayrı scaffold grupları veya group-aware out-of-fold tahminler gerekir. Kalibratör modelin kendi train örneklerine veya final testine fit edilmez. Brier, log loss ve reliability plot raporlanır; bunların tek başına genel calibration başarısını ispatlamadığı belirtilir.
- Assay eşikleri development'ta sabit bir amaçla seçilir; örneğin önceden tanımlanan minimum recall altında precision. Yeterli pozitif bulunmazsa otomatik eşik iddiası yerine sınırlılık kaydı. Klinik eşik gibi sunulmaz.
- Maskeli MLP kaybının global observation weighting ile görev-normalize weighting farkı kontrollü ablation olarak sınanır. Chirality açık/kapalı ve 1024/2048-bit deneyi yalnız development kapsamında planlanır; ayrı temsillerin başarısı ayrı raporlanır.

### Belirsizlik ve entegrasyon

- Tox21 %95 CI: scaffold grubu örnekleme; tüm grup molekülleri birlikte resample edilir. Eşlenmiş model farkında aynı grup çekimleri kullanılır. Tek sınıflı çekim oranı ve kullanılan tekrar sayısı kaydedilir; en az 2000 çekim nihai varsayımdır, CPU pilotunda süre ölçülür.
- Eğitim seed'leri aynı dış split üzerinde ortalama±SD ile raporlanır. Aynı veride seed sonuçları bağımsız veri örnekleri gibi kullanılarak yapay dar CI oluşturulmaz. Model farkları ile seed varyasyonu ayrı analiz edilir.
- KG belirsizliği hedef hastalık bazlı cluster resampling ve hastalık bazlı macro/micro tablolarıyla ayrıca değerlendirilir; yalnız sırasız endpoint bağımsızlığı varsayılmaz.
- Entegrasyon: karşılaştırmalar aynı aday havuzunda yapılır. Eksik kimya/assay sonuçları sıfır risk değildir. Ağırlıklar testte optimize edilmez; Pareto üyeliği ve ranking hassasiyeti raporlanır. Klinik risk ground truth yoksa klinik net fayda/etkinlik kazanımı iddiası kurulmaz.
- Bağımsız doğrulama veri seti bulunursa aynı 12 assay tanımı, kimyasal duplicate/tautomer/fingerprint overlap ve kaynak tarihi incelenir. Farklı endpoint'li ToxCast vb. doğrudan aynı görevin dış testi sayılmaz. Dış doğrulama yoksa A09 bunu açık sınırlılık olarak tamamlayabilir.

## Devam kuralları

1. `ACADEMIC_PROGRESS.md` ve `reports/academic/state.json` okunur.
2. Tamamlanan aşamanın evidence hash'leri kontrol edilir: `python -m repurpose.academic.cli status`.
3. Son değişiklikler `reports/academic/events.jsonl` içindedir; kullanıcı tercihleri ve bekleyen işler kaybolmaz.
4. Sonraki hazır aşama küçük birimlerle uygulanır. Deneyler run ID/seed/config/split hash ile ayrı dizinlerde tutulur. Daha önceki çıktının üzerine sessizce yazılmaz.
5. İş ancak kabul testleri ve gerçek artefaktlar doğrulanırsa `validated` yapılır. Kesintide `in_progress` kalır; sonraki oturum kontrol ederek yeniden yürütür. Uzun model eğitimi checkpoint özelliği eklenmeden epoch ortasından devam ediyormuş gibi raporlanmaz.
6. Bilimsel karar veya veri sürümü değişirse yeni protocol version açılır. Eski metrikler korunur ve yeni protokole aitmiş gibi gösterilmez.

## Kaynaklar

Bu plandaki seed sayısı, hesap bütçesi ve endpoint öncelikleri çalışma için seçilen yöntem kararlarıdır; kaynaklardan evrensel zorunluluk olarak aktarılmaz.

- [MoleculeNet — özgün benchmark makalesi](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a): moleküler benchmark, split ve model kıyaslama bağlamı.
- [scikit-learn — cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html): tuning/test ayrımı ve grup bağımlılığı.
- [scikit-learn — probability calibration](https://scikit-learn.org/stable/modules/calibration.html): ayrı veride kalibrasyon ve olasılık değerlendirmesi.
- [PyKEEN — evaluation](https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html): filtered değerlendirmede validation/test filtreleri.
- [PrimeKG resmî depo](https://github.com/mims-harvard/PrimeKG): veri kaynağı, kaynak lisansları ve OptimusKG güncellemesi.
