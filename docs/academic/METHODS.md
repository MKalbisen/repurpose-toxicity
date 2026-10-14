# Akademik yöntem ve iddia sınırları — academic-v2

11 Ekim 2026. Bu belge yöntem protokolüdür; tamamlanmamış deneylerin sonucu değildir.

## Soru ve kapsam

İki ayrı görev değerlendirilir: bilinen ilaç–hastalık endikasyonlarının geri kazanılması ve Tox21 assay aktivitesinin yeni kimyasal gruplarda tahmini. Birleşik sıralama araştırma hipotezi üretir. Assay aktivitesinin klinik toksisiteyle, bilgi çizgesi skorunun tedavi etkinliğiyle eşit olduğu varsayılmaz. Hastalara tedavi veya doz önerisi üretilmez.

Çalışma retrospektiftir. Önceki smoke test sonuçları incelendiğinden yeni split gerçek bağımsız dış doğrulama sayılmaz. Zamansal kaynak etiketleri yoktur; prospektif veya temporal değerlendirme iddiası kurulamaz. Başarı için keyfî AUC>.80/Hits@10>.70 koşulu kullanılmaz; baselines karşısındaki etki, belirsizlik ve olumsuz sonuçlar raporlanır.

## Veri

PrimeKG ham kaynağının SHA256 değeri `0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69`; Tox21 değeri `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360`. Kaynak URL/commit bilgisi `reports/data_manifest.json` içindedir. PrimeKG kaynağı sabit topluluk aynasıdır; güncel resmî sürümle eşdeğerliği henüz doğrulanmamıştır. Yazılımın MIT lisansı kaynak verilerin yeniden dağıtım izni anlamına gelmez. Hesaplama paketi kişisel deney aktarımı içindir; veri lisansları açıklığa kavuşmadan kamuya açılmamalıdır.

Resmî PrimeKG deposu artık OptimusKG önerir. Bu çalışma eski PrimeKG sürümü üzerinde sabit bir baseline çalışması olarak tanımlanır. OptimusKG'ye geçiş ayrı veri/protokol sürümü gerektirir; güncellik iddiası yapılmaz. [Resmî kaynak](https://github.com/mims-harvard/PrimeKG).

Tox21: canonical isomeric SMILES model girdisidir; tuz, yük ve stereokimya korunur. Aynı canonical girdinin yinelenen etiketleri birleştirilir, çelişen etiketler eksik yapılır. Eksik assay etiketleri negatif sayılmaz. 1024 bit, radius=2, chirality kapalı Morgan fingerprint kullanılır.

Kimyasal gruplar achiral fragment parent, tamamlanmış canonical tautomer, Bemis–Murcko scaffold ve aynı fingerprint bağlantılarının geçişli kapanışıdır. Bu, birebir aynı model girdilerinin de farklı bölümlere düşmesini engeller. Fingerprint eşitliği tek başına aynı molekülün kanıtı değildir; gruplama muhafazakâr bir tercihtir. Bütün acyclic yapılar bir gruptadır. Tautomer sınırları 1000 tautomer/1000 transform olarak sabittir; Completed olmayan örnekler bölmeden önce dışlanır ve gerekçeleri kaydedilir. [RDKit API](https://rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html).

Veri hazırlığında 7831 satırdan 7729 molekül kalmıştır: 93 tamamlanmamış tautomer, 8 geçersiz SMILES ve 1 standardizasyon hatası dışlanmıştır. Split, etiketlere bakmadan büyük grupları önce yerleştiren sabit seed=2026 yöntemiyle train/validation/calibration/test=5410/773/773/773 oluşturur. En büyük grup 1778 moleküldür. Bu gruplama dağılım kaymasına yol açar; tüm kimyasal uzayda tarafsızlık iddiası kurulmaz. Nearest-train Tanimoto medyanı test için .396, maksimumu .964'tür: sıfır kimlik/scaffold overlap, benzer kimyanın tamamen yokluğu değildir. Benzerlik denetimi yalnız yapıları kullanır.

PrimeKG: drug/disease/gene-protein/pathway türlerindeki tüm uygun satırlar alınır; dereceye göre örnekleme veya dosya sırasına göre kesme yoktur. Endikasyonlar drug→disease yönüne çevrilerek tekilleştirilir. Aynı sırasız düğüm çiftinin tüm ilişkileri aynı bölümde tutulur. Seed ile SHA256 pair dağıtımı, her entity/relation için leksikografik coverage anchor çiftleri train'e alınarak tamamlanır. Bu anchor tercihi düşük dereceli düğümleri train'e çekebilir; rastgele tamamen bağımsız edge split olarak sunulmaz. 47130 entity ve 3757160 triple, 3023797/366872/366491 olarak bölünmüştür. Train entity/relation coverage gerçek kümelerden doğrulanır.

## Eğitim, seçim ve kalibrasyon

`configs/academic_run.yaml` model ayarlarını ve 17/29/43/71/101 seed'lerini belirtir. Veri, kaynak kodu, paket sürümleri ve ayarlar eğitim öncesinde SHA256 ile kilitlenir. Ayrı final komutu seçilmiş ağırlıkları hash'leyip test erişimini kaydeder. Kod, veri veya seçilmiş ağırlık değişirse kapı kapanır. Bu teknik kayıt, araştırmacının dosyaları manuel açmasını engelleyen güvenlik sınırı değildir.

Toksisite: train prevalence, Logistic Regression (C=.1/1), assay başına XGBoost (100/300 ağaç, depth=4) ve çok görevli MLP karşılaştırılır. Logistic/XGBoost ayar seçimi assay validation ROC-AUC; MLP epoch seçimi validation macro ROC-AUC üzerinden yapılır. Bu v2 tercihi, eski XGBoost-AP seçiminden farklıdır. Tüm aileler aynı split'i kullanır; arama alanı ve ölçülen süreler raporlanır. Ailelerin birebir eşit hesap süresi olduğu iddia edilmez. Validation bir sınıflıysa ilk sabit ayar seçilir; tanımsız puan testle doldurulmaz.

MLP 1024→512→256→12, BatchNorm/ReLU/Dropout(.3), Adam(.001), batch=128, azami 100 epoch, patience=15. Maskeli BCE gözlenen etiket başına ağırlıklıdır. Assay class weight yalnız train'den hesaplanır. Tek sınıflı train görevleri MLP için geçersizdir ve tahminleri NaN kalır. XGBoost class weight yalnız train'den; Logistic Regression ağırlıksızdır.

Her model ve assay için ayrılmış calibration gruplarında logit girdili, C=1 sigmoid kalibrasyon uygulanır. Her sınıfta en az beş gözlem yoksa kalibrasyon uygulanmaz, durum kaydedilir. Train-prevalence baseline yeniden kalibre edilmez. 0.5 karar eşiği sabittir; klinik eşik veya optimize edilmiş minimum recall iddiası yoktur. Kalibrasyonun başarılı olup olmadığı raw/calibrated Brier ve reliability binleriyle değerlendirilir, varsayılmaz. [Ayrı veride kalibrasyon ilkesi](https://scikit-learn.org/stable/modules/calibration.html).

KG: PyKEEN TransE ve RotatE embedding uygulamaları, her biri 100/200 boyut × .001/.01 öğrenme oranı ile eğitilir. İki ailede ortak, açıkça tanımlı loss: mean softplus(-pozitif skor) + mean softplus(negatif skor). Batch=256, negatif=16; head/tail rastgele bozulur, bilinen train pozitifleri tam üyelik denetimiyle negatiflerden çıkarılır. Diğer bölümlerin pozitifleri eğitim negatif filtresine verilmez. Bu loss, özgün makalelerin varsayılan eğitim reçetesi olarak sunulmaz. 200 epoch üst sınırı, her beş epoch'ta validation, üç değerlendirmelik patience uygulanır.

Model ve ayar seçimi yalnız `indication`, disease sorgusu ve bütün drug adayları üzerinde filtered MRR'dir. Validation filtresi train+validation; test filtresi ancak final kapısında train+validation+test'tir. Baselines train-popularity, ortak train gen komşuları ve analitik random beklentidir. MRR/Hits bağlarda orta sıra kullanır; bu, bağlar içinde beklenen reciprocal rank değildir. Random MRR tam harmonik beklentidir; bu farklı tie konvansiyonu özellikle düz skor baseline'larında belirtilmelidir. Unfiltered Recall@K bağları ilaç ID sırasıyla çözmez; rastgele bağ çözümünün beklenen recall değerini raporlar. [PyKEEN değerlendirme ilkeleri](https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html).

## Belirsizlik ve tekrar üretim

Toksisite görev başına ROC-AUC, average precision, Precision/Recall/F1/Balanced Accuracy, Brier, log loss, örnek/pozitif sayıları raporlanır. AP trapezoidal PR integralinin eşanlamlısı olarak kullanılmaz. Macro-12 AUC ancak 12 görev değerlendirilebiliyorsa verilir; mevcut görev ortalamasının denominatörü ayrıca kaydedilir.

Beş eğitim seed'inin ortalama ve standart sapması ayrı verilir; aynı test verisindeki seed'ler bağımsız örnekler sayılmaz. Önceden tanımlı beş-seed tahmin ortalaması için kimyasal component gruplarıyla 2000 bootstrap çekimi, assay ve sabit görev setli macro AUC güven aralığı hesaplanır. XGBoost/MLP ile Logistic Regression farkları aynı grup çekimleriyle eşlenmiştir. Tek sınıfa düşen çekimler atlanıp sayılır; değişen görev denominatörüyle yapay macro CI üretilmez. Çoklu assay karşılaştırmaları keşifseldir; düzeltmesiz p-değeri veya kesin üstünlük ilanı yoktur.

KG güven aralıkları hastalık kümeleriyle oluşturulur; hem pair-micro hem disease-macro MRR raporlanır. RotatE−TransE farkları seed içinde aynı test hedefleri ve hastalık çekimleriyle eşlenir. Bunlar sabit eğitilmiş modellerin veri belirsizliğidir.

MLP ve KG epoch sonunda model/optimizer/CPU-CUDA-NumPy-Python RNG kaydı tutar. MLP shuffle generator da saklanır. Epoch ortasında kesinti son tamamlanan epoch'tan devam eder. CPU'da kesintisiz/kesintili MLP'nin birebir eşitliği test edilmiştir. Farklı GPU/driver üzerinde bit düzeyinde eşitlik garanti edilmez; donanım ve ortam kaydı tutulur.

## Yayından önce kalan işler

1. Dış makinede tam KGE aramasını bitirip kilitli değerlendirmeyi çalıştırmak.
2. PrimeKG'deki 7946 ilacın kimyasal eşleme kapsamını geliştirmek. Mevcut doğrulanmış 36 eşleme smoke kapsamındadır; tam çizge için temsilî değildir. Tox21 eğitimi bu eşlemeye bağlı değildir.
3. Aynı aday havuzunda KG-only/risk-only/hard-filter/weighted/Pareto entegrasyon ablasyonu; missingness ve eşleme seçim yanlılığını ayrıca raporlamak. Klinik risk ground truth olmadığı için net klinik fayda ölçüsü üretmemek.
4. 1024/2048 bit, chirality ve görev-normalize MLP loss duyarlılığını ayrı, development-only çalışma olarak planlamak; mevcut final test üzerinden ayar aramamak.
5. Eşleşen assay endpoint'leri ve temiz kaynak/tarih/kopya denetimi olan dış doğrulama bulmak. ToxCast gibi farklı endpoint'leri doğrudan Tox21 dış testi ilan etmemek.
6. Kaynak sürümü/lisans koşullarını yayın için doğrulamak; veri lisansı sorunlarını model başarısıyla örtmemek. Akademik hedef dergi/tez biçimine göre nihai metni ve biyolojik uzman değerlendirmesini tamamlamak.

[MoleculeNet özgün benchmark](https://pubmed.ncbi.nlm.nih.gov/29629118/) veri/split/ölçüt bağlamı sağlar. Bu projedeki component split, kalibrasyon bölümü, seed ve arama bütçeleri çalışmaya özgü tercihlerdir; standart Tox21 leaderboard sonuçlarıyla birebir karşılaştırma yapılmaz.
