# PDF incelemesi ve mühendislik kararları

## Belgenin kapsamı

Altı sayfalık belge, klinik öneri sistemi değil, ilaç yeniden konumlandırma adaylarını önceliklendiren bir araştırma sistemi tarif eder. Birinci sayfada 12 ana aşama, ikinci–beşinci sayfalarda 21 ayrıntılı adım, son sayfada klasör/teknoloji tablosu ve hedef metrikler vardır. Metin çıkarımı saklandı; birinci ve son sayfalardaki diyagram/tablo içerikleri görsel olarak kontrol edildi.

- Sayfa 1: PrimeKG → KGE → moleküler temsil → toksisite → Streamlit akışı, 12 aşamalı kabul matrisi. KGE skoru etkinlik kanıtı değildir.
- Sayfa 2: Python 3.11 ortam, PrimeKG şeması/lisans, kimyasal kimlik, coverage ve inverse leakage kontrolü, TransE.
- Sayfa 3: Aynı split üzerinde RotatE, disease–drug aday üretimi, 12 görevli Tox21, ECFP4 ve scaffold split.
- Sayfa 4: Assay başına XGBoost, maskeli çok görevli MLP, görev bazlı metrikler/CI, birleşik çıkarım ve üç sıralama yaklaşımı.
- Sayfa 5: Çevrimdışı eğitimli modellerle Streamlit, sınırlı PyVis alt çizge, aynı SMILES'tan 2D/3D, açıklamalar, testler ve gerçek deney raporu.
- Sayfa 6: Modüler dosya ağacı; Hits@10≥.70, mean ROC-AUC>.80, scaffold overlap=0 ve yaklaşık 3–5 sn çıkarım hedefleri. PrimeKG/OptimusKG güncelliğini kontrol etme notu.

## Bağımlılık sırası

Kimyasal eşleme ve split protokolleri eğitimden önce sabitlenmelidir. KG ve toksisite modelleri ayrı eğitilir; çıkarımda birleştirilir. UI yalnızca üretilmiş modelleri yükler. Model seçiminde test seti kullanılmaz; karşılaştırma raporu yalnızca çalıştırılmış deneylerden üretilir. Bu sıra PLAN.md'nin sekiz yürütme grubuna dönüştürüldü.

## Belgenin açık bıraktığı kararlar

1. **Inverse ilişki tanımı:** İsim eşleşmesiyle reverse relation tahmini yerine aynı sırasız endpoint çiftinin tüm ilişkileri gruplandı. Bu daha muhafazakâr bir sızıntı önlemidir.
2. **ID çakışmaları:** Entity anahtarı `type:source_id`. Gene ve disease sayısal kimlikleri karışmaz.
3. **Aday relation:** `indication` terapötik adaylar için kullanılır; contraindication/off-label genel çizge bağlamıdır. İlişkileri birbirine karıştıran skor üretilmez.
4. **SMILES kaynağı:** PubChem DrugBank ID sorgusu; CID synonym listesinde kaynak kimliği doğrulanır. CAS/ChEMBL synonym kimlikleri kaydedilir. İsim eşleşmesi tek başına kabul edilmez.
5. **Çoklu kimyasal kimlik:** PubChem çoklu sonuçları eşleşmemiş sayılır. Manuel mapping'de aynı drug_id için birden fazla satır hata ve ayrı CSV üretir.
6. **Tuz/stereokimya/tautomer:** İnference ve fingerprint aynı canonical isomeric SMILES'ı kullanır; tuz parçaları ve stereokimya korunur. Yalnız scaffold grup kimliği salt parent ve canonical tautomer üzerinden üretilir. Standartlaştırılamayan molekül gerekçesiyle dışlanır.
7. **Acyclic scaffold:** Boş scaffold için molecule-specific fallback kullanılmaz; bütün acyclic moleküller aynı grupta kalır.
8. **Eksik/yinelenen etiket:** NaN maskelenir. Canonical SMILES duplicate etiketleri birleştirilir; çelişen assay etiketleri NaN yapılır ve kaydedilir.
9. **PR-AUC:** Average precision olarak tanımlandı. Tek sınıflı değerlendirme metrikleri boş/status ile raporlandı. Bootstrap molekül bazlı percentile CI'dır; scaffold-kümeli belirsizlik tahmini değildir.
10. **Model seçimi:** KG validation MRR; XGBoost assay bazında validation AP; MLP validation macro ROC-AUC. Aynı test kümeleri yalnız seçilmiş modelleri değerlendirir.
11. **Risk özeti:** Max, mean ve eşik üzerindeki assay sayısı. 12 assay tamamlanmayan aday güvenli kabul edilmez ve birleşik skor bilinmeyendir.
12. **Ağırlıklı skor:** Min-max KG skorunun normalize edilmesi aday listesine bağlıdır. Varsayılan .7 araştırma tercihidir; sensitivity.csv çoklu ağırlıkta ilk 10 adayı gösterir. Biyolojik optimum/öneri iddiası yoktur.
13. **XAI:** XGBoost yerleşik TreeSHAP log-odds katkıları, bit atom/radius ortamları ve hash collision notu. Yerel KG patikaları gözlenen bağlantıları gösterir. MLP için kimyasal SHAP uygulanmadı.
14. **Kaynak güncelliği:** [Resmî PrimeKG deposu](https://github.com/mims-harvard/PrimeKG) PrimeKG'nin OptimusKG ile güncellendiğini bildiriyor. PDF'nin PrimeKG MVP şemasını ve tekrar üretilebilir baseline'ını uygulamak için sabit PrimeKG CSV kullanıldı. OptimusKG Parquet/Biolink şeması bu projede desteklenmiyor; sessizce PrimeKG diye okunmuyor.
15. **Dataverse erişim sorunu:** Bu ortamda 403 döndü. [Topluluk aynasının](https://huggingface.co/datasets/hieupth/primekg) sabit commit'i ve LFS SHA256 değeri kullanıldı. Ayna hash'i doğrulandı; güncel resmî sürümle eşdeğerlik doğrulanamadı. Manifest bunu açıkça belirtir.

## Deney sınırları

`smoke` kısa eğitimle gerçek veride tüm hattı doğrular. `full` aynı mühendislik protokolünü daha büyük çizge ve validation aramasıyla çalıştırır. Full çalıştırılmadan hedef eşikler elde edildiği veya tam akademik deneylerin bittiği söylenemez. Çoklu seed, ablasyon, dış veri seti, olasılık kalibrasyonu ve prospektif doğrulama bu teslimde gerçekleştirilmedi.

Kod/model/test durumunun ölçülmüş özeti: ../reports/implementation_status.md.
