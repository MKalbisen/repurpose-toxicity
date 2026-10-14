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