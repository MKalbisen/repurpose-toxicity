# A03 — bir sonraki oturumun iş emri

> Tarihsel iş emri. 11 Ekim 2026'da akademik-v2 veri hazırlığı uygulandı; güncel kanıtlar `data/processed/academic-v2/*/manifest.json`, yöntem `METHODS.md` içindedir. Aşağıdaki GPU yok/henüz uygulanmadı ifadeleri eski başlangıç durumunu anlatır.

Önce `ACADEMIC_PROGRESS.md`, `reports/academic/state.json`, `reports/academic/audit.json` ve `configs/academic_protocol.yaml` okunur. `source .venv/bin/activate && python -m repurpose.academic.cli status` kanıtları kontrol eder. A00/A01/A02 geçerliyse tekrar kodlanmaz. Bu iş emri A03'ün tamamlanmış olduğu anlamına gelmez.

## Ölçülmüş başlangıç

- Mevcut Tox21: 7.822 molekül, train/val/test 6258/782/782.
- Canonical SMILES ve scaffold overlap=0. **11 fingerprint eşdeğerlik grubu birden fazla splitte**. Aynı ECFP girdisi farklı gerçek moleküllerden de gelebilir; bu tek başına aynı molekül veya etiket sızıntısı kanıtı değildir. Ana deneyde model-girdi eşitliğine göre de birlikte gruplamak daha muhafazakâr protokoldür.
- Train'de 1780 acyclic molekül aynı scaffold grubunda; val/test'te 0. Bu dağılım farkı korunup raporlanmalıdır; iyi test sonucu için grubu parçalamayın.
- Canonical tautomer için enumeration completion kaydı yok; eski hazırlıkta üst sınıra ulaşma uyarıları vardı. 8 SMILES ve 1 scaffold standardizasyon hatası ayrı dosyalarda.
- KG smoke: 231 entity/1800 triple/9 relation. Indication train 567, val 75, test 64 benzersiz drug-disease çifti. Entity/relation coverage ve çift overlap kontrolleri gerçek setlerden geçti.
- Legacy RotatE'nin **validation-only drug MRR** değeri .173939; train-popularity .221466; random beklenen .101415. Bunlar nihai akademik test sonuçları değildir.
- GPU yok; CPU ve yaklaşık 31 GiB host RAM ölçüldü. T4/L4 kullanıcı tarafından gerekirse sağlanabilir. GPU erişimi veya ücretli compute kullanımı varsayılmayacak.

## A03.1 — kimyasal kimlik ve önbellek

1. Yeni sürümlü `src/repurpose/academic/prepare.py` veya alt modüller oluşturun. `smoke` verileri/modelleri üzerine yazmayın. Kaynak raw Tox21 hash'i ile kimyasal policy/RDKit sürümü ve policy hash'i cache anahtarı olsun.
2. Raw Tox21'i tekrar okuyun; valid canonical isomeric SMILES üretin, duplicate/çelişen etiket politikasını açık uygulayın. Kaynak satır kimliklerini kaybetmeyin.
3. Salt parent ve canonical tautomer **grup kimliği** üretin. `TautomerEnumerator.Enumerate(mol)` sonucunun `.status` alanını kaydedin. Completed olmayanlar için önceden belirlenmiş karantina veya muhafazakâr ortak grup politikası uygulayın; tamamlanmış gibi işaretlemeyin. RDKit API: https://rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html
4. Fingerprint ve molekül render girdisi için aynı standardizasyon policy'sini koruyun; grup standardizasyonunu sessizce tüm model input'una uygulamayın. Geçerli vektörle birebir ilişki kurun.
5. Çıktılar: chemical_identity.csv (source row, canonical, salt/stereo, parent/tautomer ID, enumeration status), excluded_molecules.csv, policy.json, cache manifest.
6. Kabul: tautomer/salt/stereo eşdeğerlik örnekleri ve failure/limit testleri; kaynak/etiket sayıları mutabık; tekrar çalıştırmada geçerli cache kullanılır. Eksik cache/yarım dosya kabul edilmez.

## A03.2 — birlikte tutulacak kimyasal gruplar ve sabit split

1. Union-find veya eşdeğer bağlı bileşen algoritmasıyla şu kimliklerden herhangi birini paylaşan molekülleri birleştirin: scaffold, standardized parent/tautomer, identical binary fingerprint. Birleştirme geçişlidir; A=B ve B=C ise A/C farklı split'e düşemez. Bütün acyclic moleküller aynı gruptadır.
2. Label/model/test performansına bakmadan `split seed=2026` ile yaklaşık 80/10/10 bileşen bölmesi üretin. Seed'i iyi metrik elde etmek için değiştirmeyin. Seed ile shuffle yalnız eşit grup boyutlarını etkiliyorsa algoritmanın sınırlamasını açıkça belgeleyin.
3. Chemistry/fingerprint/scaffold/components set kesişimleri sıfır; bütün kabul edilen satırlar tam bir kez yer alır. Boyutlar, label missingness, scaffold dağılımı ve değerlendirilmesi mümkün görev sayıları raporlanır. Nihai test model metriklerini üretmeyin.
4. Yeni `.npz` train/validation/test ve grup kimliklerini açık versiyonlu dizine kaydedin. Eski smoke splitinin testine erişildiğini manifestte koruyun; yeni split tamamen görülmemiş dış veri değildir.
5. Kabul: geçişli gruplama, acyclic, hash çakışmasıyla eşit input, deterministic split, sıfır overlap ve model seçimine test geçmemesi testleri.

## A03.3 — KG ana deney verisi

1. Resmî kaynak doğrulama durumunu sürüm/lisans manifestine kaydedin. Dataverse yine erişilemiyorsa mevcut aynayı kesin sürüm/provenans sınırlılığıyla tanımlayın; eşdeğerliği kanıtsız varsaymayın.
2. Kaynak kapsamı ve dahil etme kuralını label/model skoruna göre optimize etmeden sabitleyin. En yüksek dereceye sahip 50 ilaç ana deneyin genel ilaç evreni değildir. Dosya sırasıyla gen seçmeyin. Full veya önceden tanımlı cohort kararı compute pilotuyla ayrıca açık yazılır.
3. Type-specific kimlikler ve tek kanonik indication yönü; KG'de orientation kopyalarının yapay örnek ağırlığını ölçün. Same-pair tüm relation/directions split grupları birlikte kalsın; semantik opposite ilişkileri birbirine dönüştürmeyin.
4. Çizge büyüklüğünde mevcut Python/pandas per-group split döngüsünü benchmark edip gerekirse group arrays / streaming veri işlemesiyle değiştirin. 31 GiB RAM ile bilinmeyen büyüklükte eğitim başlatmayın; peak RSS ve süre ölçün.
5. Primary değerlendirme drug-only head ranking için train/validation/test hedef listelerini ayrı, canonical ve duplicate-free üretin. Coverage/degrees/relations ve fixed split hash'lerini raporlayın.
6. Kabul: reverse leakage, coverage, target duplicate, cohort tarafsızlık tanımı ve veri hash testleri.

## Kapanış ve sonraki aşama

A03 ancak kimyasal ve KG veri kabul koşulları birlikte sağlanırsa `validated` olur. Bir alt iş tamamlanırsa state stage'e completed gibi yazmak yerine journal note/artefakt kaydı ekleyin; yapılmayan alt işleri açık bırakın. A04 sonra CPU pilotunu, development-only eğitim/kalibrasyon akışını, run ID izolasyonunu ve optimizer/RNG/checkpoint resume desteğini uygular. Final test kapısı ve protocol freeze A05'tedir; şimdilik taslak.
