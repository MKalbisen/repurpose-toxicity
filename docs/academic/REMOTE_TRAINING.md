# Başka bilgisayarda eğitim

11 Ekim 2026. Paket: `exports/academic-v2-compute.zip`. Kişisel hesaplama aktarımı içindir; veri kaynaklarının lisansları ayrıca geçerlidir. Ham 1 GB CSV ve geçici SQLite veritabanı pakete dahil edilmez; sabitlenmiş işlenmiş veri dahildir.

## Süre ve donanım

Yerelde ölçülen RTX 3050 Ti Laptop GPU: 4 GB VRAM; PyTorch 2.11.0 CUDA 13.0. Yaklaşık 3.024 milyon train triple için epoch 129–288 saniye. Azami 200 epoch'lu tek deneme 7.3–16.1 saat. İki model × iki boyut × iki öğrenme oranı × beş seed = 40 deneme; tüm denemelerin üst epoch sınırına ulaşması yaklaşık **464 saat (19.3 gün)** tahmini verir. Erken durdurma kısaltabilir; ısınma, yük ve disk maliyeti uzatabilir. Bu ölçümden başka GPU için kesin hız veya maliyet çıkarılamaz.

Bu nedenle KGE tam eğitimi yerelde başlatılmadı. Daha hızlı CUDA GPU üzerinde çalıştırın. Pilotun en yüksek PyTorch allocation ölçümü yaklaşık 399 MiB; bu CUDA context, ekran kullanımı ve diğer süreçleri içermez. Asıl kısıt bu ayarlarda VRAM'den çok süredir. Mevcut makinenin toplam RAM'i 15.2 GiB, pilotta boş RAM 1.65 GiB idi. Başka makinede en az mevcut kapasiteyi ve birkaç GB boş RAM'i koruyun; daha fazla boş RAM rahatlatır.

## Kurulum

ZIP'i yeni bir dizine çıkarın. Python 3.11 kullanın. GPU sürücüsüne uygun PyTorch **2.11.0** kurun; CPU wheel kurmayın. Ardından:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-academic-lock.txt
python -m pip install -e '.[dev,academic]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
python -m pytest tests/test_academic_prepare.py tests/test_academic_metrics.py tests/test_academic_experiment.py -q
python -m repurpose.academic.experiment freeze
```

Windows'ta aktivasyon yerine `.venv\Scripts\python.exe` kullanabilirsiniz. CUDA wheel'in `+cu...` eki farklı olabilir; kilit PyTorch temel sürümünü denetler. Bilimsel paket sürümleri değişirse kilit bunu reddeder. Dosyaları paket içinden kullanın; satır sonlarını değiştirmeyin. `environment.txt` yerel ortam envanteridir; platforma bağımlı tam pip çıktısını Linux'a körlemesine kurmayın, `requirements-academic-lock.txt` kullanın.

`freeze` mevcut kilidi doğrular; veri veya kod değişikliğini onaylayıp eski kilidin üstüne yazmaz. Bu paket sabit split ile geldiğinden `prepare` çalıştırmak gerekmez ve ham veri pakette yoktur.

## Eğitim ve kesinti

```bash
bash scripts/train_academic_remote.sh
```

Bu komut tamamlanmış toksisite modellerini hash ile doğrulayarak atlar, gerekiyorsa eksiklerini eğitir ve TransE/RotatE aramasını çalıştırır. Kesilirse aynı komutu tekrar çalıştırın. Son tamamlanan epoch checkpoint'inden devam eder; biten denemeler yeniden eğitilmez. Aynı run üzerinde eşzamanlı iki işlem başlatmayın.

Alternatif tek komut:

```bash
python -m repurpose.academic.experiment train-kg
```

Ayarları, seed'leri, kodu veya veri dosyalarını bu run içinde değiştirmeyin. Hata olursa log ve traceback'i getirin; kilidi silerek sürdürmeyin. Daha düşük batch gibi bir gereksinim varsa yeni protokol/run gerekir.

## Geri getirilecekler

`reports/academic/runs/academic-v2/` dizininin tamamını sıkıştırıp getirin. Özellikle `protocol_lock.json`, `kg/*/*/selection.json`, `complete.json`, seçilmiş `best.pt` ve toksisite modelleri gerekir. Devam etmek gerekiyorsa `resume.pt` dosyalarını da koruyun. Çalıştırılan makinenin `python -m pip freeze` çıktısını ve `nvidia-smi` bilgisini ekleyin.

Final test otomatik açılmaz. Her iki eğitimin tamamlandığı doğrulandıktan sonra:

```bash
python -m repurpose.academic.experiment evaluate-toxicity
python -m repurpose.academic.experiment evaluate-kg
```

Bu komutlar ilk test erişimini kaydeder ve seçilmiş ağırlıkların hash'lerini sabitler. Sonrasında aynı run'da yeniden eğitim reddedilir. Bootstrap 2000 tekrarlıdır; değerlendirme de zaman alabilir. Sonuçlar `evaluation/` altına yazılır. Testten sonra iyi sonuç bulmak amacıyla ayar değiştirmeyin; değişiklik olursa ayrı keşif çalışması olarak açıklayın.
