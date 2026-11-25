# NHANES 2011–2018 · Seçilen 63 özellik

Python 3.12 ile doğrulandı. ZIP’i açın ve Terminal’i `NHANES-63-Kod` klasöründe başlatın.

## 1. Ortamı kur

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Ham veriyi çek

```bash
python prepare_sources.py
```

CDC’den dört döngünün NHANES XPT dosyaları ve kod kitapları, USDA’dan şekerli içecek hesabı için dört dönem eşleme tablosu `raw/` klasörüne kaydedilir. Ham dosyalar hazırsa doğrudan 3. adıma geçebilirsiniz.

## 3. Ana veri setini oluştur

```bash
python build_dataset.py
```

Kod her dönem içinde SEQN ile eşleştirir, çalışma grubunu oluşturur ve HbA1c ≥ %5,7 olduğunda hedefi 1 olarak yazar. `features.txt` Markdown’da seçilen 63 sütunu ve sıralarını içerir. Sonuç `results/nhanes_2011_2018_dataset.csv` dosyasına yazılır. Eleme sayıları `results/intermediate/cohort_flow.csv` dosyasındadır.

## 4. Eğitim ve testi ayır

```bash
python split_dataset.py
```

`results/train.csv` ve `results/test.csv` oluşur. Bölünme dönem ve hedef oranlarını yaklaşık korur. Aynı ana tablo için 42 tohumu aynı katılımcı ayrımını üretir.

| Çıktı | Katılımcı | Sütun |
|---|---:|---:|
| Ana veri seti | 15.668 | 66 |
| Eğitim | 12.534 | 66 |
| Test | 3.134 | 66 |

66 sütun, 63 aday özellik ve `cycle`, `seqn`, `target` alanlarından oluşur. Model girdisini eğitim sırasında ayrıca seçin. Test dosyası son değerlendirme içindir.
