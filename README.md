# NHANES 2011-2018 Diyabet ve Prediyabet Tarama Veri Seti

Proje Python 3.12 ile test edildi. Komutları proje ana klasöründen çalıştırabilirsiniz.

## 1. Kurulum

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Ham Veriyi İndirme

```bash
python data_prep/prepare_sources.py
```

CDC'den dört dönemin NHANES `.xpt` ve kod kitabı (`.htm`) dosyaları ile USDA şekerli içecek eşleme tabloları `raw/` klasörüne indirilir. `raw/` klasörü zaten hazırsa doğrudan 3. adıma geçilebilir.

## 3. Veri Setini Oluşturma

```bash
python data_prep/build_dataset.py
```

Kayıtlar her dönem içinde `SEQN` üzerinden birleştirilir ve çalışma grubu filtrelenir (20+ yaş, muayeneye katılmış, HbA1c ölçümü var, daha önce diyabet/prediyabet tanısı veya ilacı yok, hamile değil `RIDEXPRG != 1`). Hedef sütun (`target`), `HbA1c >= 5.7` ise `1`, değilse `0` olarak yazılır. Seçilen 63 özellik `features.txt` listesindedir. Ana tablo `results/nhanes_2011_2018_dataset.csv`, eleme adımları ise `results/intermediate/cohort_flow.csv` dosyasına kaydedilir.

## 4. Train / Test Ayrımı

```bash
python data_prep/split_dataset.py
```

Veri `%80` eğitim (`results/train.csv`) ve `%20` test (`results/test.csv`) olarak ayrılır. Dönem (`cycle`) ve `target` oranları korunur; `seed=42` sabit olduğu için her çalıştırmada aynı ayrım üretilir.

| Dosya | Satır | Sütun |
|---|---:|---:|
| Ana veri seti | 15.473 | 66 |
| Train | 12.378 | 66 |
| Test | 3.095 | 66 |

Tablolarda `cycle`, `seqn`, `target` ve 63 özellik sütunu bulunur. `results/test.csv` model seçimi ve hiperparametre denemelerinde kullanılmaz; sadece son test aşamasında açılır.

## 5. Testler

```bash
pytest -q
```

## Veri Seti Hakkında Notlar

- NHANES örnekleme ağırlıkları (`WTMEC2YR` vb.) kullanılmadığı için veri setindeki `~%29.6` pozitif oranı doğrudan ABD genel nüfus oranı olarak yorumlanmamalıdır.
- `RIDAGEYR` (`age`) değişkeninde 80 yaş ve üzeri kişiler NHANES tarafından `80` olarak üst sınırlı kodlanmıştır.
- `weight_change_1y_kg` ve `reported_weight_change_10y_kg` sütunlarında pozitif değer kilo artışını, negatif değer kilo kaybını gösterir (`güncel kilo - önceki kilo`).
- `diastolic_mean` hesaplanmadan önce `30 mmHg` altındaki hatalı tekil ölçümler (`0` dahil) `NaN` yapılmıştır.
- 2011-2016 dönemlerinde hayatı boyunca alkol almadığı için sorusu geçilen kişilere, 2017-2018 dönemiyle uyumlu olması için `alcohol_frequency = 0` atanmış ve `drinks_per_drinking_day` (`ALQ130`) üst sınırı tüm dönemlerde `15` yapılmıştır.
- Son 1 yılda isteyerek 10 lb üzeri kilo verdiği için (`WHQ060 == 1`) `WHQ070` sorusu atlanan kişilerde `attempted_weight_loss_1y = 1` olarak doldurulmuştur.
