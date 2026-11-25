# NHANES 2011–2018 · Type0 Prediyabet / Diyabet Tarama Veri Seti (63 Özellik)

Python 3.12 ile doğrulandı. Terminal'i `type0-diabetes-screening` klasöründe başlatın.

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

CDC'den dört döngünün NHANES XPT dosyaları ve kod kitapları, USDA'dan şekerli içecek hesabı için dört dönem eşleme tablosu `raw/` klasörüne kaydedilir. Ham dosyalar hazırsa doğrudan 3. adıma geçebilirsiniz.

## 3. Ana veri setini oluştur

```bash
python build_dataset.py
```

Kod her dönem içinde `SEQN` ile eşleştirir, çalışma grubunu oluşturur (20+ yaş, muayeneye katılmış, HbA1c ölçümü mevcut, bilinen diyabet/prediyabet ve diyabet ilacı/insülin yok, hamile değil `RIDEXPRG != 1`) ve hedef değişkeni **tanı almamış prediyabet/diyabet riski (`HbA1c >= %5,7` → 1, aksi hâlde 0)** olarak yazar. `features.txt` seçilen 63 sütunu ve sıralarını içerir. Sonuç `results/nhanes_2011_2018_dataset.csv` dosyasına yazılır. Eleme sayıları `results/intermediate/cohort_flow.csv` dosyasındadır.

## 4. Eğitim ve testi ayır

```bash
python split_dataset.py
```

`results/train.csv` ve `results/test.csv` oluşur. Bölünme dönem ve hedef oranlarını yaklaşık korur. Aynı ana tablo için 42 tohumu aynı katılımcı ayrımını üretir.

| Çıktı | Katılımcı | Sütun |
|---|---:|---:|
| Ana veri seti | 15.473 | 66 |
| Eğitim | 12.378 | 66 |
| Test | 3.095 | 66 |

66 sütun, 63 aday özellik ve `cycle`, `seqn`, `target` alanlarından oluşur. Model girdisini eğitim sırasında ayrıca seçin. Test dosyası yalnızca son değerlendirme içindir.

## Önemli Veri Notları

- **Örnekleme ağırlıkları:** NHANES örnekleme ağırlıkları (`WTMEC2YR` vb.) kullanılmamıştır; bu nedenle veri setindeki pozitif oranı (`~%29,6`) genel nüfus yaygınlığı (prevalans) değildir.
- **Yaş üst sınırı (`age`):** NHANES `RIDAGEYR` değişkeni 80 yaş ve üzerini `80` olarak üst sınırlı (`80+`) kodlar.
- **Kilo değişimi işareti:** `weight_change_1y_kg` ve `reported_weight_change_10y_kg` sütunlarında **pozitif değer kilo artışı**, **negatif değer kilo kaybı** anlamına gelir (`güncel kilo - önceki kilo`).
- **Diyastolik tansiyon (`diastolic_mean`):** Fizyolojik olarak imkânsız olan `30 mmHg` altındaki ortalamalar (ölçüm hatası / 0 değerleri) `NaN` yapılır.
