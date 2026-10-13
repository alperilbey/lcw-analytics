# LC Waikiki Kadın Kategorisi — Ürün & Yorum Analizi

lcw.com üzerinden Kadın kategorisindeki ürünleri ve müşteri yorumlarını
toplayıp **duygu analizi**, **beğeni oranı** ve **en çok tercih edilen ürünler**
istatistiği çıkaran uçtan uca veri analizi projesi.

## 🎯 Özellikler

- 🕷️ **Scraper:** lcw.com gömülü JSON + schema.org ld+json parse
- 🧠 **Sentiment:** Türkçe kural tabanlı + eğitilebilir ML modeli
- 📊 **Analiz:** Ürün/kategori bazlı istatistikler, popülerlik skoru
- 📈 **Görselleştirme:** matplotlib + plotly HTML dashboard
- 🤖 **Model:** TF-IDF + LogisticRegression ve (opsiyonel) BERTurk

## 🧠 Model Performansı

Proje, Türkçe e-ticaret yorumları üzerinde eğitilmiş bir **TF-IDF + LogisticRegression** sentiment modeli içerir.

### Eğitim Verisi
| Kaynak | Örnek Sayısı |
|--------|-------------|
| LC Waikiki (scraper) | 273 yorum |
| Trendyol (Kaggle) | 43.923 yorum |
| **Toplam** | **44.196 yorum** |

### Model Metrikleri (Test Seti: 8.837 örnek)

| Metrik | Değer |
|--------|-------|
| **Accuracy** | **0.809** |
| **Macro F1** | **0.661** |
| Weighted F1 | 0.829 |

**Sınıf Bazlı Performans:**

| Sınıf | Precision | Recall | F1 | Destek |
|-------|-----------|--------|-----|--------|
| positive | 0.963 | 0.850 | **0.903** | 6.967 |
| negative | 0.622 | 0.746 | **0.678** | 1.019 |
| neutral | 0.319 | 0.549 | 0.403 | 851 |

> **Not:** Veri setinde pozitif yorumlar baskın olduğu için (≈%79), `class_weight="balanced"` parametresi ile azınlık sınıfları dengelendi.

### Model Mimarisi

```
Metin → Temizleme (stopwords, regex) → TF-IDF (1-2 gram, 50k özellik)
     → LogisticRegression (class_weight=balanced, C=1.0)
     → {positive | neutral | negative}
```

### Yeniden Eğitim

```bash
# Kaggle veri setini indir
kaggle datasets download -d ahmetfurkandemr/trendyol-product-comments -p data/raw/ --unzip

# Modeli eğit
python scripts/train_sentiment.py

# Metrikleri görüntüle
python scripts/evaluate_model.py
```

### Kullanım

```python
from lcw_analytics.sentiment import ml_sentiment

label, score = ml_sentiment("Kumaşı çok kaliteli, bayıldım.")
# → ("positive", 0.94)
```

## 📊 Örnek Çıktılar

### Dashboard

![Dashboard](https://files.catbox.moe/1ho9o5.png)

### Grafikler

| Puan + Duygu Dağılımı | Kategori + Fiyat |
|-----------------------|------------------|
| ![Ratings](https://files.catbox.moe/gsg8jn.png) | ![Category](https://files.catbox.moe/04lfpk.png) |

### Kelime Bulutu

![Wordcloud](https://files.catbox.moe/g3502l.png)


## 🚀 Hızlı Başlangıç

```bash
# 1) Sanal ortam
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2) Bağımlılıklar
pip install -r requirements.txt

# 3) Projeyi editable modda kur
pip install -e .

# 4) Pipeline'ı çalıştır (scrape + analyze + visualize)
python -m lcw_analytics scrape
python -m lcw_analytics analyze
python -m lcw_analytics visualize

# Veya hepsi bir arada
make all
```

## 📊 Çıktılar

| Dosya | İçerik |
|-------|--------|
| `data/raw/lcw_products.csv` | Ürün listesi |
| `data/raw/lcw_reviews.csv` | Yorumlar |
| `data/processed/summary.csv` | Ürün bazlı özet |
| `outputs/figures/*.png` | Grafikler |
| `outputs/reports/dashboard.html` | İnteraktif dashboard |

## 🧠 Model Eğitimi

Detaylar için [MODEL.md](MODEL.md) dosyasına bakın.

```bash
python scripts/train_sentiment.py --model tfidf
python scripts/train_sentiment.py --model berturk
```

### Model Metrikleri (Test Seti: 7.986 örnek, binary)

| Metrik | Değer |
|--------|-------|
| **Accuracy** | **0.9274** |
| **Macro F1** | **0.8582** |
| **5-fold CV Accuracy** | **0.9294 ± 0.0047** |
| **5-fold CV Macro F1** | **0.8586 ± 0.0076** |

> **Not:** Model, 3-sınıf (pos/neutral/neg) yerine **binary (pos/neg)** olarak eğitildi. 
> 3 yıldızlı (neutral) yorumlar eğitim dışı bırakıldı; bu, %81'den %93 accuracy'ye 
> sıçrama sağladı.

## ⚠️ Yasal Uyarı

Bu proje **eğitim ve portföy** amaçlıdır. Scraping yaparken:
- `robots.txt` ve kullanım şartlarına uyulur
- Rate limit (≥1.5 sn/istek) uygulanır
- Toplanan veriler **anonimleştirilir** (yazar isimleri maskelenir)
- Ticari kullanım için LCW'dan izin alınması gerekir

## 📜 Lisans

MIT — detaylar için [LICENSE](LICENSE)