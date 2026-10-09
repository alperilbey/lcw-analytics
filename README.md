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

## ⚠️ Yasal Uyarı

Bu proje **eğitim ve portföy** amaçlıdır. Scraping yaparken:
- `robots.txt` ve kullanım şartlarına uyulur
- Rate limit (≥1.5 sn/istek) uygulanır
- Toplanan veriler **anonimleştirilir** (yazar isimleri maskelenir)
- Ticari kullanım için LCW'dan izin alınması gerekir

## 📜 Lisans

MIT — detaylar için [LICENSE](LICENSE)