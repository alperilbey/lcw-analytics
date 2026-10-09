# Models

Eğitilmiş duygu analizi modelleri bu klasörde saklanır.

## Beklenen Dosyalar

- `sentiment/model.joblib` → TF-IDF + LogisticRegression modeli
- `sentiment/vectorizer.joblib` → TF-IDF vektörleştirici
- `sentiment/metrics.json` → doğruluk, F1, sınıflandırma raporu

## Eğitim

```bash
python scripts/train_sentiment.py --model tfidf