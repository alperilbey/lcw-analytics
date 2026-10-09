"""Eğitilebilir sentiment modeli — eğitim, kaydetme, yükleme.

Weak supervision ile yıldız puanlarından etiket üretir:
    1-2 yıldız  → negative
    3 yıldız    → neutral
    4-5 yıldız  → positive
"""

import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from .config import MODEL_DIR, MODEL_PATH, VECTORIZER_PATH
from .sentiment import _temizle


def star_to_label(star) -> str:
    """Yıldız puanını (1-5) duygu etiketine çevirir."""
    try:
        s = float(star)
    except (TypeError, ValueError):
        return "neutral"
    if s <= 2:
        return "negative"
    if s == 3:
        return "neutral"
    return "positive"


def build_training_data(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(X, y) döner — temizlenmiş metin ve etiket."""
    df = df.dropna(subset=["review_text", "review_rating"]).copy()
    df["text"] = df["review_text"].astype(str).apply(_temizle)
    df = df[df["text"].str.len() > 5]
    df["label"] = df["review_rating"].apply(star_to_label)
    return df["text"].tolist(), df["label"].tolist()


def train_tfidf(
    X: list[str],
    y: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """TF-IDF + LogisticRegression eğitir, kaydeder ve metrikleri döner."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50_000,
    )
    X_train_v = vec.fit_transform(X_train)
    X_test_v = vec.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train_v, y_train)

    y_pred = clf.predict(X_test_v)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "trained_at": datetime.now().isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "labels": sorted(set(y)),
    }

    # Kaydet
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vec, VECTORIZER_PATH)

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def load_model():
    """Eğitilmiş modeli yükler (yoksa FileNotFoundError)."""
    if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists()):
        raise FileNotFoundError(
            "Model bulunamadı. Önce `python scripts/train_sentiment.py` çalıştır."
        )
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def predict(texts: list[str]) -> list[dict]:
    """Yeni yorumlar için tahmin döner."""
    clf, vec = load_model()
    cleaned = [_temizle(t) for t in texts]
    X = vec.transform(cleaned)
    labels = clf.predict(X)
    probas = clf.predict_proba(X)
    return [
        {"text": t, "label": l, "confidence": round(float(max(p)), 4)}
        for t, l, p in zip(texts, labels, probas)
    ]