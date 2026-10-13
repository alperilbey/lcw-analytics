"""Model yükleme ve tahmin servisi — singleton pattern."""

from functools import lru_cache

import joblib
import numpy as np

from ..config import MODEL_PATH, VECTORIZER_PATH, MODEL_DIR
from ..sentiment import _temizle


class SentimentService:
    """Eğitilmiş modeli yükler ve tahmin yapar."""

    def __init__(self):
        self._clf = None
        self._vec = None
        self._labels = None

    def load(self):
        """Model dosyalarını yükle."""
        if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists()):
            raise FileNotFoundError(
                f"Model bulunamadı: {MODEL_PATH}. "
                "Önce `python scripts/train_sentiment.py` çalıştırın."
            )
        self._clf = joblib.load(MODEL_PATH)
        self._vec = joblib.load(VECTORIZER_PATH)
        self._labels = list(self._clf.classes_)

    @property
    def is_loaded(self) -> bool:
        return self._clf is not None and self._vec is not None

    def predict(self, text: str) -> dict:
        """Tek metin için tahmin."""
        if not self.is_loaded:
            self.load()

        cleaned = _temizle(text)
        if not cleaned.strip():
            return {
                "text": text,
                "label": "neutral",
                "confidence": 0.0,
                "scores": {},
            }

        X = self._vec.transform([cleaned])
        label = self._clf.predict(X)[0]
        proba = self._clf.predict_proba(X)[0]
        scores = {str(l): float(p) for l, p in zip(self._labels, proba)}

        return {
            "text": text,
            "label": str(label),
            "confidence": float(np.max(proba)),
            "scores": scores,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Çoklu metin için tahmin (tek transform ile hızlı)."""
        if not self.is_loaded:
            self.load()

        cleaned = [_temizle(t) for t in texts]
        # Boş olanları işaretle
        valid_idx = [i for i, c in enumerate(cleaned) if c.strip()]
        results = [None] * len(texts)

        if valid_idx:
            X = self._vec.transform([cleaned[i] for i in valid_idx])
            labels = self._clf.predict(X)
            probas = self._clf.predict_proba(X)

            for idx, label, proba in zip(valid_idx, labels, probas):
                results[idx] = {
                    "text": texts[idx],
                    "label": str(label),
                    "confidence": float(np.max(proba)),
                    "scores": {str(l): float(p) for l, p in
                               zip(self._labels, proba)},
                }

        # Boş olanlar için default
        for i, r in enumerate(results):
            if r is None:
                results[i] = {
                    "text": texts[i],
                    "label": "neutral",
                    "confidence": 0.0,
                    "scores": {},
                }
        return results


@lru_cache(maxsize=1)
def get_service() -> SentimentService:
    """Singleton servis."""
    svc = SentimentService()
    try:
        svc.load()
    except FileNotFoundError:
        pass  # Lazy load on first request
    return svc