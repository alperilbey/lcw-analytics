"""Türkçe duygu analizi — kural tabanlı + ML model desteği."""

import re
from typing import Literal

import pandas as pd

# --- Sözlükler ---
POZITIF = {
    "güzel", "harika", "mükemmel", "muhteşem", "kaliteli", "şahane", "süper",
    "beğendim", "tavsiye", "memnun", "memnunum", "iyi", "tam", "uygun",
    "rahat", "şık", "zarif", "kullanışlı", "sağlam", "dayanıklı", "yumuşak",
    "sıcak", "tutuyor", "yakıştı", "yakışmış", "beklentimi", "karşıladı",
    "hesaplı", "ekonomik", "hızlı", "kargo", "teşekkür", "teşekkürler",
    "başarılı", "kusursuz", "sevdim", "bayıldım", "efsane", "favori",
    "tekrar", "alacağım", "alırım", "kesinlikle", "mutlaka", "çok",
}
NEGATIF = {
    "kötü", "berbat", "rezalet", "çirkin", "kalitesiz", "kullanışsız",
    "beğenmedim", "pişman", "pişmanım", "iade", "geri", "gönderdim",
    "kırık", "yırtık", "leke", "lekeli", "defolu", "hatalı", "yanlış",
    "eksik", "küçük", "büyük", "bol", "dar", "uzun", "kısa", "farklı",
    "beklemediğim", "karşılamadı", "hayal", "kırıklığı", "yetersiz",
    "pahalı", "geç", "yavaş", "tutmuyor", "yakışmadı", "söküldü",
    "attı", "çekti", "düşük", "sorun", "sorunlu", "problem", "bozuk",
    "kokuyor", "koku", "rahatsız",
}
NEGASYON = {"değil", "yok", "asla", "hiç", "değildir", "olmaz"}


def _temizle(text) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"[^\wçğıöşüâîû\s]", " ", str(text).lower())


def rule_based_sentiment(text) -> tuple[str, float]:
    """Kural tabanlı Türkçe sentiment — (label, score) döner.

    Türkçe'de negasyon kelimesi genelde duygu kelimesinden SONRA gelir:
    "güzel değil", "iyi değil", "beğenmedim" vb.
    Bu yüzden negasyon kontrolü ileriye doğru yapılır.
    """
    text = _temizle(text)
    if not text:
        return "neutral", 0.0

    words = text.split()
    poz, neg = 0, 0
    i = 0
    while i < len(words):
        w = words[i]
        bigram = " ".join(words[i:i + 2]) if i + 1 < len(words) else ""

        # 1) Bigram kontrolü (2 kelimelik öbek)
        if bigram in POZITIF:
            poz += 1
            i += 2
            continue
        if bigram in NEGATIF:
            neg += 1
            i += 2
            continue

        # 2) Negasyon kontrolü — SONRAKİ kelimeye bak (Türkçe için)
        next_is_neg = (i + 1 < len(words)) and (words[i + 1] in NEGASYON)

        if w in POZITIF:
            if next_is_neg:
                neg += 1
                i += 2          # hem duygu kelimesini hem negasyonu atla
                continue
            poz += 1
        elif w in NEGATIF:
            if next_is_neg:
                poz += 1
                i += 2
                continue
            neg += 1
        # Negasyon kelimesinin kendisi (değil/yok) tek başına sayılmaz
        i += 1

    score = poz - neg
    norm = score / max(poz + neg, 1)
    if score > 0 and poz >= neg:
        return "positive", norm
    if score < 0 and neg > poz:
        return "negative", norm
    return "neutral", norm

def apply_rule_based(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """DataFrame'e sentiment sütunları ekler."""
    df = df.copy()
    df[["sentiment", "sentiment_score"]] = df[text_col].apply(
        lambda t: pd.Series(rule_based_sentiment(t))
    )
    return df


# --- ML model desteği (BÖLÜM 2'de eğitilecek) ---

def _load_ml_model():
    """Eğitilmiş modeli yükler (varsa)."""
    try:
        import joblib
        from .config import MODEL_PATH, VECTORIZER_PATH
        if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists()):
            return None, None
        model = joblib.load(MODEL_PATH)
        vec = joblib.load(VECTORIZER_PATH)
        return model, vec
    except Exception:
        return None, None


_MODEL, _VEC = None, None


def ml_sentiment(text) -> tuple[str, float]:
    """Eğitilmiş ML modeli ile tahmin. Model yoksa kural tabanlıya düşer."""
    global _MODEL, _VEC
    if _MODEL is None:
        _MODEL, _VEC = _load_ml_model()

    if _MODEL is None:
        return rule_based_sentiment(text)

    text = _temizle(text)
    if not text:
        return "neutral", 0.0

    X = _VEC.transform([text])
    label = _MODEL.predict(X)[0]
    proba = _MODEL.predict_proba(X)[0]
    score = float(max(proba)) * (1 if label == "positive" else -1 if label == "negative" else 0)
    return str(label), score


def apply_ml(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    df = df.copy()
    df[["sentiment", "sentiment_score"]] = df[text_col].apply(
        lambda t: pd.Series(ml_sentiment(t))
    )
    return df