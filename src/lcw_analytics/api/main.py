"""FastAPI uygulaması — sentiment inference servisi."""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import MODEL_DIR
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)
from .service import get_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Başlangıçta modeli yükle."""
    print("🚀 Model yükleniyor...")
    svc = get_service()
    if not svc.is_loaded:
        try:
            svc.load()
            print("✅ Model hazır")
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
            print("   /predict çağrıldığında tekrar denenecek")
    yield
    print("👋 Servis kapanıyor")


app = FastAPI(
    title="LCW Sentiment API",
    description=(
        "Türkçe e-ticaret yorumları için duygu analizi servisi. "
        "TF-IDF + LogisticRegression modeli, 44k yorum ile eğitildi."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.get("/", tags=["root"])
async def root():
    return {
        "message": "LCW Sentiment API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    svc = get_service()
    return HealthResponse(
        status="ok",
        model_loaded=svc.is_loaded,
        version=__version__,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["system"])
async def metrics():
    """Eğitim metriklerini döner."""
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "metrics.json bulunamadı")
    with open(path, encoding="utf-8") as f:
        m = json.load(f)

    # Binary model için metrics.json yapısı farklı olabilir
    return MetricsResponse(
        trained_at=m.get("trained_at"),
        accuracy=m.get("accuracy") or m.get("cv_results", {}).get("acc_mean"),
        macro_f1=m.get("macro_f1") or m.get("cv_results", {}).get("f1_mean"),
        n_train=m.get("n_train"),
        n_test=m.get("n_test"),
        model_type=m.get("model_type", "TF-IDF + LogisticRegression"),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(req: PredictRequest):
    """Tek bir yorum için duygu tahmini."""
    svc = get_service()
    try:
        result = svc.predict(req.text)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse,
           tags=["inference"])
async def predict_batch(req: BatchPredictRequest):
    """Birden fazla yorum için toplu tahmin (max 100)."""
    svc = get_service()
    try:
        results = svc.predict_batch(req.texts)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return BatchPredictResponse(
        results=[PredictResponse(**r) for r in results],
        count=len(results),
    )