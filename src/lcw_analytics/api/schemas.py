"""Pydantic request/response şemaları."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      description="Analiz edilecek yorum metni")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "Kumaşı çok kaliteli, bayıldım."},
                {"text": "Berbat bir ürün, iade ettim."},
            ]
        }
    }


class PredictResponse(BaseModel):
    text: str
    label: str = Field(..., description="positive | negative | neutral")
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict,
                                     description="Sınıf bazlı olasılıklar")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    trained_at: str | None = None
    accuracy: float | None = None
    macro_f1: float | None = None
    n_train: int | None = None
    n_test: int | None = None
    model_type: str | None = None