"""Merkezi yapılandırma — tüm yollar ve sabitler burada."""

from pathlib import Path

# --- Yollar ---
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

for _d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# --- URL ---
BASE_URL = "https://www.lcw.com"
KADIN_URL = f"{BASE_URL}/kadin-giyim-t-10"

# --- HTTP ---
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
    "Referer": f"{BASE_URL}/",
}

REQUEST_TIMEOUT = 25
REQUEST_DELAY = 1.5   # saniye
MAX_PRODUCTS = 200    # güvenlik sınırı

# --- Dosya adları ---
PRODUCTS_CSV = RAW_DIR / "lcw_products.csv"
REVIEWS_CSV = RAW_DIR / "lcw_reviews.csv"
SUMMARY_CSV = PROCESSED_DIR / "summary.csv"

# --- Model ---
MODEL_DIR = MODELS_DIR / "sentiment"
MODEL_PATH = MODEL_DIR / "model.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"