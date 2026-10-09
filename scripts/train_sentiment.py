"""Sentiment modeli eğitim script'i.

Kullanım:
    python scripts/train_sentiment.py
"""

import sys
import sqlite3
from pathlib import Path
import pandas as pd
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcw_analytics.config import REVIEWS_CSV, MODEL_DIR
from lcw_analytics.model import build_training_data, train_tfidf

console = Console()


def load_lcw_data() -> pd.DataFrame:
    """LCW scraper çıktısını yükler."""
    if not REVIEWS_CSV.exists():
        console.print(f"[yellow]⚠️ {REVIEWS_CSV} yok — LCW verisi atlanıyor.[/]")
        return pd.DataFrame()
    df = pd.read_csv(REVIEWS_CSV)
    console.print(f"[blue]→ {len(df)} LCW yorumu yüklendi.[/]")
    return df[["review_text", "review_rating"]].copy()


def load_trendyol_data() -> pd.DataFrame:
    """Kaggle Trendyol SQLite veritabanını yükler."""
    db_path = Path("data/raw/TrendyolProduct.sqlite3")
    if not db_path.exists():
        console.print(f"[yellow]⚠️ {db_path} yok — Trendyol verisi atlanıyor.[/]")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT Comment_Content, Comment_Evaluation FROM TBL_Comment",
            conn,
        )
    finally:
        conn.close()

    # Sütun adlarını projedeki standart isimlere çevir
    df = df.rename(columns={
        "Comment_Content":    "review_text",
        "Comment_Evaluation": "review_rating",
    })
    console.print(f"[blue]→ {len(df)} Trendyol yorumu yüklendi.[/]")
    return df


def train_tfidf_pipeline():
    console.print("[bold]🎯 Sentiment modeli eğitimi başlıyor...[/]\n")

    # 1) Veri setlerini yükle
    df_lcw = load_lcw_data()
    df_trendyol = load_trendyol_data()

    # 2) Birleştir
    frames = [d for d in (df_lcw, df_trendyol) if not d.empty]
    if not frames:
        console.print("[red]❌ Hiçbir veri seti bulunamadı.[/]")
        return

    df_all = pd.concat(frames, ignore_index=True)
    console.print(f"\n[bold]📊 Toplam {len(df_all)} yorum ile eğitim yapılacak.[/]\n")

    # 3) Eğitim verisini hazırla (weak supervision)
    X, y = build_training_data(df_all)
    console.print(f"[blue]→ Eğitim verisi: {len(X)} örnek[/]")
    console.print(f"[blue]→ Sınıf dağılımı: {pd.Series(y).value_counts().to_dict()}[/]")

    # 4) Modeli eğit
    metrics = train_tfidf(X, y)

    console.print("\n[bold green]✅ Eğitim tamamlandı[/]")
    console.print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    console.print(f"  Macro F1:   {metrics['macro_f1']:.4f}")
    console.print(f"  Train set:  {metrics['n_train']} örnek")
    console.print(f"  Test set:   {metrics['n_test']} örnek")
    console.print(f"\n[green]📁 Model kaydedildi: {MODEL_DIR}[/]")


if __name__ == "__main__":
    train_tfidf_pipeline()