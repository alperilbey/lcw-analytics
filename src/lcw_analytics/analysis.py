"""İstatistiksel analiz ve özet çıkarma."""

import pandas as pd


def product_summary(df_reviews: pd.DataFrame, df_products: pd.DataFrame) -> pd.DataFrame:
    """Ürün bazlı özet — yorum sayısı, puan, pozitif oran, popülerlik."""
    s = df_reviews.groupby(["product_id", "product_name"]).agg(
        yorum_sayisi=("review_text", "count"),
        ort_puan=("review_rating", "mean"),
        pozitif_oran=("sentiment", lambda x: (x == "positive").mean()),
        negatif_oran=("sentiment", lambda x: (x == "negative").mean()),
        ort_sentiment=("sentiment_score", "mean"),
    ).reset_index()

    s = s.merge(
        df_products[["product_id", "category", "brand", "price", "discount_pct", "stock"]],
        on="product_id", how="left"
    )
    s["populerlik_skoru"] = s["yorum_sayisi"] * s["pozitif_oran"] * s["ort_puan"]
    return s.sort_values("populerlik_skoru", ascending=False)


def category_summary(df_summary: pd.DataFrame) -> pd.DataFrame:
    """Kategori bazlı özet."""
    return df_summary.groupby("category").agg(
        urun_sayisi=("product_id", "count"),
        toplam_yorum=("yorum_sayisi", "sum"),
        ort_puan=("ort_puan", "mean"),
        ort_pozitif=("pozitif_oran", "mean"),
        ort_fiyat=("price", "mean"),
    ).reset_index().sort_values("toplam_yorum", ascending=False)


def kpi_dict(df_products, df_reviews, df_summary) -> dict:
    return {
        "toplam_urun": len(df_products),
        "yorumu_olan_urun": len(df_summary),
        "toplam_yorum": len(df_reviews),
        "ortalama_puan": round(df_reviews["review_rating"].mean(), 2),
        "pozitif_oran": round((df_reviews["sentiment"] == "positive").mean(), 3),
        "negatif_oran": round((df_reviews["sentiment"] == "negative").mean(), 3),
    }