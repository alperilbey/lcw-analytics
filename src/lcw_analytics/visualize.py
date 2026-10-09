"""Görselleştirme fonksiyonları."""

import matplotlib
matplotlib.use("Agg")   # headless ortamlar için
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud

from .config import FIGURES_DIR, REPORTS_DIR


def plot_ratings_and_sentiment(df_reviews, df_summary, fname="01_ratings_sentiment.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.countplot(x="review_rating", data=df_reviews, ax=axes[0], color="steelblue")
    axes[0].set_title("Puan Dağılımı")

    sns.countplot(
        x="sentiment", data=df_reviews, ax=axes[1],
        order=["positive", "neutral", "negative"],
        hue="sentiment",
        palette={"positive": "#55A868", "neutral": "#C7C7C7", "negative": "#C44E52"},
        legend=False,
    )
    axes[1].set_title("Duygu Dağılımı")

    top10 = df_summary.head(10).sort_values("yorum_sayisi")
    axes[2].barh(top10["product_name"].str.slice(0, 35), top10["yorum_sayisi"], color="coral")
    axes[2].set_title("En Çok Yorumlanan 10 Ürün")

    plt.tight_layout()
    path = FIGURES_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_category_and_price(df_summary, fname="02_category_price.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cat = df_summary.groupby("category").agg(
        ort_puan=("ort_puan", "mean"),
        yorum=("yorum_sayisi", "sum"),
    ).sort_values("yorum", ascending=False).head(10)

    axes[0].barh(cat.index, cat["ort_puan"], color="mediumseagreen")
    axes[0].set_xlim(0, 5)
    axes[0].set_title("Kategori Bazlı Ortalama Puan")

    valid = df_summary.dropna(subset=["price", "pozitif_oran"])
    axes[1].scatter(
        valid["price"], valid["pozitif_oran"] * 100,
        alpha=0.6, s=valid["yorum_sayisi"] * 20, c="steelblue",
    )
    axes[1].set_xlabel("Fiyat (TL)")
    axes[1].set_ylabel("Pozitif Oran (%)")
    axes[1].set_title("Fiyat vs Pozitif Yorum")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_wordcloud(df_reviews, fname="03_wordcloud.png"):
    text = " ".join(df_reviews["review_text"].dropna().astype(str))
    if not text.strip():
        return None

    wc = WordCloud(width=1200, height=600, background_color="white",
                   colormap="viridis").generate(text)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    path = FIGURES_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def build_html_dashboard(df_products, df_reviews, df_summary,
                          run_date: str, fname="dashboard.html"):
    """Plotly tabanlı interaktif HTML dashboard."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Puan Dağılımı", "Duygu Dağılımı",
                        "En Beğenilen 10 Ürün", "Kategori Bazlı Ortalama"),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}]],
    )

    rating_counts = df_reviews["review_rating"].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=rating_counts.index.astype(str), y=rating_counts.values,
               marker_color="steelblue", name="Puan"),
        row=1, col=1,
    )

    sent_counts = df_reviews["sentiment"].value_counts()
    fig.add_trace(
        go.Pie(labels=sent_counts.index, values=sent_counts.values,
               marker_colors=["#55A868", "#C7C7C7", "#C44E52"]),
        row=1, col=2,
    )

    top10 = df_summary.head(10).sort_values("populerlik_skoru")
    fig.add_trace(
        go.Bar(x=top10["populerlik_skoru"],
               y=top10["product_name"].str.slice(0, 30),
               orientation="h", marker_color="coral"),
        row=2, col=1,
    )

    cat = df_summary.groupby("category")["ort_puan"].mean().sort_values().tail(10)
    fig.add_trace(
        go.Bar(x=cat.values, y=cat.index, orientation="h",
               marker_color="mediumseagreen"),
        row=2, col=2,
    )

    fig.update_layout(
        height=800, showlegend=False,
        title_text=f"LC Waikiki Kadın Kategorisi — Analiz Dashboard ({run_date})",
        title_x=0.5,
    )

    path = REPORTS_DIR / fname
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path