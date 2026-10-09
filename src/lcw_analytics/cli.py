"""Komut satırı arayüzü: `lcw scrape | analyze | visualize | all`."""

import sys
from datetime import datetime

import pandas as pd
from rich.console import Console

from . import scraper, sentiment, analysis, visualize
from .config import PRODUCTS_CSV, REVIEWS_CSV, SUMMARY_CSV

console = Console()


def cmd_scrape():
    console.print("[bold blue]→ Ürünler çekiliyor...[/]")
    products = scraper.fetch_products()
    if not products:
        console.print("[red]❌ Ürün çekilemedi[/]")
        return
    df_p = pd.DataFrame(products)
    df_p.to_csv(PRODUCTS_CSV, index=False)
    console.print(f"[green]✅ {len(df_p)} ürün → {PRODUCTS_CSV}[/]")

    console.print("[bold blue]→ Yorumlar çekiliyor...[/]")
    all_reviews = []
    for i, p, revs in scraper.fetch_all_reviews(products):
        all_reviews.extend(revs)
        if i % 10 == 0:
            console.print(f"  [{i}/{len(products)}] toplam yorum: {len(all_reviews)}")

    df_r = pd.DataFrame(all_reviews)
    df_r.to_csv(REVIEWS_CSV, index=False)
    console.print(f"[green]✅ {len(df_r)} yorum → {REVIEWS_CSV}[/]")


def cmd_analyze():
    if not PRODUCTS_CSV.exists() or not REVIEWS_CSV.exists():
        console.print("[red]❌ Önce `lcw scrape` çalıştırın[/]")
        return
    df_p = pd.read_csv(PRODUCTS_CSV)
    df_r = pd.read_csv(REVIEWS_CSV)

    console.print("[bold blue]→ Sentiment analizi...[/]")
    df_r = sentiment.apply_ml(df_r)

    console.print("[bold blue]→ Ürün özeti...[/]")
    df_s = analysis.product_summary(df_r, df_p)
    df_s.to_csv(SUMMARY_CSV, index=False)

    console.print(f"[green]✅ {len(df_s)} ürün özeti → {SUMMARY_CSV}[/]")


def cmd_visualize():
    if not SUMMARY_CSV.exists():
        console.print("[red]❌ Önce `lcw analyze` çalıştırın[/]")
        return
    df_p = pd.read_csv(PRODUCTS_CSV)
    df_r = pd.read_csv(REVIEWS_CSV)
    df_s = pd.read_csv(SUMMARY_CSV)
    # sentiment sütunu yoksa ekle
    if "sentiment" not in df_r.columns:
        df_r = sentiment.apply_ml(df_r)

    run_date = datetime.now().strftime("%Y-%m-%d")
    p1 = visualize.plot_ratings_and_sentiment(df_r, df_s)
    p2 = visualize.plot_category_and_price(df_s)
    p3 = visualize.plot_wordcloud(df_r)
    p4 = visualize.build_html_dashboard(df_p, df_r, df_s, run_date)

    console.print(f"[green]✅ Grafikler: {p1}, {p2}, {p3}[/]")
    console.print(f"[green]✅ Dashboard: {p4}[/]")


def main():
    args = sys.argv[1:]
    if not args or args[0] == "all":
        cmd_scrape()
        cmd_analyze()
        cmd_visualize()
        return
    cmd = args[0]
    if cmd == "scrape":
        cmd_scrape()
    elif cmd == "analyze":
        cmd_analyze()
    elif cmd == "visualize":
        cmd_visualize()
    else:
        console.print(f"[red]Bilinmeyen komut: {cmd}[/]")
        console.print("Kullanım: lcw [scrape|analyze|visualize|all]")


if __name__ == "__main__":
    main()