"""lcw.com scraper — ürünler ve yorumlar."""

import json
import time
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .config import (
    BASE_URL, KADIN_URL, HEADERS, REQUEST_TIMEOUT, REQUEST_DELAY, MAX_PRODUCTS
)


# ---------- HTTP ----------

def _get(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            print(f"⚠️ Status {r.status_code}: {url}")
            return None
        return r.text
    except requests.RequestException as e:
        print(f"❌ {type(e).__name__}: {url}")
        return None


# ---------- Ürün listesi ----------

def extract_catalog_model(html: str) -> Optional[dict]:
    """`var catalogModel = {...}` bloğunu dengeli parantezle çıkarır."""
    marker = "var catalogModel = "
    idx = html.find(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(html)):
        ch = html[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(html[start:i + 1])
    return None


def _parse_price(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace("TL", "").strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_product(p: dict) -> dict:
    bc = p.get("BreadCrump") or {}
    img = p.get("DefaultOptionImageUrl")
    if not img:
        imgs = p.get("OptionImageList") or []
        if imgs and isinstance(imgs[0], dict):
            img = imgs[0].get("MediumImage") or imgs[0].get("LargeImage")

    return {
        "product_id":   p.get("OptionId"),
        "model_id":     p.get("ModelId"),
        "product_name": p.get("ProductDescription"),
        "brand":        p.get("Brand") or "LC Waikiki",
        "category":     bc.get("Level3") or p.get("Category") or bc.get("Level1"),
        "subcategory":  bc.get("Level4"),
        "gender":       bc.get("Level1"),
        "price":        p.get("PriceValue") or _parse_price(p.get("Price")),
        "old_price":    _parse_price(p.get("OldPrice")),
        "discount_pct": p.get("DiscountRatio") or 0,
        "rating":       p.get("AvgScore") or 0.0,
        "review_count": p.get("ReviewCount") or 0,
        "stock":        p.get("AvailableStock"),
        "color_hex":    p.get("MainColorHexColor") or p.get("MainColorHexCode"),
        "image_url":    img,
        "product_url":  BASE_URL + p["ModelUrl"] if p.get("ModelUrl") else None,
    }


def fetch_products(url: str = KADIN_URL) -> list[dict]:
    """Kategori sayfasından ürün listesini çeker."""
    html = _get(url)
    if not html:
        return []
    catalog = extract_catalog_model(html)
    if not catalog or "CatalogList" not in catalog:
        return []
    items = catalog["CatalogList"]["Items"][:MAX_PRODUCTS]
    products = [_normalize_product(p) for p in items]
    return [p for p in products if p["product_name"]]


# ---------- Yorumlar ----------

def _parse_reviews(html: str, product: dict) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    product_ld = None
    for b in soup.find_all("script", type="application/ld+json"):
        try:
            j = json.loads(b.string or "{}")
            if j.get("@type") == "Product":
                product_ld = j
                break
        except Exception:
            continue

    if not product_ld:
        return []

    agg = product_ld.get("aggregateRating") or {}
    avg_rating = agg.get("ratingValue")
    total = agg.get("reviewCount")

    reviews = product_ld.get("review", [])
    if not isinstance(reviews, list):
        reviews = [reviews] if reviews else []

    rows = []
    for r in reviews:
        author = r.get("author")
        if isinstance(author, dict):
            author = author.get("name")
        ro = r.get("reviewRating") or {}
        rating = ro.get("ratingValue") if isinstance(ro, dict) else ro

        rows.append({
            "product_id":     product["product_id"],
            "product_name":   product["product_name"],
            "category":       product["category"],
            "brand":          product["brand"],
            "avg_rating":     avg_rating,
            "total_reviews":  total,
            "review_author":  author,
            "review_rating":  rating,
            "review_text":    r.get("reviewBody") or r.get("description"),
            "review_date":    r.get("datePublished"),
        })
    return rows


def fetch_reviews(product: dict) -> list[dict]:
    url = product.get("product_url")
    if not url:
        return []
    html = _get(url)
    if not html:
        return []
    return _parse_reviews(html, product)


def fetch_all_reviews(products: list[dict], delay: float = REQUEST_DELAY):
    """Tüm ürünler için yorumları sıralı çeker. Generator döner."""
    for i, p in enumerate(products, 1):
        revs = fetch_reviews(p)
        yield i, p, revs
        time.sleep(delay)