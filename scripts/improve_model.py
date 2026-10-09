"""Sentiment modeli iyileştirme deneyleri.

3 farklı yaklaşımı karşılaştırır:
    1. 3-sınıf (mevcut) vs 2-sınıf (binary: pos/neg)
    2. GridSearchCV ile hiperparametre optimizasyonu
    3. 5-fold Cross-Validation

En iyi modeli seçip models/sentiment/ altına kaydeder.
"""

import json
import sys
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcw_analytics.config import REVIEWS_CSV, MODEL_DIR, MODEL_PATH, VECTORIZER_PATH
from lcw_analytics.sentiment import _temizle

console = Console()


# ============================================================
# 1) VERİ YÜKLEME
# ============================================================

def star_to_label_3(star) -> str:
    """3-sınıflı etiket: neg / neutral / pos."""
    try:
        s = float(star)
    except (TypeError, ValueError):
        return "neutral"
    if s <= 2:
        return "negative"
    if s == 3:
        return "neutral"
    return "positive"


def star_to_label_2(star) -> str:
    """Binary etiket: sadece neg / pos (3 yıldız atılır)."""
    try:
        s = float(star)
    except (TypeError, ValueError):
        return None
    if s <= 2:
        return "negative"
    if s >= 4:
        return "positive"
    return None  # 3 yıldız atılır


def load_data() -> pd.DataFrame:
    """LCW + Trendyol verilerini yükleyip birleştirir."""
    frames = []

    # LCW
    if REVIEWS_CSV.exists():
        df_lcw = pd.read_csv(REVIEWS_CSV)[["review_text", "review_rating"]]
        frames.append(df_lcw)
        console.print(f"[blue]→ {len(df_lcw)} LCW yorumu[/]")

    # Trendyol SQLite
    db_path = Path("data/raw/TrendyolProduct.sqlite3")
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        try:
            df_ty = pd.read_sql_query(
                "SELECT Comment_Content, Comment_Evaluation FROM TBL_Comment", conn
            )
        finally:
            conn.close()
        df_ty = df_ty.rename(columns={
            "Comment_Content": "review_text",
            "Comment_Evaluation": "review_rating",
        })
        frames.append(df_ty)
        console.print(f"[blue]→ {len(df_ty)} Trendyol yorumu[/]")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["review_text", "review_rating"]).copy()
    df["text"] = df["review_text"].astype(str).apply(_temizle)
    df = df[df["text"].str.len() > 5].reset_index(drop=True)
    return df


# ============================================================
# 2) DENEY FONKSİYONLARI
# ============================================================

def experiment_binary_vs_3class(df: pd.DataFrame):
    """3-sınıf ve 2-sınıf modelleri karşılaştırır."""
    console.rule("[bold]DENEY 1: 3-sınıf vs Binary")

    results = {}

    # --- 3 sınıflı ---
    df3 = df.copy()
    df3["label"] = df3["review_rating"].apply(star_to_label_3)
    X, y = df3["text"].tolist(), df3["label"].tolist()

    pipe3 = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                                   sublinear_tf=True, max_features=50_000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                    C=1.0, n_jobs=-1, random_state=42)),
    ])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)
    pipe3.fit(X_tr, y_tr)
    y_pred = pipe3.predict(X_te)
    rep3 = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    results["3-sınıf"] = {
        "accuracy": accuracy_score(y_te, y_pred),
        "macro_f1": rep3["macro avg"]["f1-score"],
        "n_samples": len(X),
    }
    console.print(f"  3-sınıf  → acc={results['3-sınıf']['accuracy']:.4f}  "
                  f"macro_f1={results['3-sınıf']['macro_f1']:.4f}  "
                  f"(n={len(X)})")

    # --- Binary ---
    df2 = df.copy()
    df2["label"] = df2["review_rating"].apply(star_to_label_2)
    df2 = df2.dropna(subset=["label"])
    X2, y2 = df2["text"].tolist(), df2["label"].tolist()

    pipe2 = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                                   sublinear_tf=True, max_features=50_000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                    C=1.0, n_jobs=-1, random_state=42)),
    ])
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X2, y2, test_size=0.2,
                                                    random_state=42, stratify=y2)
    pipe2.fit(X_tr2, y_tr2)
    y_pred2 = pipe2.predict(X_te2)
    rep2 = classification_report(y_te2, y_pred2, output_dict=True, zero_division=0)
    results["binary"] = {
        "accuracy": accuracy_score(y_te2, y_pred2),
        "macro_f1": rep2["macro avg"]["f1-score"],
        "n_samples": len(X2),
        "report": rep2,
    }
    console.print(f"  Binary   → acc={results['binary']['accuracy']:.4f}  "
                  f"macro_f1={results['binary']['macro_f1']:.4f}  "
                  f"(n={len(X2)})")

    return results, pipe3, pipe2


def experiment_gridsearch(df: pd.DataFrame, sample_size: int = 20_000):
    """GridSearchCV ile en iyi hiperparametreleri bulur."""
    console.rule("[bold]DENEY 2: GridSearchCV (binary)")

    df2 = df.copy()
    df2["label"] = df2["review_rating"].apply(star_to_label_2)
    df2 = df2.dropna(subset=["label"])
    X, y = df2["text"].tolist(), df2["label"].tolist()

    # Hız için örnekle
    if len(X) > sample_size:
        X_s, _, y_s, _ = train_test_split(
            X, y, train_size=sample_size, random_state=42, stratify=y
        )
        X, y = X_s, y_s
        console.print(f"[yellow]→ Hız için {sample_size} örneğe indirildi[/]")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                    n_jobs=-1, random_state=42)),
    ])

    param_grid = {
        "tfidf__ngram_range":     [(1, 1), (1, 2)],
        "tfidf__min_df":          [2, 3],
        "tfidf__max_features":    [30_000, 50_000],
        "clf__C":                 [0.5, 1.0, 2.0],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline, param_grid,
        cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1,
    )
    grid.fit(X, y)

    console.print(f"\n[green]✅ En iyi skor (macro F1): {grid.best_score_:.4f}[/]")
    console.print(f"[green]En iyi parametreler:[/]")
    for k, v in grid.best_params_.items():
        console.print(f"   {k} = {v}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def experiment_cross_validation(df: pd.DataFrame, pipeline, n_splits: int = 5,
                                 sample_size: int = 25_000):
    """5-fold CV ile güvenilir metrikler üretir."""
    console.rule(f"[bold]DENEY 3: {n_splits}-fold Cross-Validation")

    df2 = df.copy()
    df2["label"] = df2["review_rating"].apply(star_to_label_2)
    df2 = df2.dropna(subset=["label"])
    X, y = df2["text"].tolist(), df2["label"].tolist()

    if len(X) > sample_size:
        X_s, _, y_s, _ = train_test_split(
            X, y, train_size=sample_size, random_state=42, stratify=y
        )
        X, y = X_s, y_s
        console.print(f"[yellow]→ Hız için {sample_size} örneğe indirildi[/]")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []

    for i, (tr, te) in enumerate(cv.split(X, y), 1):
        X_tr = [X[j] for j in tr]
        X_te = [X[j] for j in te]
        y_tr = [y[j] for j in tr]
        y_te = [y[j] for j in te]

        # Pipeline'ı klonla
        from sklearn.base import clone
        p = clone(pipeline)
        p.fit(X_tr, y_tr)
        y_pred = p.predict(X_te)

        a = accuracy_score(y_te, y_pred)
        f = classification_report(y_te, y_pred, output_dict=True,
                                   zero_division=0)["macro avg"]["f1-score"]
        accs.append(a)
        f1s.append(f)
        console.print(f"  Fold {i}: acc={a:.4f}  macro_f1={f:.4f}")

    import numpy as np
    console.print(f"\n[green]📊 Ortalama accuracy: {np.mean(accs):.4f} "
                  f"(± {np.std(accs):.4f})[/]")
    console.print(f"[green]📊 Ortalama macro F1: {np.mean(f1s):.4f} "
                  f"(± {np.std(f1s):.4f})[/]")

    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}


# ============================================================
# 3) KAYDETME
# ============================================================

def save_best_model(pipeline, metrics: dict):
    """En iyi pipeline'ı models/sentiment/ altına kaydeder."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["clf"], MODEL_PATH)
    joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_PATH)

    metrics["trained_at"] = datetime.now().isoformat()
    metrics["model_type"] = "TF-IDF + LogisticRegression (binary)"
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    console.print(f"\n[green]✅ Model kaydedildi: {MODEL_DIR}[/]")


# ============================================================
# 4) ANA AKIŞ
# ============================================================

def main():
    console.print("[bold]🎯 Sentiment Model İyileştirme Başlıyor...[/]\n")

    df = load_data()
    console.print(f"\n[bold]📊 Toplam {len(df)} yorum[/]\n")

    # Deney 1: 3-sınıf vs binary
    results, pipe3, pipe2 = experiment_binary_vs_3class(df)

    # Karar: Binary daha iyi ise onunla devam
    if results["binary"]["macro_f1"] > results["3-sınıf"]["macro_f1"]:
        console.print("\n[green]→ Binary sınıflandırma daha iyi, "
                      "GridSearch'e binary ile devam[/]\n")
        working_pipe = pipe2
    else:
        console.print("\n[yellow]→ 3-sınıf daha iyi, ama yine de binary "
                      "GridSearch denenecek[/]\n")
        working_pipe = pipe2

    # Deney 2: GridSearch
    best_pipe, best_params, best_score = experiment_gridsearch(df)

    # Deney 3: Cross-Validation
    cv_results = experiment_cross_validation(df, best_pipe)

    # Rapor
    console.rule("[bold]SONUÇ")
    table = Table(title="Deney Karşılaştırması")
    table.add_column("Deney", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("Macro F1", justify="right")

    table.add_row("3-sınıf (mevcut)",
                  f"{results['3-sınıf']['accuracy']:.4f}",
                  f"{results['3-sınıf']['macro_f1']:.4f}")
    table.add_row("Binary (baseline)",
                  f"{results['binary']['accuracy']:.4f}",
                  f"{results['binary']['macro_f1']:.4f}")
    table.add_row("GridSearch best",
                  "—",
                  f"{best_score:.4f}")
    table.add_row("5-fold CV ort.",
                  f"{cv_results['acc_mean']:.4f} ± {cv_results['acc_std']:.4f}",
                  f"{cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
    console.print(table)

    # Kaydet
    final_metrics = {
        "best_params": best_params,
        "gridsearch_score": best_score,
        "cv_results": cv_results,
        "baseline_3class": results["3-sınıf"],
        "baseline_binary": {
            "accuracy": results["binary"]["accuracy"],
            "macro_f1": results["binary"]["macro_f1"],
            "n_samples": results["binary"]["n_samples"],
            "classification_report": results["binary"]["report"],
        },
    }
    save_best_model(best_pipe, final_metrics)

    console.print("\n[bold green]🎉 İyileştirme tamamlandı![/]")
    console.print("[blue]→ Yeni modeli kullanmak için:[/]")
    console.print("   python -m lcw_analytics analyze")


if __name__ == "__main__":
    main()