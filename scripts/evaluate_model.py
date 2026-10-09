"""Eğitilmiş sentiment modelinin metriklerini görüntüler.

Kullanım:
    python scripts/evaluate_model.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcw_analytics.config import MODEL_DIR


def main():
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        print(f"❌ {metrics_path} bulunamadı — önce eğit:")
        print("   python scripts/train_sentiment.py")
        return

    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)

    print("=" * 55)
    print("📊 SENTIMENT MODELİ — DEĞERLENDİRME RAPORU")
    print("=" * 55)
    print(f"📅 Eğitim tarihi:  {m['trained_at']}")
    print(f"🎯 Accuracy:       {m['accuracy']:.4f}")
    print(f"📊 Macro F1:       {m['macro_f1']:.4f}")
    print(f"📚 Train seti:     {m['n_train']} örnek")
    print(f"📚 Test seti:      {m['n_test']} örnek")
    print()
    print("📋 Sınıf bazlı metrikler:")
    print(f"  {'Sınıf':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Destek':>10}")
    print("  " + "-" * 52)

    for label, scores in m["classification_report"].items():
        if isinstance(scores, dict):
            print(f"  {label:<12} "
                  f"{scores['precision']:>10.3f} "
                  f"{scores['recall']:>10.3f} "
                  f"{scores['f1-score']:>10.3f} "
                  f"{scores['support']:>10.0f}")
        else:
            # "accuracy" ve "macro avg" gibi sayısal satırlar
            print(f"  {label:<12} {'':>10} {'':>10} {scores:>10.3f}")

    print()
    print("🔢 Confusion Matrix (satır=gerçek, kolon=tahmin):")
    labels = m["labels"]
    print(f"  {'':>10}", end="")
    for l in labels:
        print(f"{l:>10}", end="")
    print()
    for i, row in enumerate(m["confusion_matrix"]):
        print(f"  {labels[i]:>10}", end="")
        for val in row:
            print(f"{val:>10}", end="")
        print()
    print("=" * 55)


if __name__ == "__main__":
    main()