import json
import sys
from pathlib import Path as _Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# When running scripts/* directly, ensure project root is on sys.path so
# top-level package imports like `src` work.
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_holdout, resolve_data_path
from src.intent_model import build_intent_pipeline

PLOTS = ROOT / "models" / "plots"


def load_config() -> tuple[str, str]:
    path = ROOT / "models" / "best_config.json"
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        return cfg.get("model_name", "logistic_regression"), cfg.get(
            "feature_set", "combined"
        )
    return "logistic_regression", "combined"


def plot_comparison() -> None:
    path = ROOT / "models" / "model_comparison.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["by_feature_set"]["combined"]["ranking"]
    names = [r["model"] for r in rows]
    f1 = [r["cv_f1_macro_mean"] for r in rows]
    acc = [r["test_accuracy"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, f1, width=w, label="CV F1 macro")
    ax.bar(x + w / 2, acc, width=w, label="Test accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Порівняння алгоритмів (combined features)")
    fig.tight_layout()
    fig.savefig(PLOTS / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_class_distribution() -> None:
    path = ROOT / "models" / "data_quality.json"
    if not path.exists():
        return
    q = json.loads(path.read_text(encoding="utf-8"))
    dist = q["intent_distribution"]
    labels = list(dist.keys())
    counts = [dist[l]["count"] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts, color="#22c55e")
    ax.set_title("Розподіл класів intent")
    ax.set_ylabel("Кількість")
    fig.tight_layout()
    fig.savefig(PLOTS / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_learning_curve() -> None:
    """Plot learning curve (train vs val F1 macro) and save to models/plots."""
    try:
        from sklearn.model_selection import learning_curve
        import numpy as _np
        from src.intent_model import load_intent_data, build_intent_pipeline
        from src.evaluation import resolve_data_path
    except Exception:
        return

    data_path = resolve_data_path(ROOT)
    texts, labels = load_intent_data(data_path)
    model_name, feature_set = load_config()
    pipe = build_intent_pipeline(model_name, feature_set)

    train_sizes, train_scores, val_scores = learning_curve(
        pipe,
        texts,
        labels,
        cv=5,
        scoring="f1_macro",
        train_sizes=_np.linspace(0.2, 1.0, 5),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_mean, label="train F1")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    ax.plot(train_sizes, val_mean, label="val F1")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
    ax.set_xlabel("Train set size")
    ax.set_ylabel("F1 macro")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "learning_curve.png", dpi=150)
    plt.close(fig)


def plot_roc_pr_curve() -> None:
    """Plot multiclass ROC and PR curves if model supports predict_proba."""
    try:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        from src.intent_model import load_intent_data, build_intent_pipeline
        from src.evaluation import resolve_data_path
        import numpy as _np
    except Exception:
        return

    data_path = resolve_data_path(ROOT)
    texts, labels = load_intent_data(data_path)
    model_name, feature_set = load_config()
    pipe = build_intent_pipeline(model_name, feature_set)

    try:
        y_score = pipe.predict_proba(texts)
    except Exception:
        # predict_proba not available
        return

    classes = sorted(set(labels))
    y_true = label_binarize(labels, classes=classes)

    # ROC
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.get_cmap("tab10")
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors(i % 10), label=f"{cls} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multiclass ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "roc_curve.png", dpi=150)
    plt.close(fig)

    # Precision-Recall
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        ap = average_precision_score(y_true[:, i], y_score[:, i])
        ax.plot(recall, precision, color=colors(i % 10), label=f"{cls} (AP={ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(PLOTS / "pr_curve.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix() -> None:
    model_name, feature_set = load_config()
    pipe = build_intent_pipeline(model_name, feature_set)
    holdout = evaluate_holdout(pipe, resolve_data_path(ROOT))

    cm = np.array(holdout["confusion_matrix"])
    labels = holdout["labels"]

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix ({model_name})")
    fig.tight_layout()
    fig.savefig(PLOTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def write_model_card() -> None:
    model_path = ROOT / "models" / "intent_model.joblib"
    metrics_path = ROOT / "models" / "metrics.json"
    tuning_path = ROOT / "models" / "tuning_results.json"
    compare_path = ROOT / "models" / "model_comparison.json"

    model_name, feature_set = load_config()
    card = {
        "task": "Intent classification (4 classes)",
        "input": "Текст повідомлення клієнта (UA/EN)",
        "model_type": model_name,
        "features": feature_set,
        "classes": ["create_order", "track_order", "cancel_order", "faq"],
    }

    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        rep = m.get("classification_report", {})
        card["metrics"] = {
            "accuracy": rep.get("accuracy"),
            "f1_macro": rep.get("macro avg", {}).get("f1-score"),
            "train_size": m.get("train_size"),
            "test_size": m.get("test_size"),
        }

    if tuning_path.exists():
        card["hyperparameters"] = json.loads(tuning_path.read_text(encoding="utf-8"))

    if compare_path.exists():
        card["baseline_comparison"] = json.loads(compare_path.read_text(encoding="utf-8"))[
            "recommended"
        ]

    if model_path.exists():
        size_kb = round(model_path.stat().st_size / 1024, 2)
        card["artifact_size_kb"] = size_kb

    out = ROOT / "models" / "model_card.json"
    out.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Model card: {out}")


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_comparison()
    plot_class_distribution()
    plot_learning_curve()
    plot_roc_pr_curve()
    plot_confusion_matrix()
    write_model_card()
    print(f"Plots: {PLOTS}")


if __name__ == "__main__":
    main()
