from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, learning_curve
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from run_experiments import (SEED, bootstrap_ci, char_vec, combined_vec,
                             load_data, normalize_text, word_vec)

ALGOS = {
    "linear_svc": lambda: LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
    "logistic_regression": lambda: LogisticRegression(
        C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=24, class_weight="balanced_subsample",
        random_state=SEED, n_jobs=-1),
    "complement_nb": lambda: ComplementNB(alpha=0.1),
}
FEATS = {"char": char_vec, "word": word_vec, "combined": combined_vec}
FEAT_COLORS = {"char": "#1f77b4", "word": "#ff7f0e", "combined": "#2ca02c"}


def build(algo: str, feat: str) -> Pipeline:
    return Pipeline([("tfidf", FEATS[feat]()), ("clf", ALGOS[algo]())])


def split(texts, labels, groups):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, te = next(sgkf.split(texts, labels, groups))
    pick = lambda seq, idx: [seq[i] for i in idx]
    return (pick(texts, tr), pick(texts, te), pick(labels, tr), pick(labels, te),
            pick(groups, tr), "за групами шаблонів")


def acc_ci(y_true, y_pred, n=1000):
    rng = np.random.default_rng(SEED)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    s = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        s.append(accuracy_score(y_true[i], y_pred[i]))
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# 1. Розподіл класів
def plot_class_distribution(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    order = ["create_order", "track_order", "cancel_order", "faq"]
    for a, col, title in ((ax[0], "source", "за джерелом даних"), (ax[1], "lang", "за мовою")):
        if col in df.columns:
            tab = df.pivot_table(index="intent", columns=col, values="text", aggfunc="count",
                                 fill_value=0).reindex(order)
            tab.plot(kind="bar", stacked=True, ax=a, rot=0)
            a.legend(title=col, fontsize=8)
        else:
            df["intent"].value_counts().reindex(order).plot(kind="bar", ax=a, rot=0, color="#22c55e")
        a.set_title(f"Розподіл основних класів intent ({title})")
        a.set_xlabel("")
        a.set_ylabel("Кількість")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=200)
    plt.close()


# 2. Порівняння моделей (accuracy, F1) з довірчими інтервалами
def evaluate_all(X_tr, y_tr, X_te, y_te, ood) -> pd.DataFrame:
    rows = []
    for algo in ALGOS:
        for feat in FEATS:
            print(f"  {algo} / {feat}")
            m = build(algo, feat).fit(X_tr, y_tr)
            for split_name, (X, y) in (("test", (X_te, y_te)),) + ((("ood", ood),) if ood else ()):
                p = m.predict(X)
                f_lo, f_hi = bootstrap_ci(y, p)
                a_lo, a_hi = acc_ci(y, p)
                rows.append({
                    "split": split_name, "model": algo, "feature_set": feat,
                    "accuracy": accuracy_score(y, p), "acc_lo": a_lo, "acc_hi": a_hi,
                    "f1_macro": f1_score(y, p, average="macro", zero_division=0),
                    "f1_lo": f_lo, "f1_hi": f_hi, "n": len(y)})
    return pd.DataFrame(rows)


def plot_model_bars(res: pd.DataFrame, split_name: str, metric: str, lo: str, hi: str,
                    title: str, fname: Path) -> None:
    d = res[res["split"] == split_name]
    algos = list(ALGOS)
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for k, feat in enumerate(FEATS):
        sub = d[d["feature_set"] == feat].set_index("model").reindex(algos)
        x = np.arange(len(algos)) + (k - 1) * width
        err = [sub[metric] - sub[lo], sub[hi] - sub[metric]]
        ax.bar(x, sub[metric], width, yerr=err, capsize=3, label=feat, color=FEAT_COLORS[feat])
    ax.set_xticks(np.arange(len(algos)))
    ax.set_xticklabels(algos)
    low = max(0.0, np.floor((d[lo].min() - 0.05) * 20) / 20)
    ax.set_ylim(low, 1.02)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(title="набір ознак")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


# 3. Матриця помилок
def plot_cm(y_true, y_pred, labels, title: str, fname: Path) -> None:
    labels = [label for label in labels if label != "other"]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    included = np.isin(y_true, labels) & np.isin(y_pred, labels)
    y_true = y_true[included]
    y_pred = y_pred[included]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax[0], cmap="Blues",
                                                           colorbar=False, xticks_rotation=30)
    ax[0].set_title(f"{title}: кількість")
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ConfusionMatrixDisplay(norm, display_labels=labels).plot(ax=ax[1], cmap="Blues",
                                                             colorbar=False, xticks_rotation=30,
                                                             values_format=".2f")
    ax[1].set_title(f"{title}: частка по рядках (recall)")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


# 4. Крива навчання
def plot_learning_curve(X_tr, y_tr, g_tr, fname: Path) -> None:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    sizes, tr, va = learning_curve(
        build("linear_svc", "combined"), X_tr, y_tr, groups=g_tr, cv=cv,
        scoring="f1_macro", train_sizes=np.linspace(0.1, 1.0, 8), shuffle=True,
        random_state=SEED, n_jobs=1, error_score=np.nan)
    plt.figure(figsize=(8, 4.8))
    for vals, name, color in ((tr, "train F1", "#1f77b4"), (va, "val F1 (CV)", "#ff7f0e")):
        m, s = np.nanmean(vals, axis=1), np.nanstd(vals, axis=1)
        plt.plot(sizes, m, label=name, color=color, marker="o")
        plt.fill_between(sizes, m - s, m + s, color=color, alpha=0.15)
    plt.xlabel("Розмір навчальної вибірки")
    plt.ylabel("F1 macro")
    plt.title("Крива навчання (linear_svc, combined)")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    gap = float(np.nanmean(tr[-1]) - np.nanmean(va[-1]))
    print(f"  розрив train–val на повному обсязі: {gap:.3f}")


# 5. До / після розширення
def compare_before_after(orig_path: Path, new_path: Path, ood, out: Path) -> None:
    rows = []
    for tag, path in (("оригінал", orig_path), ("розширений", new_path)):
        texts, labels, groups = load_data(path)
        X_tr, X_te, y_tr, y_te, _, proto = split(texts, labels, groups)
        for algo in ("linear_svc", "logistic_regression"):
            own = build(algo, "combined").fit(X_tr, y_tr)
            own_f1 = f1_score(y_te, own.predict(X_te), average="macro", zero_division=0)
            full = build(algo, "combined").fit(texts, labels)
            p = full.predict(ood[0])
            rows.append({
                "dataset": tag, "n_rows": len(texts), "model": algo,
                "own_split_protocol": proto, "own_test_f1": round(own_f1, 4),
                "ood_f1_macro": round(f1_score(ood[1], p, average="macro", zero_division=0), 4),
                "ood_accuracy": round(accuracy_score(ood[1], p), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "before_after.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    models = ["linear_svc", "logistic_regression"]
    for k, tag in enumerate(("оригінал", "розширений")):
        vals = [df[(df.dataset == tag) & (df.model == m)]["ood_f1_macro"].iloc[0] for m in models]
        x = np.arange(len(models)) + (k - 0.5) * 0.35
        bars = ax.bar(x, vals, 0.35, label=f"{tag} ({int(df[df.dataset == tag].n_rows.iloc[0])} рядків)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("F1-macro на ручному тесті")
    ax.set_title("Вплив розширення датасету (один і той самий ручний тест)")
    ax.legend(loc="upper center", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "before_after_ood.png", dpi=200)
    plt.close()
    print(df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intents_merged.csv")
    ap.add_argument("--ood", default=None, help="ручний тест (CSV: text,intent)")
    ap.add_argument("--compare-with", default=None,
                    help="старий (оригінальний) датасет для порівняння «до/після»")
    ap.add_argument("--out", default="reports/figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.data)
    texts, labels, groups = load_data(Path(args.data))
    X_tr, X_te, y_tr, y_te, g_tr, proto = split(texts, labels, groups)
    print(f"Датасет: {len(texts)} рядків; поділ: {proto}; train={len(X_tr)}, test={len(X_te)}")

    ood = None
    if args.ood:
        o = pd.read_csv(args.ood)
        ood = ([normalize_text(t) for t in o["text"].astype(str)], o["intent"].astype(str).tolist())
        print(f"Ручний тест: {len(ood[0])} прикладів")

    print("1/5 розподіл класів")
    plot_class_distribution(df_raw, out)

    print("2/5 порівняння моделей")
    res = evaluate_all(X_tr, y_tr, X_te, y_te, ood)
    res.round(4).to_csv(out / "all_models_results.csv", index=False)
    plot_model_bars(res, "test", "accuracy", "acc_lo", "acc_hi",
                    "Порівняння моделей: accuracy на тесті", out / "all_models_test_accuracy.png")
    plot_model_bars(res, "test", "f1_macro", "f1_lo", "f1_hi",
                    "Порівняння моделей: F1-macro на тесті", out / "all_models_test_f1_macro.png")
    if ood:
        plot_model_bars(res, "ood", "f1_macro", "f1_lo", "f1_hi",
                        "Порівняння моделей: F1-macro на ручному тесті",
                        out / "all_models_ood_f1_macro.png")

    print("3/5 матриця помилок")
    main_model = build("linear_svc", "combined").fit(X_tr, y_tr)
    labels_sorted = sorted(set(labels))
    plot_cm(y_te, main_model.predict(X_te), labels_sorted, "linear_svc, тест", out / "confusion_matrix.png")
    if ood:
        plot_cm(ood[1], main_model.predict(ood[0]), labels_sorted, "linear_svc, ручний тест",
                out / "confusion_matrix_ood.png")

    print("4/5 крива навчання")
    plot_learning_curve(X_tr, y_tr, g_tr, out / "learning_curve.png")

    if args.compare_with and ood:
        print("5/5 порівняння до/після")
        compare_before_after(Path(args.compare_with), Path(args.data), ood, out)
    else:
        print("5/5 пропущено (потрібні --compare-with і --ood)")

    print(f"\nГотово. Графіки у {out}/")


if __name__ == "__main__":
    main()
