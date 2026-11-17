from __future__ import annotations

import argparse
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

SEED = 42
CYR = re.compile(r"[а-яіїєґ]", re.IGNORECASE)

# Дані та препроцесинг
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text).strip().lower())
    return re.sub(r"\s+", " ", text)


def load_data(path: Path):
    """Повертає texts, labels, groups (groups=None, якщо колонки group немає)."""
    df = pd.read_csv(path)
    texts = [normalize_text(t) for t in df["text"].astype(str)]
    groups = df["group"].astype(str).tolist() if "group" in df.columns else None
    return texts, df["intent"].astype(str).tolist(), groups


def lang_of(text: str) -> str:
    return "UA" if CYR.search(text) else "EN"

# Ознаки
def char_vec() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           max_features=60_000, sublinear_tf=True)


def word_vec() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                           max_features=30_000, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w\w+\b")


def combined_vec() -> FeatureUnion:
    return FeatureUnion([("char", char_vec()), ("word", word_vec())])


# Базова модель на правилах
class RuleBaseline(BaseEstimator, ClassifierMixin):
    """Ключові слова UA/EN. Не навчається; потрібна як baseline «без ML»."""

    RULES = [
        ("cancel_order", r"скасу|скасув|відмін|анулю|не хочу замовл|cancel|abort"),
        ("track_order", r"де мо[єї]|статус|коли (привез|буде|достав)|кур'?єр|"
                        r"відстеж|where is|status|track|when will|eta"),
        ("create_order", r"замов|хочу|прин[еи]сі|додайте|\d+\s*(x|х|шт|порц)?\s*\w+.*"
                         r"(вул|вулиц|адрес|address|street)|order|i want|i'd like|deliver to"),
    ]

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        return self

    def predict(self, X):
        out = []
        for t in X:
            label = "faq"
            for name, pat in self.RULES:
                if re.search(pat, t):
                    label = name
                    break
            out.append(label)
        return np.array(out)


# Реєстр моделей
def build_models(with_embeddings: bool) -> dict:
    def pipe(vec, clf):
        return Pipeline([("tfidf", vec), ("clf", clf)])

    models = {
        "rules": RuleBaseline(),
        "svc_word": pipe(word_vec(), LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)),
        "svc_char": pipe(char_vec(), LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)),
        "svc_combined": pipe(combined_vec(), LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)),
        "logreg_combined": pipe(combined_vec(), LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED)),
        "cnb_combined": pipe(combined_vec(), ComplementNB(alpha=0.1)),
        "rf_combined": pipe(combined_vec(), RandomForestClassifier(
            n_estimators=200, max_depth=24, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1)),
    }
    if with_embeddings:
        try:
            models["embeddings_logreg"] = make_embedding_model()
        except Exception as exc:  # бібліотеки/мережі немає
            print(f"[skip] embeddings: {exc}")
    return models


def make_embedding_model() -> Pipeline:
    """Мультимовні ембедінги + LogReg (потрібен sentence-transformers)."""
    from sentence_transformers import SentenceTransformer
    from sklearn.base import TransformerMixin

    class Embedder(BaseEstimator, TransformerMixin):
        def __init__(self, name="paraphrase-multilingual-MiniLM-L12-v2"):
            self.name = name
            self._model = None

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            if self._model is None:
                self._model = SentenceTransformer(self.name)
            return self._model.encode(list(X), show_progress_bar=False)

    return Pipeline([("emb", Embedder()),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])


# Метрики: bootstrap CI, McNemar, затримка
def bootstrap_ci(y_true, y_pred, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def mcnemar_p(y_true, pred_a, pred_b) -> float:
    """Точний тест Макнемара (двобічний) для двох класифікаторів."""
    y_true = np.asarray(y_true)
    a_ok, b_ok = np.asarray(pred_a) == y_true, np.asarray(pred_b) == y_true
    b01 = int(np.sum(a_ok & ~b_ok))
    b10 = int(np.sum(~a_ok & b_ok))
    if b01 + b10 == 0:
        return 1.0
    return float(binomtest(b01, b01 + b10, 0.5).pvalue)


def measure_latency(model, texts, runs=300):
    """Затримка обробки ОДНОГО повідомлення (препроцесинг+ознаки+предикт), мс."""
    sample = [texts[i % len(texts)] for i in range(runs)]
    model.predict([sample[0]])  # прогрів
    times = []
    for t in sample:
        t0 = time.perf_counter()
        model.predict([normalize_text(t)])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


# Шум: символьні помилки набору
def add_noise(text: str, rate: float, rng: random.Random) -> str:
    chars, out, i = list(text), [], 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            op = rng.choice(["swap", "drop", "dup"])
            if op == "swap" and i + 1 < len(chars):
                out.extend([chars[i + 1], c])
                i += 2
                continue
            if op == "dup":
                out.extend([c, c])
            # drop: символ пропускаємо
        else:
            out.append(c)
        i += 1
    return "".join(out)


# Основний цикл
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intents_merged.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--embeddings", action="store_true",
                    help="додати baseline на ембедінгах (потрібен sentence-transformers)")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--ood", default=None,
                    help="окремий тест, написаний вручну (CSV: text,intent)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    texts, labels, groups = load_data(Path(args.data))
    if groups is not None:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        tr_idx, te_idx = next(sgkf.split(texts, labels, groups))
        X_tr = [texts[i] for i in tr_idx]; y_tr = [labels[i] for i in tr_idx]
        g_tr = [groups[i] for i in tr_idx]
        X_te = [texts[i] for i in te_idx]; y_te = [labels[i] for i in te_idx]
        print("Розбиття: за групами шаблонів (StratifiedGroupKFold)")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
        g_tr = None
        print("Розбиття: випадкове стратифіковане (колонки group немає)")
    print(f"train={len(X_tr)} test={len(X_te)} classes={sorted(set(labels))}")

    ood = None
    if args.ood:
        odf = pd.read_csv(args.ood)
        ood = ([normalize_text(t) for t in odf["text"].astype(str)],
               odf["intent"].astype(str).tolist())
        print(f"OOD-тест (ручний): {len(ood[0])} прикладів")

    models = build_models(args.embeddings)
    if g_tr is not None:
        cv = StratifiedGroupKFold(n_splits=args.cv, shuffle=True, random_state=SEED)
    else:
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED)

    rows, preds = [], {}
    fitted = {}
    for name, model in models.items():
        print(f"== {name}")
        m = clone(model)
        # CV тільки на train
        cv_f1 = cross_val_score(m, X_tr, y_tr, groups=g_tr, cv=cv, scoring="f1_macro", n_jobs=1)
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0
        y_pred = m.predict(X_te)
        lo, hi = bootstrap_ci(y_te, y_pred)
        lat_med, lat_p95 = measure_latency(m, X_te)
        tmp = out / f"_tmp_{name}.joblib"
        joblib.dump(m, tmp)
        size_kb = tmp.stat().st_size / 1024
        tmp.unlink()
        ood_acc = ood_f1 = None
        if ood is not None:
            ood_pred = m.predict(ood[0])
            ood_acc = round(accuracy_score(ood[1], ood_pred), 4)
            ood_f1 = round(f1_score(ood[1], ood_pred, average="macro", zero_division=0), 4)
        rows.append({
            "model": name,
            "cv_f1_macro_mean": round(float(cv_f1.mean()), 4),
            "cv_f1_macro_std": round(float(cv_f1.std()), 4),
            "test_accuracy": round(accuracy_score(y_te, y_pred), 4),
            "test_f1_macro": round(f1_score(y_te, y_pred, average="macro", zero_division=0), 4),
            "f1_ci95_low": round(lo, 4),
            "f1_ci95_high": round(hi, 4),
            "ood_accuracy": ood_acc,
            "ood_f1_macro": ood_f1,
            "train_time_s": round(train_time, 3),
            "latency_ms_median": round(lat_med, 3),
            "latency_ms_p95": round(lat_p95, 3),
            "model_size_kb": round(size_kb, 1),
        })
        preds[name], fitted[name] = np.asarray(y_pred), m

    df = pd.DataFrame(rows).sort_values(
        ["cv_f1_macro_mean", "latency_ms_median"], ascending=[False, True])
    df.to_csv(out / "model_comparison.csv", index=False)
    print(df.to_string(index=False))

    best = df.iloc[0]["model"]
    mc = {n: mcnemar_p(y_te, preds[best], preds[n]) for n in preds if n != best}
    json.dump({"best": best, "mcnemar_p_vs_best": mc},
              open(out / "mcnemar.json", "w"), ensure_ascii=False, indent=2)

    lang_rows = []
    langs = np.array([lang_of(t) for t in X_te])
    for name, p in preds.items():
        for lg in ("UA", "EN"):
            mask = langs == lg
            if mask.sum() >= 5:
                lang_rows.append({
                    "model": name, "lang": lg, "n": int(mask.sum()),
                    "f1_macro": round(f1_score(np.asarray(y_te)[mask], p[mask],
                                               average="macro", zero_division=0), 4)})
    pd.DataFrame(lang_rows).to_csv(out / "per_language.csv", index=False)

    rates = [0.0, 0.05, 0.10, 0.20, 0.30]
    noise = {n: [] for n in preds}
    for r in rates:
        rng = random.Random(SEED)
        noisy = [add_noise(t, r, rng) for t in X_te]
        for name, m in fitted.items():
            noise[name].append(f1_score(y_te, m.predict(noisy), average="macro", zero_division=0))
    plt.figure(figsize=(7, 4.5))
    for name, vals in noise.items():
        plt.plot([r * 100 for r in rates], vals, marker="o", label=name)
    plt.xlabel("Рівень шуму, % символів")
    plt.ylabel("F1-macro")
    plt.title("Стійкість до друкарських помилок")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "noise_robustness.png", dpi=200)
    plt.close()
    json.dump({"rates": rates, "f1_macro": noise},
              open(out / "noise_robustness.json", "w"), indent=2)

    fracs = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    curve = {n: [] for n in models if n != "rules"}
    for f in fracs:
        for name in curve:
            vals = []
            for rep in range(3):
                if f < 1.0:
                    Xs, _, ys, _ = train_test_split(
                        X_tr, y_tr, train_size=f, random_state=SEED + rep, stratify=y_tr)
                else:
                    Xs, ys = X_tr, y_tr
                try:
                    mm = clone(models[name]).fit(Xs, ys)
                    vals.append(f1_score(y_te, mm.predict(X_te), average="macro", zero_division=0))
                except ValueError:
                    vals.append(np.nan)
            curve[name].append(float(np.nanmean(vals)))
    plt.figure(figsize=(7, 4.5))
    for name, vals in curve.items():
        plt.plot([int(f * len(X_tr)) for f in fracs], vals, marker="o", label=name)
    plt.xlabel("Розмір навчальної вибірки")
    plt.ylabel("F1-macro на тесті")
    plt.title("Залежність якості від обсягу даних")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "learning_curve.png", dpi=200)
    plt.close()
    json.dump({"train_sizes": [int(f * len(X_tr)) for f in fracs], "f1_macro": curve},
              open(out / "learning_curve.json", "w"), indent=2)

    print(f"\nГотово. Результати у {out}/")
    print(f"Найкраща модель за CV F1 на train: {best}")


if __name__ == "__main__":
    main()
