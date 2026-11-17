from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

SEED = 42

DEFAULT_OOS = [
    "Яка завтра погода у Чернівцях?", "Розкажи анекдот", "Як дістатися до вокзалу?",
    "Скільки буде двічі по двадцять п'ять?", "Переклади слово apple українською",
    "Порадь фільм на вечір", "Який курс долара сьогодні?", "Допоможи написати вірш",
    "What is the capital of France?", "Tell me a joke", "How do I reset my password?",
    "Play some music please", "Who won the football match yesterday?",
    "Book me a flight to Warsaw", "Що таке штучний інтелект?", "Запиши мене до лікаря",
    "Where can I buy a laptop?", "Як навчитися грати на гітарі?",
    "Скільки коштує квиток на потяг до Києва?", "Set an alarm for seven o'clock",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text).strip().lower()))


def build_pipeline() -> Pipeline:
    vec = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 max_features=60_000, sublinear_tf=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                 max_features=30_000, sublinear_tf=True,
                                 token_pattern=r"(?u)\b\w\w+\b")),
    ])
    base = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    return Pipeline([("tfidf", vec), ("clf", clf)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intents_merged.csv")
    ap.add_argument("--oos", default=None, help="файл із запитами поза доменом")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--min-coverage", type=float, default=0.90,
                    help="мінімальна частка запитів, що обробляються автоматично")
    ap.add_argument("--target-precision", type=float, default=0.98,
                    help="мінімальна точність серед автоматично оброблених запитів")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    texts = [normalize(t) for t in df["text"].astype(str)]
    labels = df["intent"].astype(str).tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)

    model = build_pipeline().fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)
    pred = model.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    correct = pred == np.asarray(y_te)

    oos = DEFAULT_OOS
    if args.oos:
        oos = [l.strip() for l in Path(args.oos).read_text(encoding="utf-8").splitlines() if l.strip()]
    oos_conf = model.predict_proba([normalize(t) for t in oos]).max(axis=1)

    thresholds = np.linspace(0.30, 0.99, 70)
    coverage, precision, oos_accepted = [], [], []
    for th in thresholds:
        acc_mask = conf >= th
        coverage.append(float(acc_mask.mean()))
        precision.append(float(correct[acc_mask].mean()) if acc_mask.any() else 1.0)
        oos_accepted.append(float((oos_conf >= th).mean()))

    chosen = None
    for th, cov, pr, oa in zip(thresholds, coverage, precision, oos_accepted):
        if pr >= args.target_precision and cov >= args.min_coverage:
            chosen = {"threshold": round(float(th), 3), "coverage": round(cov, 4),
                      "precision_on_accepted": round(pr, 4),
                      "oos_wrongly_accepted": round(oa, 4),
                      "oos_rejected": round(1 - oa, 4)}

    result = {
        "n_test": len(y_te), "n_oos": len(oos),
        "target_precision": args.target_precision,
        "min_coverage": args.min_coverage,
        "chosen": chosen,
        "mean_conf_in_domain": round(float(conf.mean()), 4),
        "mean_conf_oos": round(float(oos_conf.mean()), 4),
        "note": "Якщо chosen=None, обмеження недосяжні: знизьте цільову точність/покриття або розширте дані.",
    }
    json.dump(result, open(out / "threshold.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(coverage, precision, marker=".")
    ax[0].set_xlabel("Покриття (частка оброблена автоматично)")
    ax[0].set_ylabel("Точність серед оброблених")
    ax[0].set_title("Крива «ризик–покриття»")
    ax[0].grid(alpha=0.3)
    ax[1].hist(conf, bins=20, alpha=0.7, label="у домені")
    ax[1].hist(oos_conf, bins=20, alpha=0.7, label="поза доменом")
    if chosen:
        ax[1].axvline(chosen["threshold"], color="k", linestyle="--", label="поріг")
    ax[1].set_xlabel("Впевненість моделі")
    ax[1].set_ylabel("Кількість запитів")
    ax[1].set_title("Розподіл впевненості")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(out / "risk_coverage.png", dpi=200)


if __name__ == "__main__":
    main()
