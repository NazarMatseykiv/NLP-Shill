from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import resolve_data_path
from src.intent_model import (
    build_intent_pipeline,
    calibrate_pipeline,
    load_intent_dataset,
)
from src.menu import MenuMatcher

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(resolve_data_path(ROOT)))
    ap.add_argument("--oos", default=None, help="файл із запитами поза доменом")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--min-coverage", type=float, default=0.75,
                    help="мінімальна частка запитів, що обробляються автоматично")
    ap.add_argument("--target-precision", type=float, default=0.98,
                    help="мінімальна точність серед автоматично оброблених запитів")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    texts, labels, groups = load_intent_dataset(Path(args.data))
    from sklearn.model_selection import StratifiedGroupKFold

    train_idx, test_idx = next(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            texts, labels, groups
        )
    )
    X_tr = [texts[i] for i in train_idx]
    X_te = [texts[i] for i in test_idx]
    y_tr = [labels[i] for i in train_idx]
    y_te = [labels[i] for i in test_idx]
    g_tr = [groups[i] for i in train_idx]

    config_path = ROOT / "models" / "best_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    tuning_path = ROOT / "models" / "tuning_results.json"
    if tuning_path.exists():
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        if (
            tuning.get("model_name") == config.get("model_name")
            and tuning.get("feature_set") == config.get("feature_set")
            and tuning.get("data_sha256")
            == hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
        ):
            model_params = tuning.get("best_params", {})
        else:
            model_params = {}
    else:
        model_params = {}
    pipeline = build_intent_pipeline(
        config.get("model_name", "logistic_regression"),
        config.get("feature_set", "combined"),
    )
    if model_params:
        pipeline.set_params(**model_params)
    model = calibrate_pipeline(
        pipeline,
        X_tr,
        y_tr,
        g_tr,
    )
    proba = model.predict_proba(X_te)
    pred = model.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    correct = (pred == np.asarray(y_te)) & (pred != "other")
    in_scope = pred != "other"

    oos = DEFAULT_OOS
    if args.oos:
        oos = [l.strip() for l in Path(args.oos).read_text(encoding="utf-8").splitlines() if l.strip()]
    menu = MenuMatcher(ROOT / "data" / "menu.json")
    from src.preprocess import normalize_text

    oos_proba = model.predict_proba([normalize_text(t, menu) for t in oos])
    oos_pred = model.classes_[oos_proba.argmax(axis=1)]
    oos_conf = oos_proba.max(axis=1)
    production_path = ROOT / "models" / "intent_model.joblib"
    production_model = joblib.load(production_path) if production_path.exists() else model
    production_proba = production_model.predict_proba(
        [normalize_text(t, menu) for t in oos]
    )
    production_pred = production_model.classes_[production_proba.argmax(axis=1)]
    production_conf = production_proba.max(axis=1)

    thresholds = np.linspace(0.30, 0.99, 70)
    coverage, precision = [], []
    validation_oos_accepted, production_oos_accepted, oos_accepted = [], [], []
    for th in thresholds:
        acc_mask = (conf >= th) & in_scope
        coverage.append(float(acc_mask.mean()))
        precision.append(float(correct[acc_mask].mean()) if acc_mask.any() else 1.0)
        validation_accepts = (oos_conf >= th) & (oos_pred != "other")
        production_accepts = (production_conf >= th) & (production_pred != "other")
        validation_oos_accepted.append(float(validation_accepts.mean()))
        production_oos_accepted.append(float(production_accepts.mean()))
        oos_accepted.append(float((validation_accepts | production_accepts).mean()))

    all_candidates = list(zip(thresholds, coverage, precision, oos_accepted))
    candidates = [
        row
        for row in all_candidates
        if row[1] >= args.min_coverage and row[1] > 0
    ]
    eligible = [row for row in candidates if row[2] >= args.target_precision]
    safe = [row for row in eligible if row[3] == 0]
    zero_ood = [row for row in all_candidates if row[1] > 0 and row[3] == 0]
    if safe:
        selected = safe[-1]
    elif zero_ood:
        selected = max(zero_ood, key=lambda row: (row[1], row[2], row[0]))
    elif eligible:
        selected = max(eligible, key=lambda row: (-row[3], row[1], row[0]))
    elif candidates:
        selected = max(candidates, key=lambda row: (row[2], -row[3], row[1], row[0]))
    else:
        candidates = [row for row in all_candidates if row[1] > 0]
        selected = (
            max(candidates, key=lambda row: (row[2], -row[3], row[1], row[0]))
            if candidates
            else (float(thresholds[0]), 0.0, 1.0, 1.0)
        )
    th, cov, pr, oa = selected
    constraints_met = cov >= args.min_coverage and pr >= args.target_precision
    chosen = {
        "threshold": round(float(th), 3),
        "coverage": round(cov, 4),
        "precision_on_accepted": round(pr, 4),
        "oos_wrongly_accepted": round(oa, 4),
        "oos_rejected": round(1 - oa, 4),
        "constraints_met": constraints_met,
        "oos_target_met": oa == 0,
        "validation_oos_wrongly_accepted": round(
            float(((oos_conf >= th) & (oos_pred != "other")).mean()), 4
        ),
    }

    result = {
        "n_test": len(y_te), "n_oos": len(oos),
        "target_precision": args.target_precision,
        "min_coverage": args.min_coverage,
        "chosen": chosen,
        "mean_conf_in_domain": round(float(conf.mean()), 4),
        "mean_conf_oos_validation": round(float(oos_conf.mean()), 4),
        "mean_conf_oos_production": round(float(production_conf.mean()), 4),
        "threshold_sweep": [
            {
                "threshold": round(float(threshold), 3),
                "coverage": round(covered, 4),
                "precision_on_accepted": round(precise, 4),
                "validation_oos_accepted": round(valid_ood, 4),
                "production_oos_accepted": round(prod_ood, 4),
            }
            for threshold, covered, precise, valid_ood, prod_ood in zip(
                thresholds,
                coverage,
                precision,
                validation_oos_accepted,
                production_oos_accepted,
            )
        ],
        "note": "Якщо constraints_met=false, обрано найкращий доступний компроміс; поріг потрібно переглянути після розширення даних.",
    }
    (out / "threshold.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(coverage, precision, marker=".")
    ax[0].set_xlabel("Покриття (частка оброблена автоматично)")
    ax[0].set_ylabel("Точність серед оброблених")
    ax[0].set_title("Крива «ризик–покриття»")
    ax[0].grid(alpha=0.3)
    ax[1].hist(conf, bins=20, alpha=0.7, label="у домені")
    ax[1].hist(production_conf, bins=20, alpha=0.7, label="поза доменом")
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
