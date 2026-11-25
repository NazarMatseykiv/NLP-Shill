import json
from pathlib import Path

import pandas as pd
from src.data_quality import analyze_dataset
from src.evaluation import resolve_data_path
from src.menu import MenuMatcher
from src.preprocess import normalize_text
from src.slots import SlotExtractor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import StratifiedGroupKFold

from src.intent_model import build_intent_pipeline, load_intent_dataset

ROOT = Path(__file__).resolve().parent

SLOT_TEST_CASES = [
    {
        "text": "на Руську 22 кв 9",
        "address_contains": "Руську 22",
    },
    {
        "text": "Два борщі та салат на вул. Шевченка 12",
        "expected_dishes": ["Borscht / Борщ", "Caesar Salad"],
        "address_contains": "Шевченка",
    },
    {
        "text": "Two burgers and fries delivered to 12 Main Street pay by card, leave at the door",
        "expected_dishes": ["Burger", "French Fries"],
        "address_contains": "Main Street",
        "payment": "card",
        "comment_contains": "leave at the door",
    },
    {
        "text": "На вул. Шевченка 12 хочу піцу пепероні та four cheese pizza, карткою",
        "expected_dishes": [
            "Pizza Pepperoni / Піца Пепероні",
            "Pizza Quattro Formaggi / Піца 4 сири",
        ],
        "exact_dishes": [
            "Pizza Pepperoni / Піца Пепероні",
            "Pizza Quattro Formaggi / Піца 4 сири",
        ],
        "address_contains": "Шевченка",
        "payment": "card",
    },
]

SLOT_DISH_CASES = [
    ("борщ", "Borscht / Борщ"),
    ("varenyky", "Varenyky / Вареники"),
    ("salad", "Caesar Salad"),
    ("burger", "Burger"),
    ("fries", "French Fries"),
    ("coffee", "Coffee"),
    ("sushi", "Sushi Roll"),
    ("tacos", "Tacos"),
    ("carbonara", "Carbonara / Паста Карбонара"),
]

SLOT_TEMPLATES = [
    ("Хочу {qty} {dish} на вул. Шевченка {number}", "вул. Шевченка"),
    ("Please bring {qty} {dish} to {number} Main Street", "Main Street"),
    ("На вул. Франка {number}, будь ласка, {qty} {dish}", "вул. Франка"),
    ("Order {qty} {dish} to {number} Oak Road", "Oak Road"),
    ("Привезіть {qty} {dish}, адреса: проспект Перемоги {number}", "проспект Перемоги"),
    ("Could you deliver {qty} {dish} to {number} Park Avenue?", "Park Avenue"),
    ("Для доставки на Хрещатик {number} додайте {qty} {dish}", "Хрещатик"),
    ("Send {qty} {dish} to {number} River Street", "River Street"),
    ("Доставте {qty} {dish} на вул. Соборна {number}", "вул. Соборна"),
    ("At {number} Baker Street, I'd like {qty} {dish}", "Baker Street"),
]

QUANTITY_FORMS = {
    1: ["1", "one", "один"],
    2: ["2", "two", "два"],
    3: ["3", "three", "три"],
    4: ["4", "four", "чотири"],
    5: ["5", "five", "п'ять"],
}


def build_slot_eval_cases() -> list[dict]:
    cases = []
    for template_idx, (template, street) in enumerate(SLOT_TEMPLATES):
        for dish_idx, (dish, dish_name) in enumerate(SLOT_DISH_CASES):
            quantity = dish_idx % len(QUANTITY_FORMS) + 1
            number = 10 + template_idx * len(SLOT_DISH_CASES) + dish_idx
            text = template.format(
                qty=QUANTITY_FORMS[quantity][template_idx % 3],
                dish=dish,
                number=number,
            )
            if template_idx in (1, 3, 5, 7, 9):
                address = f"{number} {street}"
            else:
                address = f"{street} {number}"
            cases.append({
                "text": text,
                "dish_name": dish_name,
                "quantity": quantity,
                "address": address,
            })
    return cases


def slot_f1(cases: list[dict], predictions: list[dict]) -> dict:
    scores = {}
    for slot_name, expected_key, predicted_key in (
        ("dishes", "dish_name", "dishes"),
        ("quantities", "quantity", "dishes"),
        ("addresses", "address", "address"),
    ):
        tp = fp = fn = 0
        for index, (case, result) in enumerate(zip(cases, predictions)):
            if slot_name == "dishes":
                expected = {(index, case[expected_key])}
                predicted = {
                    (index, item["dish_name"]) for item in result[predicted_key]
                }
            elif slot_name == "quantities":
                expected = {(index, case["dish_name"], case[expected_key])}
                predicted = {
                    (index, item["dish_name"], item["quantity"])
                    for item in result[predicted_key]
                }
            else:
                expected = {(index, case[expected_key].casefold())}
                actual = result[predicted_key]
                predicted = (
                    {(index, actual.casefold())} if actual else set()
                )
            tp += len(expected & predicted)
            fp += len(predicted - expected)
            fn += len(expected - predicted)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores[slot_name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall else 0.0,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
    return scores


def evaluate_intent_cv(model_name: str, feature_set: str) -> dict:
    data = resolve_data_path(ROOT)
    texts, labels, groups = load_intent_dataset(data)
    pipe = build_intent_pipeline(model_name, feature_set)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(texts, labels, groups))
    scores = cross_val_score(
        pipe, texts, labels, cv=splits, scoring="f1_macro", n_jobs=-1
    )
    return {
        "model": model_name,
        "feature_set": feature_set,
        "cv_folds": 5,
        "f1_macro_mean": float(scores.mean()),
        "f1_macro_std": float(scores.std()),
        "f1_macro_scores": [float(s) for s in scores],
    }


def evaluate_entity_masking_cv(model_name: str, feature_set: str) -> dict:
    data = resolve_data_path(ROOT)
    df = pd.read_csv(data)
    masked_texts, labels, groups = load_intent_dataset(data)
    unmasked_texts = [
        normalize_text(text, mask_entities=False) for text in df["text"].astype(str)
    ]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(masked_texts, labels, groups))
    scores = {}
    for name, texts in (
        ("masked", masked_texts),
        ("unmasked", unmasked_texts),
    ):
        model = build_intent_pipeline(model_name, feature_set)
        values = cross_val_score(
            model, texts, labels, cv=splits, scoring="f1_macro", n_jobs=-1
        )
        scores[name] = [float(value) for value in values]
    return {
        "cv_folds": len(splits),
        "masked_f1_macro": scores["masked"],
        "unmasked_f1_macro": scores["unmasked"],
        "masked_mean": sum(scores["masked"]) / len(scores["masked"]),
        "unmasked_mean": sum(scores["unmasked"]) / len(scores["unmasked"]),
        "mean_delta": (
            sum(scores["masked"]) - sum(scores["unmasked"])
        ) / len(scores["masked"]),
    }


def evaluate_transliteration_cv(model_name: str, feature_set: str) -> dict:
    data = resolve_data_path(ROOT)
    df = pd.read_csv(data)
    texts, labels, groups = load_intent_dataset(data)
    if "lang" not in df.columns:
        return {"support": 0}
    translit_mask = df["lang"].astype(str).eq("TRANSLIT").to_numpy()
    if not translit_mask.any():
        return {"support": 0}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(texts, labels, groups))
    predictions = cross_val_predict(
        build_intent_pipeline(model_name, feature_set),
        texts,
        labels,
        cv=splits,
        n_jobs=-1,
    )
    y_true = [label for label, keep in zip(labels, translit_mask) if keep]
    y_pred = [label for label, keep in zip(predictions, translit_mask) if keep]
    return {
        "support": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_slots() -> dict:
    menu = MenuMatcher(ROOT / "data" / "menu.json")
    extractor = SlotExtractor(menu)
    labeled_cases = build_slot_eval_cases()
    labeled_predictions = [
        extractor.extract(case["text"]).to_dict() for case in labeled_cases
    ]
    passed = 0
    details = []
    for case in SLOT_TEST_CASES:
        slots = extractor.extract(case["text"]).to_dict()
        dish_names = [d["dish_name"] for d in slots["dishes"]]
        ok = all(name in dish_names for name in case.get("expected_dishes", []))
        if case.get("exact_dishes"):
            ok = ok and set(dish_names) == set(case["exact_dishes"])
        quantities = {dish["dish_name"]: dish["quantity"] for dish in slots["dishes"]}
        ok = ok and all(
            quantities.get(name) == quantity
            for name, quantity in case.get("expected_quantities", {}).items()
        )
        if case.get("address_contains"):
            address = (slots["address"] or "").casefold()
            expected_address = case["address_contains"].casefold()
            ok = ok and expected_address in address
        if case.get("payment"):
            ok = ok and slots["payment"] == case["payment"]
        if case.get("comment_contains"):
            comment = (slots["comment"] or "").casefold()
            expected_comment = case["comment_contains"].casefold()
            ok = ok and expected_comment in comment
        passed += int(ok)
        details.append({"text": case["text"], "ok": ok, "slots": slots})
    return {
        "slot_tests_total": len(SLOT_TEST_CASES),
        "slot_tests_passed": passed,
        "slot_accuracy": passed / len(SLOT_TEST_CASES),
        "slot_annotated_messages": len(labeled_cases),
        "slot_f1": slot_f1(labeled_cases, labeled_predictions),
        "details": details,
    }


def main() -> None:
    data = resolve_data_path(ROOT)
    quality = analyze_dataset(data)
    print("Data quality:", json.dumps(quality["intent_distribution"], ensure_ascii=False))

    cfg_path = ROOT / "models" / "best_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        model_name, feature_set = cfg["model_name"], cfg["feature_set"]
    else:
        model_name, feature_set = "logistic_regression", "combined"

    print(f"\nCV for selected model ({model_name}, {feature_set})...")
    intent_cv = evaluate_intent_cv(model_name, feature_set)
    print(json.dumps(intent_cv, ensure_ascii=False, indent=2))

    print("\nEntity masking ablation (same grouped folds)...")
    masking_cv = evaluate_entity_masking_cv(model_name, feature_set)
    print(json.dumps(masking_cv, ensure_ascii=False, indent=2))

    print("\nTransliteration grouped-CV subset...")
    transliteration_cv = evaluate_transliteration_cv(model_name, feature_set)
    print(json.dumps(transliteration_cv, ensure_ascii=False, indent=2))

    print("\nSlot extraction...")
    slot_eval = evaluate_slots()
    print(json.dumps(slot_eval, ensure_ascii=False, indent=2))

    result = {
        "data_quality": quality,
        "intent_cv": intent_cv,
        "entity_masking_cv": masking_cv,
        "transliteration_cv": transliteration_cv,
        "slots": slot_eval,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    out = ROOT / "models" / "evaluation_full.json"
    out.write_text(serialized, encoding="utf-8")
    report = ROOT / "reports" / "evaluation_full.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(serialized, encoding="utf-8")
    print(
        f"\nSaved evaluation: {out}\n"
        f"Saved report: {report}"
    )


if __name__ == "__main__":
    main()
