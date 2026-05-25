from __future__ import annotations

import csv
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "intents_merged.csv"
SEED_UA = ROOT / "data" / "intents_train.csv"
SEED_EN = ROOT / "data" / "seed" / "english_orders.csv"

CLINC_URL = "https://calmcode.io/static/data/clinc.csv"
SNIPS_BOOK_RESTAURANT = (
    "https://raw.githubusercontent.com/snipsco/nlu-benchmark/master/"
    "2017-06-custom-intent-engines/BOOK_RESTAURANT/train_BOOK_RESTAURANT.json"
)

# CLINC150: мапінг назв intent
CLINC_TO_INTENT: dict[str, str] = {
    "order": "create_order",
    "reorder": "create_order",
    "confirm": "create_order",
    "order_status": "track_order",
    "delivery_date": "track_order",
    "tracking": "track_order",
    "cancel": "cancel_order",
    "cancel_reservation": "cancel_order",
    "remove": "cancel_order",
    "calories": "faq",
    "cookbook": "faq",
    "expire_date": "faq",
    "food_last_exp": "faq",
    "gluten_free": "faq",
    "how_busy": "faq",
    "ingredients_list": "faq",
    "macros": "faq",
    "nutrition_info": "faq",
    "recipe": "faq",
    "restaurant_reviews": "faq",
    "restaurant_reservation": "faq",
    "restaurant_suggestions": "faq",
    "what_is_on_menu": "faq",
    "meal_suggestion": "faq",
    "accept_reservations": "faq",
    "ingredients": "faq",
    "menu": "faq",
    "outofscope": "faq",
}


def fetch_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "food-delivery-nlp/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def load_csv_pairs(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or row.get("utterance") or "").strip()
            intent = (row.get("intent") or row.get("label") or "").strip()
            if text and intent:
                rows.append((text, intent))
    return rows


def download_clinc(max_per_intent: int = 120) -> list[tuple[str, str]]:
    print(f"CLINC150: {CLINC_URL}")
    raw = fetch_text(CLINC_URL)
    lines = raw.strip().splitlines()
    reader = csv.DictReader(lines)
    buckets: dict[str, list[tuple[str, str]]] = {}
    for row in reader:
        label = (row.get("label") or row.get("intent") or "").strip()
        text = (row.get("text") or row.get("utterance") or "").strip()
        mapped = CLINC_TO_INTENT.get(label)
        if not mapped or not text:
            continue
        buckets.setdefault(mapped, []).append((text, mapped))
    out: list[tuple[str, str]] = []
    for intent, items in buckets.items():
        out.extend(items[:max_per_intent])
        print(f"  CLINC -> {intent}: {min(len(items), max_per_intent)}")
    return out


def download_snips() -> list[tuple[str, str]]:
    print(f"SNIPS BookRestaurant: {SNIPS_BOOK_RESTAURANT}")
    data = json.loads(fetch_text(SNIPS_BOOK_RESTAURANT))
    rows: list[tuple[str, str]] = []
    for item in data.get("data", []):
        text = (item.get("utterance") or "").strip()
        if text:
            rows.append((text, "create_order"))
    print(f"  SNIPS create_order: {len(rows)}")
    return rows


def dedupe(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for text, intent in rows:
        key = re.sub(r"\s+", " ", text.lower().strip())
        if key in seen:
            continue
        seen.add(key)
        out.append((text, intent))
    return out


def main() -> None:
    merged: list[tuple[str, str]] = []
    merged.extend(load_csv_pairs(SEED_UA))
    merged.extend(load_csv_pairs(SEED_EN))
    print(f"Seed UA+EN: {len(merged)}")

    try:
        merged.extend(download_clinc())
    except Exception as e:
        print(f"CLINC skip ({e})")

    try:
        merged.extend(download_snips())
    except Exception as e:
        print(f"SNIPS skip ({e})")

    merged = dedupe(merged)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "intent", "source"])
        for text, intent in merged:
            w.writerow([text, intent, "merged"])
    print(f"\nЗбережено {len(merged)} рядків -> {OUT}")
    print("Далі: python train.py")


if __name__ == "__main__":
    main()
