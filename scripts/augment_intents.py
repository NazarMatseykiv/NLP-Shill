import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "intents_train.csv"

STREETS = [
    "Шевченка",
    "Франка",
    "Хрещатик",
    "Саксаганського",
    "Бандери",
    "Грушевського",
    "Лесі Українки",
]

DISH_PHRASES = [
    "борщ",
    "два борщі",
    "піцу",
    "салат",
    "бургер",
    "вареники",
    "каву",
    "колу",
]

TEMPLATES_CREATE = [
    "Хочу замовити {dish} на вул. {street} {num}",
    "Доставте {dish} на вул. {street} {num}",
    "Замовляю {dish} на проспект {street} {num}",
    "Привезіть {dish} на вул. {street} {num} кв. {num}",
]

TEMPLATES_TRACK = [
    "Де моє замовлення?",
    "Статус замовлення",
    "Коли привезуть?",
    "Відстежити замовлення",
]

TEMPLATES_CANCEL = [
    "Скасуйте замовлення",
    "Відміна замовлення",
    "Не потрібно доставляти скасуйте",
]

TEMPLATES_FAQ = [
    "Скільки коштує доставка?",
    "Який час роботи?",
    "Які страви в меню?",
]


def load_existing() -> set[str]:
    with CSV.open(encoding="utf-8") as f:
        return {row["text"] for row in csv.DictReader(f)}


def append_rows(rows: list[tuple[str, str]]) -> int:
    existing = load_existing()
    new_rows = [(t, i) for t, i in rows if t not in existing]
    if not new_rows:
        return 0
    with CSV.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for text, intent in new_rows:
            w.writerow([text, intent])
    return len(new_rows)


def main() -> None:
    random.seed(42)
    generated: list[tuple[str, str]] = []
    for _ in range(40):
        t = random.choice(TEMPLATES_CREATE).format(
            dish=random.choice(DISH_PHRASES),
            street=random.choice(STREETS),
            num=random.randint(1, 40),
        )
        generated.append((t, "create_order"))
    for t in TEMPLATES_TRACK * 5:
        generated.append((t, "track_order"))
    for t in TEMPLATES_CANCEL * 5:
        generated.append((t, "cancel_order"))
    for t in TEMPLATES_FAQ * 5:
        generated.append((t, "faq"))
    n = append_rows(generated)
    print(f"Додано {n} нових речень у {CSV}")


if __name__ == "__main__":
    main()
