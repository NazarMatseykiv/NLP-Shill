import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "intents_merged.csv"
SOURCES = [
    ROOT / "data" / "intents_train.csv",
    ROOT / "data" / "seed" / "english_orders.csv",
]


def main() -> None:
    seen: set[str] = set()
    rows: list[list[str]] = []
    for path in SOURCES:
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                text = row["text"].strip()
                intent = row["intent"].strip()
                key = re.sub(r"\s+", " ", text.lower())
                if key not in seen:
                    seen.add(key)
                    rows.append([text, intent, path.name])
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "intent", "source"])
        w.writerows(rows)
    print(f"Локально зібрано {len(rows)} рядків -> {OUT}")


if __name__ == "__main__":
    main()
