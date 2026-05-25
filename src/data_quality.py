from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .intent_model import load_intent_data
from .preprocess import normalize_text


def analyze_dataset(csv_path: Path) -> dict:
    texts, labels = load_intent_data(csv_path)
    df = pd.DataFrame({"text": texts, "intent": labels})

    normalized = [normalize_text(t) for t in texts]
    keys = [re.sub(r"\s+", " ", t) for t in normalized]
    duplicates = len(keys) - len(set(keys))
    lengths = [len(t.split()) for t in normalized]

    intent_counts = Counter(labels)
    total = len(df)

    per_intent = {
        intent: {
            "count": count,
            "share": round(count / total, 4),
        }
        for intent, count in intent_counts.items()
    }

    imbalance_ratio = max(intent_counts.values()) / max(min(intent_counts.values()), 1)

    return {
        "path": str(csv_path),
        "total_samples": total,
        "unique_intents": len(intent_counts),
        "intent_distribution": per_intent,
        "class_imbalance_ratio": round(imbalance_ratio, 2),
        "duplicate_texts": duplicates,
        "completeness_empty_text": int((df["text"].str.strip() == "").sum()),
        "avg_tokens": round(sum(lengths) / len(lengths), 2),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "quality_notes": {
            "accuracy": "Тексти нормалізовані (lower, NFC); дублікати видаляються при merge",
            "completeness": "Порожні рядки відсутні" if df["text"].str.strip().ne("").all() else "Є порожні",
            "consistency": "Єдиний формат intent: create_order | track_order | cancel_order | faq",
            "timeliness": "Статичний корпус; оновлення — scripts/download_datasets.py",
        },
    }
