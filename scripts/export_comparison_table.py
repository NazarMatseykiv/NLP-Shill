import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "models" / "model_comparison.json"
out = ROOT / "models" / "model_comparison_table.csv"

if not src.exists():
    raise SystemExit("Спочатку: python compare_models.py")

data = json.loads(src.read_text(encoding="utf-8"))
rows = data["by_feature_set"]["combined"]["ranking"]
df = pd.DataFrame(rows)
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"Збережено: {out}")
