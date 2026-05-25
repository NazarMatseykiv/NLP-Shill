## Рекомендовані публічні датасети

### 1. CLINC150 (основний для intent + FAQ)

- **Опис:** 150 класів намірів, англійська, 23 700 реплік.
- **Стаття:** Larson, E., et al. (2019). *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction.* EMNLP-IJCNLP.
- **Посилання:** https://calmcode.io/static/data/clinc.csv
- **Завантаження в проєкті:** `python scripts/download_datasets.py`

### 2. SNIPS (замовлення / ресторан)

- **Опис:** 7 інтентів; для курсової використовується `BookRestaurant` (~1800 реплік) → `create_order`.
- **Стаття:** Coucke, A., et al. (2018). *Snips Voice Platform.*
- **GitHub:** https://github.com/snipsco/nlu-benchmark

### 3. Власний корпус (UA + EN)

- `data/intents_train.csv` — українські фрази
- `data/seed/english_orders.csv` — англійські шаблони

## Як зібрати merged-датасет

```powershell
# Локально (без мережі):
python scripts/build_merged_local.py

# Повний (CLINC + SNIPS + локальні):
python scripts/download_datasets.py
```

Результат: `data/intents_merged.csv`
