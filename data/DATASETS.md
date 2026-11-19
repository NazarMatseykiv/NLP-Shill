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
- `data/intents_merged.csv` — корпус для навчання з групами шаблонів; він також
  містить 49 вручну сформульованих FAQ-прикладів про оплату, алергени,
  доставку, мінімальну суму, графік і повернення. Частина FAQ навмисно згадує
  страви та оформлення замовлення, а також додані приклади української
  транслітерації. Є також окремий клас `other` для запитів поза тематикою
  доставки їжі.

## Як зібрати merged-датасет

```powershell
# Локально (без мережі):
python scripts/build_merged_local.py

# Повний (CLINC + SNIPS + локальні):
python scripts/download_datasets.py
```

Результат: `data/intents_merged.csv`

Для оцінки слотів запускайте `python evaluate.py`: він рахує F1 на 100
контрольованих повідомленнях із наперед заданими стравами, кількістю та
адресами. Ця синтетична оцінка перевіряє якість на відомих форматах і не є
заміною розміченого реального трафіку.

Для порівняння TF-IDF, ембедінгів і гібридної моделі встановіть опційні
залежності та запустіть групову оцінку:

```powershell
python -m pip install -r requirements-embeddings.txt
python experiments/run_experiments.py --embeddings --cv 5 --out reports/embedding_benchmark
```
