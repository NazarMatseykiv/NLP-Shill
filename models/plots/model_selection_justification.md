# Обґрунтування вибору моделей-кандидатів для класифікації текстових замовлень

Для задачі класифікації текстових замовлень було протестовано декілька моделей: LinearSVC, Logistic Regression, Random Forest, Complement Naive Bayes. Оцінювання проводилося на різних типах ознак (char, word, combined).

**Результати**:
- Найкращі результати за макро-F1 та accuracy показали LinearSVC та Logistic Regression на комбінованих ознаках (див. графіки `combined_test_accuracy.png`, `combined_test_f1_macro.png`).
- Random Forest та Naive Bayes поступаються за якістю, особливо на word- та combined-ознаках.
- LinearSVC стабільно лідирує на всіх типах ознак, має найкращий баланс точності та швидкості.

**Висновок**:
LinearSVC обрано як основну модель для класифікації, оскільки вона демонструє найвищу якість на тестових даних та стабільність на різних ознаках. Logistic Regression також розглядається як сильний кандидат для ансамблю чи резервної моделі.

Детальні порівняльні графіки та матриця помилок збережені у папці `models/plots`:
- `combined_test_accuracy.png`, `combined_test_f1_macro.png` — порівняння моделей на комбінованих ознаках
- `all_models_test_accuracy.png`, `all_models_test_f1_macro.png` — загальне порівняння
- `confusion_matrix_best_model.png` — матриця помилок для LinearSVC

