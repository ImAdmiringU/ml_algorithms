# 🤖 ML Algorithms from Scratch

**NumPy only. No shortcuts.**

[![Built from Scratch](https://img.shields.io/badge/built-from%20scratch-blue)]()

---

## 📁 Структура проекта

```text
ml_algorithms/
├── decision_tree.py          # CART (классификация + регрессия)
├── knn.py                    # KNN (классификация + регрессия)
├── linear_models.py          # LogisticRegression, LinearRegression
├── preprocessing.py          # StandardScaler, MinMaxScaler
├── base.py                   # Вспомогательные функции (entropy, mse, gain...)
│
├── ensembles/                # Ансамблевые методы
│   └── random_forest.py      # Random Forest (классификация + регрессия)
│
└── cpp_extensions/           # C++ утилиты (опционально, для ускорения)
    ├── cpp_utils.cpp         # Реализация
    ├── cpp_utils.h           # Заголовки
    └── wrapper.cpp           # pybind11 обёртка для Python
```

---

## ✅ Статус реализации

| Категория | Алгоритмы |
|-----------|-----------|
| 📊 Линейные модели | ✅ LogisticRegression, ✅ LinearRegression |
| 🌳 Деревья | ✅ DecisionTreeClassifier, ✅ DecisionTreeRegressor |
| 👥 Ленивые методы | ✅ KNNClassifier, ✅ KNNRegressor |
| 🔧 Предобработка | ✅ StandardScaler, ✅ MinMaxScaler |
| ⏳ Ансамбли | ⏳ Random Forest |
| ⏳ Кластеризация | ⏳ k-Means |
| ⏳ Снижение размерности | ⏳ PCA, t-SNE |


---

## ⚡ Быстрый пример

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from linear_models import LogisticRegression

# Генерация синтетических данных
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование в pandas (ожидают все модели)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

# Обучение
model = LogisticRegression(regularization='l2', C=1.0)
model.fit(X_train, y_train)

# Предсказание и оценка
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
```

---

## 🧠 Важное замечание
Все модели ожидают на вход pd.DataFrame для признаков и pd.Series для целевой переменной.
Предобработка данных (масштабирование, кодирование категорий, обработка пропусков) — ответственность пользователя.
Внутри модели работают с NumPy для производительности.

---

## 📦 Зависимости

### Обязательные (Python)
- `NumPy`
- `Pandas`

### Для запуска тестов и примеров
- `Scikit-learn` (только для генерации данных)

### Для сборки C++ расширения (опционально)
- Компилятор C++11 (GCC 4.8+, Clang 3.3+, MSVC 2019+)
- `pybind11` — установка: `pip install pybind11`

---

## 📚 Источники
Реализации написаны для изучения внутреннего устройства алгоритмов. Вдохновлено курсом mlcourse.ai и документацией scikit-learn.
