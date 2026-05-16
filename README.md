# 🤖 ML Algorithms from Scratch

**NumPy only. No shortcuts.**

[![Built from Scratch](https://img.shields.io/badge/built-from%20scratch-blue)]()

---

## 📁 Структура проекта

```text
ml_algorithms/
├── decision_tree.py      # CART (классификация + регрессия)
├── knn.py                # KNN (классификация + регрессия)
├── linear_models.py      # LogisticRegression, LinearRegression
├── preprocessing.py      # StandardScaler, MinMaxScaler
└── base.py               # Вспомогательные функции (entropy, mse, gain...)
```

---

## ✅ Статус реализации

| Категория | Алгоритмы |
|-----------|-----------|
| 📊 Линейные модели | ✅ LinearRegression, ✅ LogisticRegression |
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
Внутри модели работают с numpy для производительности.

---

## 📦 Зависимости
* NumPy
* Pandas
* Scikit-learn (только для тестов и генерации данных)

---

## 📚 Источники
Реализации написаны для изучения внутреннего устройства алгоритмов. Вдохновлено курсом mlcourse.ai и документацией scikit-learn.
