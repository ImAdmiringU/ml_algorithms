import numpy as np
import pandas as pd
from decision_tree import DecisionTreeRegressor

class BaseGradientBoosting:
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 1e-1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int | None = None):
        '''
        Инициализация гиперпараметров
        '''
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.random_state: int | None = random_state

        '''
        Атрибуты модели
        '''
        self.trees: list = list()
        self.initial_pred: float = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        '''
        Метод для обучения модели

        Параметры
        ---------
        X : pd.DataFrame
            Датасет с векторами наблюдений
        y : pd.Series
            Вектор с таргет значениями
        '''

        # Создаем инициализацию предсказаний на 0 итерации
        self._initial_prediction(y=y.values)

        current_pred = np.ones(len(y)) * self.initial_pred

        # Линейное обучение ансамбля моделей на residuals
        for i in range(self.n_estimators):
            # Получение текущих residuals
            residuals = self._compute_residuals(y=y, pred=current_pred)

            # Модель дерева i-й итерации
            temp_tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                              min_samples_split=self.min_samples_split,
                                              min_samples_leaf=self.min_samples_leaf)
            
            # Обучение модели дерева i-й итерации
            temp_tree.fit(X=X, y=pd.Series(residuals))

            # Добавляем модель в общий список
            self.trees.append(temp_tree)

            # Обновление общего предикта
            current_pred += self.learning_rate * temp_tree.predict(X=X)

    def _raw_predict(self, X: pd.DataFrame) -> np.array:
        '''
        Метод для предикта сырых значений

        Параметры
        ---------
        X : pd.DataFrame
            Датафрейм с векторами наблюдений

        Возвращаемое значение
        ---------------------
        res : np.array
            Вектор предиктов
        '''

        # Инициализация начального предикта
        res = np.ones(len(X)) * self.initial_pred

        # Итерационно линейно проходим по моделям
        for i in range(self.n_estimators):
            res += self.learning_rate * self.trees[i].predict(X=X)

        return res

    def _initial_prediction(self, y: np.array) -> None:
        raise NotImplementedError
    
    def _compute_residuals(self, y: np.array, pred: np.array) -> np.array:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.array:
        raise NotImplementedError


class GradientBoostingClassifier(BaseGradientBoosting):
    pass


class GradientBoostingRegressor(BaseGradientBoosting):
    pass
