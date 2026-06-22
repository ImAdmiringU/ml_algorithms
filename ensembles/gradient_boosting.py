import numpy as np
import pandas as pd
from decision_tree import DecisionTreeRegressor

class BaseGradientBoosting:
    def __init__(self,
                 n_estimators: int,
                 learning_rate: float,
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 random_state: int | None):
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
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 1e-1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int | None = None):
        super().__init__(n_estimators,
                         learning_rate,
                         max_depth,
                         min_samples_split,
                         min_samples_leaf,
                         random_state)

    def _initial_prediction(self, y: np.array) -> None:
        '''
        Метод для инициализации исходного предикта

        Параметры
        ---------
        y : np.array
            Вектор таргет-значений датасета для обучения
        '''

        # Граница - эпсилон - для исключения логарифма от 0
        eps: float = 1e-15

        y_mean = np.clip(np.mean(y), a_min=eps, a_max=1-eps)

        self.initial_pred = y_mean / (1 - y_mean)

    def _sigmoid(self, pred: np.array) -> np.array:
        '''
        Возвращение вероятности отнесения к классу

        Параметры
        ---------
        pred : pd.Series
            Вектор текущих предиктов

        Возвращаемое значение
        ---------------------
        np.array
            Вероятности отнесения к
            классу 0/1
        '''
        return 1 / (1 + np.exp(-pred))

    def _compute_residuals(self, y, pred) -> np.array:
        '''
        Метод для расчета текущих значений residuals

        Параметры
        ---------
        y : np.array
            Вектор истинных меток класса
        pred : np.array
            Текущие значения предиктов

        Возвращаемое значение
        ---------------------
        res : np.array
            Вектор значений антиградиента функции ошибки
            предыдущего ансамбля
        '''

        res = y - self._sigmoid(pred)

        return res
    
    def predict_proba(self, X: pd.DataFrame) -> np.array:
        '''
        Метод для получения вероятностей предикта

        Параметры
        ---------
        X : pd.DataFrame
            Вектор наблюдений для которых выполняется предикт

        Возвращаемое значение
        ---------------------
        raw : np.array
            Вектор вероятностей отнесения к классу
            в соответствии с каждым наблюдением датасета
        '''

        raw = self._sigmoid(self._raw_predict(X=X))

        return raw

    def predict(self, X: pd.DataFrame) -> np.array:
        '''
        Метод для получения предиктов класса

        Параметры
        ---------
        X : pd.DataFrame
            Датасет с наблюдениями

        Возвращаемое значение
        ---------------------
        y_pred : np.array
            Вектор предиктов классов для
            соответствующих наблюдений
        '''

        y_pred = np.array([1 if value >= 0.5 else 0 for value in self.predict_proba(X=X)])

        return y_pred

class GradientBoostingRegressor(BaseGradientBoosting):
    pass
