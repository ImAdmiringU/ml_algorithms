import base
import numpy as np
import pandas as pd

class BaseLinearModel:
    def __init__(self,
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 tolerance: float = 0.0001,
                 regularization: str | None = None,
                 C: float = 1.0):
        '''
        Гиперпараметры градиентного спуска
        '''
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations
        self.tolerance: float = tolerance

        '''
        Параметры регуляризации
        '''
        self.regularization: str | None = regularization
        self.C: float = C

        '''
        Атрибуты модели
        '''
        self.weights: np.array = None
        self.bias: float = 0.0
        self.loss_history: list = list()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        '''
        Метод для итерационного обучения модели

        Параметры
        ---------
        X : pd.DataFrame
            Обработанный датафрейм c векторами
            экземпляров объектов для обучения
        y : pd.Series
            Таргет. Целевое значение
            для каждого вектора в X
        '''

        # Инициализируем веса
        self.weights = np.zeros(len(X.columns))

        for i in range(self.n_iterations):
            # Предварительный вектор предиктов
            temp_pred = self._linear_combination(X=X)
            
            # Рассчитываем loss
            temp_loss = self._compute_loss(y_true=y.values, y_pred=temp_pred.values)

            if self.loss_history:
                # Разница в loss на текущей итерации
                difference = np.abs(temp_loss - self.loss_history[-1])

                if difference < self.tolerance:
                    break
            
            self.loss_history.append(temp_loss)

            # Расчет градиента
            grad_w, grad_b = self._compute_gradient(X=X,
                                                    y_true=y,
                                                    y_pred=temp_pred)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def _linear_combination(self, X: pd.DataFrame) -> pd.Series:
        '''
        Предварительное получение предикта путем расчета
        линейной комбинации каждого наблюдения и
        текущих значений весов + bias

        Параметры
        ---------

        X : pd.DataFrame
            Датасет с векторами наблюдений

        Возвращаемое значение
        ---------------------

        y_pred : pd.Series
            Вектор предиктов для наблюдений
        '''
        y_pred = X @ self.weights + self.bias

        return y_pred

    def _compute_loss(self, y_true: np.array, y_pred: np.array) -> float:
        raise NotImplementedError()

    def _compute_gradient(self, X: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series) -> float:
        raise NotImplementedError()

    def predict(self, X: pd.DataFrame) -> np.array:
        raise NotImplementedError()


class LogisticRegression(BaseLinearModel):
    def __init__(self,
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 tolerance: float = 0.0001,
                 regularization: str | None = None,
                 C: float = 1.0,
                 threshold: float = 0.5):
        super().__init__(learning_rate, n_iterations, tolerance, regularization, C)

        # Граница для классификации: 1 if >= threshold else 0
        self.threshold: float = threshold

    def _sigmoid(self, z: pd.Series) -> pd.Series:
        '''
        Возвращение вероятности вместо класса
        путем применения сигмоиды на линейную комбинацию

        Параметры
        ---------
        z : pd.Series
            Вектор предиктов после линейной комбинации

        Возвращаемое значение
        ---------------------
        pd.Series
            Вероятности отнесения к
            классу 0...1
        '''
        return 1 / (1 + np.exp(-z))

    def _linear_combination(self, X: pd.DataFrame) -> pd.Series:
        '''
        Предварительное получение предикта путем расчета
        линейной комбинации каждого наблюдения и
        текущих значений весов + bias. Затем предикты
        пропускаются через сигмоиду, для получания
        вероятностей отнесения к классу 1

        Параметры
        ---------

        X : pd.DataFrame
            Датасет с векторами наблюдений

        Возвращаемое значение
        ---------------------

        y_pred : pd.Series
            Вектор вероятностей отнесения к классу 1
        '''

        y_pred = self._sigmoid(X.T @ self.weights + self.bias)

        return y_pred

    def _compute_loss(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        '''
        Расчет Loss'a при текущих весах вектора self.weights

        Параметры
        ---------

        y_true : pd.Series
            Вектор с таргетом, истинный класс
            для соответствующих векторов
        y_pred : pd.Series
            Вектор вероятностей отнесения к классу 1
            для соответствующих векторов

        Возвращаемое значение
        ---------------------
        loss : float
            Численное значение ошибки на текущей итерации        
        '''

        loss = -np.mean(y_true * (np.log(y_pred)) + (1 - y_true) * (np.log(1 - y_pred)))

        return loss
