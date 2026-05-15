import numpy as np
import pandas as pd

class StandardScaler:
    def __init__(self):
        '''
        Атрибуты объекта:
        mean_   - среднее
        std_    - стандартное отклонение
        '''
        self.mean_: pd.Series = None
        self.std_: pd.Series = None

    def fit(self, X: pd.DataFrame) -> None:
        '''
        Обучение объекта на текущей выборке
        и получение для него mean и std
        
        Параметры
        ---------
        X : pd.DataFrame
            Датасет с векторами наблюдений
        '''
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Преобразование исходного датасет
        к нормализованному виду

        Параметры
        ---------
        X : pd.DataFrame
            Датасет с векторами наблюдений

        Временные переменные
        --------------------
        scaled_X : pd.DataFrame
            Нормализованный датасет

        Возвращаемое значение
        ---------------------
        scaled_X : pd.DataFrame
            Нормализованный датасет
        '''

        scaled_X = (X - self.mean_) / self.std_

        return scaled_X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Метод объединяющий fit и transform.
        Сначала обучаем объект под датасет,
        далее преобразуем и возвращаем
        нормализованным

        Параметры
        ---------
        X : pd.DataFrame
            Необработанынй датасет с наблюдениями
        '''

        # Обучаем объект
        self.fit(X=X)

        # Возвращаем нормализованный датасет
        return self.transform(X=X)


class MinMaxScaler:
    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_min: float = feature_range[0]
        self.feature_max: float = feature_range[1]
        self.min_: np.array = None
        self.max_: np.array = None
        self.scale_: np.array = None
        self.mask: np.array = None
        
    def fit(self, X: pd.DataFrame) -> None:
        '''
        Получения исходных параметров
        для скейлинга

        X_std = (X - X_min) / (X_max - X_min)

        Используем scale и min, max, для экономии
        памяти, не сохраняя исходный датасет

        Параметры
        ---------
        X : pd.DataFrame
            Исходный датафрейм с наблюдениями
        '''

        self.min_ = X.min().values
        self.max_ = X.max().values

        # Маска для исключения деления на ноль (0)
        self.mask = (self.min_ == self.max_)

        self.scale_ = np.where(self.mask,
                               0,
                               (self.feature_max - self.feature_min) / (self.max_ - self.min_))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Преобразование датафрейма в соответствии
        с вычисленным self.scale_

        Параметры
        ---------
        X : pd.DataFrame
            Датасет для скейлинга

        Возвращаемое значение
        ---------------------
        X_scaled : pd.DataFrame
            Датасет после скейлинга
        '''

        X_scaled = np.where(self.mask,
                            self.feature_min,
                            self.feature_min + (X - self.min_) * self.scale_)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Объединенный метод обучения и преобразования

        Параметры
        ---------
        X : pd.DataFrame
            Датасет для скейлинга
        
        Возвращаемое значение
        ---------------------
        X_scaled : pd.DataFrame
            Датасет после скейлинга
        '''

        self.fit(X=X)
        X_scaled = self.transform(X=X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Операция обратная transform. Возврат к исходному
        датафрейму до произведения скейлинга

        Параметры
        ---------
        X : pd.DataFrame
            Скалированный датасет
        
        Возвращаемое значение
        ---------------------
        X_inversed : pd.DataFrame
            Датасет с исходными значениями
        '''
        pass
