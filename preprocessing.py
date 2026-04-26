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
