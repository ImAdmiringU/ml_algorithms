import numpy as np
import pandas as pd

class KMeans:
    def __init__(self,
                 n_clusters: int = 8,
                 max_iter: int = 300,
                 tol: float = 0.0001):
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.tol: float = tol

        self.centroids: np.array = None
        self.labels_: np.array = None
        self.inertia_: np.array = None

    def fit(self, X: pd.DataFrame) -> None:
        '''
        Метод для обучения

        Параметры
        ---------
        X : pd.DataFrame
            Датасет с неразмеченными наблюдениями.
        '''
        pass

    def predict(self, X: pd.DataFrame) -> np.array:
        '''
        Метод для кластеризации

        Параметры
        ---------
        X : pd.DataFrame
            Датасет с неразмеченными наблюдениями

        Возвращаемое значение
        ---------------------
        res : np.array
            Вектор кластеров для соответствующих наблюдений
        '''
        pass

    def fit_predict(self, X: pd.DataFrame) -> np.array:
        '''
        Объединенный метод для обучения -> кластеризации

        Параметры
        ---------
        X : pd.DataFrame
            Датасет с неразмеченными наблюдениями

        Возвращаемое значение
        ---------------------
        res : np.array
            Вектор кластеров для соответствующих наблюдений
        '''
        pass
