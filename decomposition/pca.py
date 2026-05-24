import numpy as np
import pandas as pd

class PCA:
    def __init__(self,
                 n_components: int | None = None):
        self.n_components: int | None = n_components
        self.components: np.array = None
        self.explained_var_: np.array = None
        self.mean_: np.array = None

    def fit(self, X: pd.DataFrame) -> None:
        '''
        Метод для получения основных атрибутов датасета,
        компонент для снижения размерности

        Параметры
        ---------
        X : pd.DataFrame
            Датасет с векторами наблюдений
        '''

        # Вычисление среднего по фичам
        self.mean_ = np.mean(X.values, axis=0)

        # Вычисление ковариационной матрицы
        # на основе центрированных данных
        cov_m = np.cov(X.values - self.mean_, rowvar=False)

        # Получение собственных чисел и векторов
        eigenvalues, eigenvectors = np.linalg.eigh(cov_m)

        # Получение срезов в соответствии с
        # вкладом каждой компоненты (по убыванию)
        indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, indices[:self.n_components]].T
        self.explained_var_ = eigenvalues[indices] / np.sum(eigenvalues)
