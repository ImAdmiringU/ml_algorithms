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
        cov_m = np.cov(X.values, rowvar=False)

        # Получение собственных чисел и векторов
        eigenvalues, eigenvectors = np.linalg.eigh(cov_m)

        # Получение срезов в соответствии с
        # вкладом каждой компоненты (по убыванию)
        indices = np.argsort(eigenvalues)[::-1]

        if self.n_components is not None:
            k = self.n_components
        elif self.n_components is None:
            k = len(X.columns)

        self.components = eigenvectors[:, indices[:k]].T
        self.explained_var_ = eigenvalues[indices] / np.sum(eigenvalues)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Метод для снижения размерности в соответствии
        с полученными компонентами

        Параметры
        ---------
        X : pd.DataFrame
            Датасет для которого производится
            снижение размерности

        Возвращаемое значение
        ---------------------
        res : pd.DataFrame
            Датафрейм с меньшей размерностью
        '''

        # Центрирование входных данных
        X_centered = X - self.mean_

        # Получение проекции данных
        res = X_centered @ self.components.T

        return res

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Объединенный метод обучения и преобразования
        
        Параметры
        ---------
        X : pd.DataFrame
            Датасет для которого производится
            снижение размерности

        Возвращаемое значение
        ---------------------
        res : pd.DataFrame
            Датафрейм с меньшей размерностью
        '''

        self.fit(X=X)

        res = self.transform(X=X)

        return res
