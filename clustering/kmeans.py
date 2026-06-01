import numpy as np
import pandas as pd

class KMeans:
    def __init__(self,
                 n_clusters: int = 8,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 random_state: int | None = None):
        '''
        Гиперпараметры модели
        '''
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.random_state: int | None = random_state

        '''
        Данные по кластерам
        '''
        self.centroids: np.array = None
        self.labels_: np.array = None
        self.inertia_: float = 0.0

    def fit(self, X: pd.DataFrame) -> None:
        '''
        Метод для обучения

        Параметры
        ---------
        X : pd.DataFrame
            Датасет с неразмеченными наблюдениями.
        '''

        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Инициализация вектора кластеров
        self.labels_ = np.zeros(len(X), dtype='int')

        # Инициализация центроидов
        indices = np.random.choice(len(X),
                                   replace=False,
                                   size=self.n_clusters)
        self.centroids = X.values[indices]

        for _ in range(self.max_iter):
            # Сбрасывание значения inertia
            self.inertia_ = 0.0

            for j in range(len(self.labels_)):
                # Обновление кластера соответствующего наблюдения
                self.labels_[j] = self._get_label(X.values[j], is_fit=True)
            
            temp_centroids = self.centroids.copy()

            # Обновление координат центроидов
            for cluster in np.arange(self.n_clusters):
                temp_centroids[cluster] = np.mean(X.values[self.labels_ == cluster], axis=0)

            # Смещение центроидов
            distance_diff = np.sum(np.sum((self.centroids - temp_centroids)**2, axis=1)) <= self.tol

            if distance_diff:
                break
            else:
                self.centroids = temp_centroids

    def _get_label(self, v: np.array, is_fit: bool = False) -> int:
        '''
        Метод для получения кластера в соттветствии
        с расстоянием до центроида

        Параметры
        ---------
        v : np.array
            Вектор наблюдения
        
        is_fit : bool
            default: False
            Если True, выполняется апдейт inertia

        Возвращаемое значение
        ---------------------
        temp_label : int
            Текущее значение кластера для
            соответсттвующего наблюдения
        '''

        # Получение вектора квадартов расстояний до каждого центроида
        # и извлечение кластера с минимальным расстоянием
        distances = np.sum((self.centroids - v)**2, axis=1)
        temp_label = distances.argmin()

        if is_fit:
            # Сумма квадратов расстояний от вектора до центроида
            self.inertia_ += distances[temp_label]

        return temp_label

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

        # Инициализация пустого вектора кластеров
        res = np.zeros(len(X), dtype='int')

        for i in range(len(res)):
            res[i] = self._get_label(X.values[i])

        return res

    def fit_predict(self, X: pd.DataFrame) -> np.array:
        '''
        Объединенный метод для обучения -> кластеризации

        Параметры
        ---------
        X : pd.DataFrame
            Датасет с неразмеченными наблюдениями

        Возвращаемое значение
        ---------------------
        self.labels_ : np.array
            Вектор кластеров для соответствующих наблюдений
        '''
        
        # Обучение модели
        self.fit(X=X)

        # Возвращение собственного вектора кластеров
        return self.labels_
