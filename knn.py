import numpy as np
import pandas as pd
import base

class BaseKNN:
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 metric='euclidian'):
        
        '''
        Инициализация гиперпараметров
        -----------------------------

        n_neighbours : int
            Количество ближайших соседей
            для учета голосов классификации
            или регрессии
        weights : str
            Какой вклад вносится каждым
            соседним объектом

            'unifom' - [1, 1, 1] - все соседы
            имеют равные вес
            'distance' - вклад каждого соседа
            обратно пропорционален расстоянию
            до него
        metric : str
            manhattan (L1-norm), euclidian (L2-norm),
            cosine (косинусное расстояние)
        '''
        self.n_neighbors: int = n_neighbors
        self.weights: str = weights
        self.metric: str = metric

        '''
        Инициализация атрибутов объекта
        -------------------------------

        X_train : pd.DataFrame
            Датасет с нормализованными
            векторами наблюдений для обучения
        y_train : pd.Series
            Вектор с таргет значениями для
            каждого наблюдения
        '''
        self.X_train: pd.DataFrame = None
        self.y_train: pd.Series = None

    def fit(self, X, y) -> None:
        pass

    def predict(self, X) -> np.array:
        pass

    def _predict_row(self, X) -> int | float:
        pass

    def _distance(self, a, b) -> float:
        pass

    def _get_weights(self, distances) -> np.array:
        pass

    def _aggregate(self, neighbor_labels, neighbor_distances) -> int | float:
        raise NotImplementedError
