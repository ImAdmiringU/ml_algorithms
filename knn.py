import numpy as np
import pandas as pd

class BaseKNN:
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 metric='euclidian'):
        
        '''
        Инициализация гиперпараметров
        -----------------------------

        n_neighbors : int
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

        X_train : np.array
            Датасет с нормализованными
            векторами наблюдений для обучения
        y_train : np.array
            Вектор с таргет значениями для
            каждого наблюдения
        '''
        self.X_train: np.array = None
        self.y_train: np.array = None

    def fit(self, X, y) -> None:
        '''
        Запись датасета и таргета в
        соответствующие атрибуты объекта
        '''
        self.X_train = X.to_numpy()
        self.y_train = y.to_numpy()

    def predict(self, X: pd.DataFrame) -> np.array:
        '''
        Решение классификационной или
        регрессионной задачи для каждого
        наблюдения

        Параметры
        ---------

        X : pd.DataFrame
            Датасет для которого предсказывается
            таргет значение
        '''
        temp_predictions = np.zeros(len(X))

        for i in range(len(temp_predictions)):
            temp_predictions[i] = self._predict_row(X.iloc[i, :].to_numpy())

        return temp_predictions

    def _predict_row(self, X: np.array) -> int | float:
        '''
        Решение классификационной или
        регрессионной задачи для одного
        наблюдения

        Параметры
        ---------
        X : np.array
            n-мерный вектор одного наблюдения
            для которого предсказывается
            значение таргета

        Возвращаемое значение
        ---------------------
        int | float
            таргет значение
            текущего вектора
        '''

        # Вектор расстояний от векторов датасета (X_train[i])
        # до вектора для которого решаем задачу (X)
        temp_distances = np.zeros(len(self.X_train))

        # Циклом проходим по всем векторам датасета для обучения
        # и определяем расстояние до вектора, для которого
        # расчитываем таргет значение
        for i in range(len(temp_distances)):
            temp_distances[i] = self._distance(X, self.X_train[i])
        
        # Получаем индексы ближайших соседей
        indx = np.argsort(temp_distances)[:self.n_neighbors]

        return self._aggregate(self.y_train[indx], temp_distances[indx])

    def _distance(self, a: np.array, b: np.array) -> float:
        '''
        В зависимости от указанной метрики
        модели, возвращает L1 или L2 норму
        вектора разности

        Параметры
        ---------
        a : np.array
            Вектор для которого предсказывается
            таргет значение
        b : np.array
            Вектор записанного наблюдения из датасета
            в self.X_train[i]

        Возвращаемое значение
        ---------------------
        float
            Соответствующее значение нормы вектора
        '''

        match self.metric:
            case 'manhattan':
                return np.linalg.norm(a - b, 1)
            case 'euclidian':
                return np.linalg.norm(a - b, 2)

    def _get_weights(self, distances) -> np.array:
        '''
        Метод, возвращающий np.array вектор
        весов, в зависимости от параметра
        модели
        '''
        if self.weights == 'uniform':
            return np.ones(len(distances))
        elif self.weights == 'distance':
            # Эпсилон eps = 0.00001 - для
            # предотвращения деления на ноль
            eps = 0.00001
            
            return 1 / (distances + eps)
        else:
            return None

    def _aggregate(self, neighbor_labels, neighbor_distances) -> int | float:
        '''
        Реализация классификации или регрессора
        в соответствующем классе наследнике
        '''
        raise NotImplementedError


class KNNClassifier(BaseKNN):
    def _aggregate(self, neighbor_labels, neighbor_distances) -> int:
        # Двумерный массив с классом и весом
        # для соответствующего вектора
        data = np.c_[neighbor_labels, self._get_weights(neighbor_distances)]
        
        # Словарь для подсчета голосов
        # для каждого класса
        count_dict = dict()

        # Подсчет голосов в зависимости от весов
        for i in range(len(data)):
            count_dict[data[i, 0]] = count_dict.get(data[i, 0], 0) + data[i, 1]

        if len(count_dict.items()) == 1:
            return list(count_dict.keys())[0]
        else:
            if list(count_dict.items())[0][1] >= list(count_dict.items())[1][1]:
                return list(count_dict.items())[0][0]
            else:
                return list(count_dict.items())[1][0]


class KNNRegressor(BaseKNN):
    def _aggregate(self, neighbor_labels, neighbor_distances) -> float:
        # Получаем веса в соответствии
        # с параметром модели
        weights = self._get_weights(neighbor_distances)

        # Взвешенное среднее
        # сумма (метка * вес) / сумма весов
        weighted_mean = np.sum(neighbor_labels * weights) / np.sum(weights)

        return weighted_mean
