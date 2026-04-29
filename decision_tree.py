import base
import numpy as np
import pandas as pd

class BaseDecisionTree:
    def __init__(self,
                 criterion: str,
                 max_depth: int | None,
                 min_samples_split: int,
                 min_samples_leaf: int):
        
        '''
        Инициализация гиперпараметров
        '''
        self.criterion: str = criterion
        self.max_depth: int | None = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf

        '''
        Атрибуты текущего узла дерева
        '''
        self.feature_index: int = None
        self.threshold: float = None
        self.impurity: float = None
        self.value: any = None
        self.left_node: DecisionTreeClassifier | DecisionTreeRegressor = None
        self.right_node: DecisionTreeClassifier | DecisionTreeRegressor = None

    @property
    def is_leaf(self) -> bool:
        '''
        Возвращаем True, если y объекта нет
        левой и правой ноды -данный объект
        считается листом. Возвращаем False
        в ином случае
        '''
        return True if (self.left_node is None and self.right_node is None) else False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        '''
        Точка входа для обучения дерева

        Параметры
        ---------
        X : pd.DataFrame
            Обработанный датафрейм c векторами
            экземпляров объектов для обучения
        y : pd.Series
            Таргет. Целевое значение класса
            для каждого вектора в X
        '''
        self._fit(X, y)

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        '''
        Метод для рекурсивного обучения дерева

        Параметры
        ---------
        X : pd.DataFrame
            Обработанный датафрейм c векторами
            экземпляров объектов для обучения
        y : pd.Series
            Таргет. Целевое значение класса
            для каждого вектора в X
        '''
        
        self.impurity = self._initial_impurity(y_data=y.values)
        self.value = self._leaf_value(y=y.values)

        # Предварительное определение условий
        # создания следующих нод
        depth_ok = (self.max_depth is None) or (self.max_depth != 0)
        size_ok = self.min_samples_split <= len(X)
        impurity_ok = self.impurity > 0

        if depth_ok and size_ok and impurity_ok:
            # Поиск фичи и границы оптимального сплита
            self.feature_index, self.threshold = self._find_best_split(X=X, y=y)

            if self.feature_index is not None:
                new_depth = None if self.max_depth is None else self.max_depth - 1

                # Маска для разделения на левую и правую ноду (pd.DataFrame)
                mask = X.iloc[:, self.feature_index] < self.threshold
                left_split = (X[mask], y[mask]) # значения меньше threshold
                right_split = (X[~mask], y[~mask]) # значения больше threshold

                if len(left_split[0]) >= self.min_samples_leaf:
                    self.left_node = self.__class__(criterion=self.criterion,
                                                    max_depth=new_depth,
                                                    min_samples_split=self.min_samples_split,
                                                    mins_samples_leaf=self.min_samples_leaf)
                    self.left_node._fit(*left_split)

                if len(right_split[0]) >= self.min_samples_leaf:
                    self.right_node = self.__class__(criterion=self.criterion,
                                                     max_depth=new_depth,
                                                     min_samples_split=self.min_samples_split,
                                                     min_smaples_leaf=self.min_samples_leaf)
                    self.right_node._fit(*right_split)

    def _initial_impurity(self, y: np.array) -> float:
        raise NotImplementedError()
    
    def _leaf_value(self, y: np.array) -> int | float:
        raise NotImplementedError()

    def _find_best_split(self, X: pd.DataFrame, y: pd.Series) -> tuple[int, float]:
        '''
        Нахождение наилучшего сплита

        Параметры
        ---------
        X : pd.DataFrame
            Обработанный датафрейм c векторами
            экземпляров объектов для обучения
        y : pd.Series
            Таргет. Целевое значение класса
            для каждого вектора в X

        Временные переменные
        --------------------
        best_splits : list
            Список лучших сплитов, состоящий из
            индекса, границы сплита, значения
            критерия сплита
        best_feature : int
            Лучший индекс фичи для сплита
        best_threshold : float
            Лучшее значение фичи по которой
            происходит сплит
        best_impurity : float
            Лучшее значение критерия в узле
        X_y_data : np.array
            Объединенные данные из X и y
            преобразованные из pd.DataFrame
            в np.array

        Возвращаемое значение
        ---------------------
        res : tuple[int, float]
            Кортеж из индекса лучшей фичи,
            значения для сплита
        '''
        X_y_data = np.hstack((X.values, y.values.reshape(len(y), -1)))
        best_splits: list = []
        
        # Проходим циклом по всем фичам через индекс, соединяя данные по фиче (np.array) c
        # таргет-значением в y (np.array), получая двумерный массив для сортировки и сплита
        for i in range(len(X.columns)):
            
            best_feature: int = None
            best_threshold: float = None
            best_impurity: float = None

            # Получаем индексы сортировки через argsort() для соответствующих значений
            # фичи, первые [indexes] задают сортировку для X_y_data, последующие [slice]
            # делают срез по всем строкам, но только нужная фича i и последний столбец
            # y-таргет данные (индекс -1)
            temp = X_y_data[X_y_data[:, i].argsort()][:, [i, -1]]

            for j in range(1, len(temp)):
                left_val, right_val = temp[j - 1, 0], temp[j, 0]

                if temp[j - 1, -1] != temp[j, -1]:
                    temp_threshold = (left_val + right_val) / 2
                    mask = temp[:, 0] < temp_threshold
                    
                    # разбиение на левую и правую ноду
                    # по среднему значению на границе
                    # между классами
                    temp_left_data = temp[mask]
                    temp_right_data = temp[~mask]

                    if len(temp) < self.min_samples_split or (len(temp_left_data) < self.min_samples_leaf
                                                              or len(temp_right_data) < self.min_samples_leaf):
                        continue
                    
                    # Предварительный расчет impurity
                    # по текущему сплиту
                    temp_impurity = self._calculate_impurity(temp_left_data[:, -1],
                                                             temp_right_data[:, -1])
                    
                    if best_impurity is None or best_impurity < temp_impurity:
                        best_feature = i
                        best_threshold = temp_threshold
                        best_impurity = temp_impurity
                else:
                    continue
            
            best_splits.append((best_feature,
                               best_threshold,
                               best_impurity))
            
        # Убираем None элементы, если не нашелся сплит
        # Если список окажется пустым, возвращаем кортеж из
        # двух None значений
        valid_splits = [i for i in best_splits if i[0] is not None]

        if not valid_splits:
            return (None, None)

        res = sorted(valid_splits, key=lambda x: x[-1])

        return res[-1][:2]
        
    def _calculate_impurity(self, y_left, y_right):
        raise NotImplementedError()

    def predict(self, X) -> np.array:
        '''
        Точка входа для предикта
        класса объекта. Применение
        дерева ко всем объектам

        Параметры
        ---------
        X : pd.DataFrame
            Датафрейм с векторами объектов,
            для которых будет осуществляться
            предсказание классов

        Возвращаемое значение
        ---------------------
        res : np.array
            Вектор numpy, с предсказанными
            классами для векторов X
        '''

        res = np.zeros(len(X))

        # С помощью цикла проходимся по всем векторам X
        # и определяем для каждого класс, сохраняя в вектор res
        for i in range(len(res)):
            res[i] = self._predict_row(X.iloc[i, :])
        
        return res

    def _predict_row(self, X) -> int:
        '''
        Возвращает класс для
        соответствующего вектора

        Параметры
        ---------
        X : pd.Series
            Pandas серия из одного вектора
            для которого выполняется
            определение класса

        Возвращаемое значение
        ---------------------
        temp_class : int
            Значение класса в листе дерева
        '''

        if self.is_leaf:
            return self.value
        else:
            # X - это pd.Series из одного вектора
            # поэтому используем такой вид среза.
            # В данном случае фичи располагаются
            # как индексы
            if X.iloc[self.feature_index] < self.threshold:
                return self.left_node._predict_row(X=X)
            else:
                return self.right_node._predict_row(X=X)


class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self,
                 criterion: str = 'entropy',
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf)

    def _initial_impurity(self, y: np.array):
        match self.criterion:
            case 'entropy':
                return base.entropy(y=y)
            case 'gini':
                return base.gini(y=y)
    
    def _leaf_value(self, y: np.array) -> int:
        '''
        Определение класса объекта в узле
        и присвоение этого значения
        атрибуту объекта

        Параметры
        ---------
        y : np.array
            Таргет. Целевое значение класса
            для каждого вектора в X
        '''

        # Используя метод np.bincount() подсчитываем вхождения
        # через метод argmax() берем наиболее частый класс (индекс)
        temp_class = np.bincount(y).argmax()

        return temp_class
    
    def _calculate_impurity(self, y_left, y_right):
        match self.criterion:
            case 'entropy':
                return base.information_gain(self.impurity,
                                             y_left,
                                             y_right)
            case 'gini':
                pass


class DecisionTreeRegressor(BaseDecisionTree):
    def __init__(self,
                 criterion: str = 'mse',
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf)
