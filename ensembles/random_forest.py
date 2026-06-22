import numpy as np
import pandas as pd
from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
# from cpp_backend.decision_tree_cpp import DTClassifierCPP, DTRegressorCPP

class BaseRandomForest:
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int | None = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: int | str = 'sqrt',
                 bootstrap: bool = True,
                 use_cpp: bool = False):
        self.n_estimators: int = n_estimators
        self.max_depth: int | None = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.max_features: int | str = max_features
        self.bootstrap: bool = bootstrap
        self.use_cpp: bool = use_cpp

        '''
        Список с отдельными экземплярами деревьев RF
        '''
        self.trees: list = list()

    def _bootstrap_sample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        '''
        Создание среза по входному датасету
        для получения bootstrap сэмпла

        Параметры
        ---------
        X : pd.DataFrame
            Исходный датасет с наблюдениями
        y : pd.Series
            Таргет

        Возвращаемое значение
        ---------------------
        res : tuple[pd.DataFrame, pd.Series]
            Bootstrap сэмпл на основе исходного датасета
        '''

        # Получение индексов для среза
        indices = np.random.choice(len(X), len(X), replace=self.bootstrap)
        
        # Срез по наблюдениям
        res = X.iloc[indices], y.iloc[indices]

        return res
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        '''
        Метод для обучения деревьев RF

        Параметры
        ---------
        X : pd.DataFrame
            Исходный датасет со всеми наблюдениями
        y : pd.Series
            Исходный вектор таргет
        '''

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X=X, y=y)

            # Создаем новое дерево на каждой итерации
            tree = self._create_tree()

            # Обучение соответствующего дерева
            tree.fit(X_sample, y_sample)

            # Добавляем объект в список для формирования RF
            self.trees.append(tree)
    
    def predict(self, X: pd.DataFrame) -> np.array:
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

        # Получаем вектора предиктов каждого дерева
        predictions = np.array([tree.predict(X) for tree in self.trees])

        # Агрегируем вектора из предиктов
        res = self._aggregate(predictions=predictions)

        return res

    def _create_tree(self):
        raise NotImplementedError()
    
    def _aggregate(self, predictions: np.array) -> np.array:
        raise NotImplementedError()


class RandomForestClassifier(BaseRandomForest):
    def _create_tree(self) -> DecisionTreeClassifier:
        '''
        Метод для создания экземпляра DecisionTreeClassifier
        с заданными параметрами

        Возвращаемое значение
        ---------------------
        tree : DecisionTreeClassifier | DTClassifierCPP
            Экземпляр соответствующего объекта
        '''

        if self.use_cpp:
            # Будущая реализация сплита на C++
            # tree = DTClassifierCPP(max_depth=self.max_depth,
            #                        min_samples_split=self.min_samples_split,
            #                        min_samples_leaf=self.min_samples_leaf,
            #                        max_features=self.max_features)
            pass
        else:
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf,
                                          max_features=self.max_features)

        return tree

    def _aggregate(self, predictions: np.array) -> np.array:
        '''
        Метод для агрегации и получения итогового
        вектора классов путем голосования большинства

        Параметры
        ---------
        predictions : np.array
            Двумерный массив с векторами предиктов
            каждого экземпляра дерева

        Возвращаемое значение
        ---------------------
        res : np.array
            Итоговый вектор предиктов по
            всем деревьям
        '''

        # Формирование большинства для каждого наблюдения
        # по всем деревьям
        res = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(),
                                  axis=0,
                                  arr=predictions)
        
        return res


class RandomForestRegressor(BaseRandomForest):
    def _create_tree(self) -> DecisionTreeRegressor:
        '''
        Метод для создания экземпляра DecisionTreeRegressor
        с заданными параметрами

        Возвращаемое значение
        ---------------------
        tree : DecisionTreeRegressor | DTRegressorCPP
            Экземпляр соответствующего объекта
        '''

        if self.use_cpp:
            # Будущая реализация сплита на C++
            # tree = DTRegressorCPP(max_depth=self.max_depth,
            #                       min_samples_split=self.min_samples_split,
            #                       min_samples_leaf=self.min_samples_leaf,
            #                       max_features=self.max_features)
            pass
        else:
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf,
                                         max_features=self.max_features)

        return tree
    
    def _aggregate(self, predictions: np.array) -> np.array:
        '''
        Метод для агрегации и получения итогового
        вектора средних значений

        Параметры
        ---------
        predictions : np.array
            Двумерный массив с векторами предиктов
            каждого экземпляра дерева

        Возвращаемое значение
        ---------------------
        res : np.array
            Итоговый вектор предиктов по
            всем деревьям
        '''

        # Формирование средних значений
        # по всем деревьям
        res = np.mean(predictions, axis=0)
        
        return res