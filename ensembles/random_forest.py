import numpy as np
import pandas as pd
from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

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


class RandomForestClassifier(BaseRandomForest):
    pass


class RandomForestRegressor(BaseRandomForest):
    pass