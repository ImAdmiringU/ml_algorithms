import numpy as np
import pandas as pd

class BaseGradientBoosting:
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rage: float = 1e-1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 subsample: float = 1.0,
                 random_state: int | None = None):
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rage
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.subsample: float = subsample
        self.random_state: int | None = random_state

        self.trees: list = list()
        self.initial_pred: float = None


class GradientBoostingClassifier(BaseGradientBoosting):
    pass


class GradientBoostingRegressor(BaseGradientBoosting):
    pass
