import base
import numpy as np
import pandas as pd

class BaseLinearModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def _predict_proba(self, X):
        pass

    def _add_regularization(self, gradient):
        pass

    def _compute_loss(self, y_ture, y_pred):
        raise NotImplementedError()

    def _compute_gradient(self, X, y_true, y_pred):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

