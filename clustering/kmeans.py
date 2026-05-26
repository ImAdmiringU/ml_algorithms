import numpy as np
import pandas as pd

class KMeans:
    def __init__(self,
                 n_clusters: int = 8,
                 max_iter: int = 300,
                 tol: float = 0.0001):
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.tol: float = tol

        self.centroids: np.array = None
        self.labels_: np.array = None
        self.inertia_: np.array = None

    def fit(self):
        pass

    def predict(self):
        pass

    def fit_predict(self):
        pass
