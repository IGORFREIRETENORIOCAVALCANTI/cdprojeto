import numpy as np


class Preprocessing:

    @staticmethod
    def build_design_matrix(X):
        ones = np.ones((len(X), 1))
        return np.hstack((ones, X))
