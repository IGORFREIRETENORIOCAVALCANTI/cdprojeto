import numpy as np


class Bootstrap:

    def __init__(self, X):
        self.theta = None
        self.X = X

    def calculate_bootstrap(self, bootstraps, estimator):
        n = len(self.X)
        theta = []

        for i in range(bootstraps):
            X_bootstrap = np.random.choice(n, size=n, replace=True)
            theta.append(estimator(self.X[X_bootstrap]))

        self.theta = theta

    def mean(self):
        return np.mean(self.theta)

    def std(self):
        return np.std(self.theta)
