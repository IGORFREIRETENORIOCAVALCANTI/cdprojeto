from scipy.stats import norm
import numpy as np


class ConfidenceInterval:

    def __init__(self, data, alpha) -> None:
        self.data = data
        self.n = len(data)
        self.mean = np.mean(self.data)
        self.std = np.std(self.data, ddof=0)
        self.margin_of_error = norm.ppf(1 - alpha / 2) * (self.std / np.sqrt(self.n))

    def calculate_lower_bound(self):
        return self.mean - self.margin_of_error

    def calculate_upper_bound(self):
        return self.mean + self.margin_of_error

    def interval(self):
        return self.calculate_lower_bound(), self.calculate_upper_bound()
