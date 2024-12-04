import abc
import numpy as np


class OptimizerStrategy(abc.ABC):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def update_model(self, X, y, model):
        """Implement Update Weigth Strategy"""


class SteepestDescentMethod(OptimizerStrategy):

    def __init__(self, learning_rate) -> None:
        super().__init__(learning_rate)

    def update_model(self, X, y, model) -> None:
        XT_Xw = X.T @ X @ model.w
        XT_y = X.T @ y
        w_updated = model.w - self.learning_rate * (2 / len(X)) * (XT_Xw - XT_y)
        model.w = w_updated


class NewtonsMethod(OptimizerStrategy):

    def __init__(self, learning_rate) -> None:
        super().__init__(learning_rate)

    def update_model(self, X, y, model) -> None:
        XT_X_inv = np.linalg.inv(X.T @ X)
        XT_y = X.T @ y
        w_updated = (1 - self.learning_rate) * model.w + self.learning_rate * XT_X_inv @ XT_y
        model.w = w_updated
