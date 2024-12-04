import abc
import numpy as np


class Model(abc.ABC):

    def __init__(self) -> None:
        self._w = None
        super().__init__()

    @abc.abstractmethod
    def predict(self, X) -> np.ndarray:
        """Implement the predict method"""

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value


class LinearModel(Model):
    def predict(self, X) -> np.ndarray:
        return X @ self.w
