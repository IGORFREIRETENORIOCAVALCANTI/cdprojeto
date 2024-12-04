import numpy as np
import scipy.stats as st
import abc
import src.rai.pla.optimizers as opt
import src.rai.pla.models as models
import src.rai.pla.stop_criteria as stop
from src.rai.pla.preprocessing import Preprocessing


class Algorithm(abc.ABC):
    def __init__(self, optimizer_strategy: opt.OptimizerStrategy, model: models.Model) -> None:
        self._errors = None
        self._iteration = None
        self.algorithm_observers = []
        self.optimizer_strategy = optimizer_strategy
        self.model = model
        self._rmse = None

    def add(self, observer):
        if observer not in self.algorithm_observers:
            self.algorithm_observers.append(observer)
        else:
            print('Failed to add: {}'.format(observer))

    def remove(self, observer):
        try:
            self.algorithm_observers.remove(observer)
        except ValueError:
            print('Failed to remove: {}'.format(observer))

    def notify_iteration(self):
        [o.notify_iteration(self) for o in self.algorithm_observers]

    def notify_started(self):
        [o.notify_started(self) for o in self.algorithm_observers]

    def notify_finished(self):
        [o.notify_finished(self) for o in self.algorithm_observers]

    @abc.abstractmethod
    def fit(self, X, y, stop_criteria: stop.CompositeStopCriteria):
        """Implement the fit method"""

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, value):
        self._iteration = value

    @property
    def errors(self):
        return self._errors

    @errors.setter
    def errors(self, value):
        self._errors = value

    @property
    def rmse(self):
        return self._rmse

    @rmse.setter
    def rmse(self, value):
        self._rmse = value


class PLA(Algorithm):

    def fit(self, X, y, stop_criteria: stop.CompositeStopCriteria):
        self.iteration = 0
        n, d = X.shape
        self.model.w = np.zeros((d + 1, 1)) + st.norm.rvs(size=(d + 1, 1))
        X_design_matrix = Preprocessing.build_design_matrix(X)
        self.notify_started()

        while True:
            yhat = self.model.predict(X_design_matrix)
            self.errors = y - yhat
            self.rmse = 1.0 / len(X) * np.square(np.linalg.norm(self.errors))
            self.optimizer_strategy.update_model(X_design_matrix, y, self.model)
            self.iteration += 1
            self.notify_iteration()

            if stop_criteria.isFinished(self):
                break

        self.notify_finished()
