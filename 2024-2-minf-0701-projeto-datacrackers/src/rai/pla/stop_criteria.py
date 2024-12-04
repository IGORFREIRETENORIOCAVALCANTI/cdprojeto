from __future__ import annotations

import abc
from typing import List

import src.rai.pla.algorithms as al


class StopCriteria(abc.ABC):

    @abc.abstractmethod
    def isFinished(self, alg: al.Algorithm) -> bool:
        """Implement stop criterium"""


class MinErrorStopCriteria(StopCriteria):

    def __init__(self, min_error) -> None:
        self.min_error = min_error

    def isFinished(self, alg: al.Algorithm) -> bool:
        return alg.rmse < self.min_error


class MaxIterationStopCriteria(StopCriteria):

    def __init__(self, max_iteration) -> None:
        self.max_iteration = max_iteration

    def isFinished(self, alg: al.Algorithm) -> bool:
        return alg.iteration >= self.max_iteration


class CompositeStopCriteria(StopCriteria):

    def __init__(self) -> None:
        self.stop_criteria: List[StopCriteria] = []

    def isFinished(self, alg: al.Algorithm) -> bool:
        for criteria in self.stop_criteria:
            if criteria.isFinished(alg):
                return True
        return False

    def add(self, element: StopCriteria) -> None:
        self.stop_criteria.append(element)
