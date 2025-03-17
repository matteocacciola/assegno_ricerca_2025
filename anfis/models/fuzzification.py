from abc import ABC, abstractmethod
import numpy as np
from skfuzzy import gaussmf, trimf, trapmf, sigmf, gbellmf, gauss2mf


class Fuzzification(ABC):
    @abstractmethod
    def fuzzify(self, x: np.ndarray) -> np.ndarray:
        pass


class GaussianFuzzification(Fuzzification):
    def fuzzify(self, x: np.ndarray) -> np.ndarray:
        c = float(np.mean(x))
        sigma = float(np.std(x))
        return gaussmf(x, c, sigma)


class Gaussian2Fuzzification(Fuzzification):
    def fuzzify(self, x: np.ndarray) -> np.ndarray:
        c1 = float(np.mean(x) - np.std(x))
        sigma1 = float(np.std(x))
        c2 = float(np.mean(x) + np.std(x))
        sigma2 = float(np.std(x))
        return gauss2mf(x, c1, sigma1, c2, sigma2)


class TrapezoidFuzzification(Fuzzification):
    def fuzzify(self, x: np.ndarray) -> np.ndarray:
        a = float(np.mean(x) - np.std(x))
        b = float(np.mean(x))
        c = float(np.mean(x))
        d = float(np.mean(x) + np.std(x))

        return trapmf(x, [a, b, c, d])


class TriangleFuzzification(Fuzzification):
    def fuzzify(self, x: np.ndarray, **kwargs) -> np.ndarray:
        a = float(np.mean(x) - np.std(x))
        b = float(np.mean(x))
        c = float(np.mean(x) + np.std(x))

        return trimf(x, [a, b, c])


class SigmoidFuzzification(Fuzzification):
    def fuzzify(self, x: np.ndarray, **kwargs) -> np.ndarray:
        b = float(np.mean(x))
        c = float(np.std(x))
        return sigmf(x, b, c)


class BellFuzzification(Fuzzification):
    def fuzzify(self, x: np.ndarray, **kwargs) -> np.ndarray:
        a = float(np.std(x))
        b = 2
        c = float(np.mean(x))

        return gbellmf(x, a, b, c)
