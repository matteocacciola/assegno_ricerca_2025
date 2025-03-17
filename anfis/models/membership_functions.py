from abc import ABC, abstractmethod
import numpy as np
from skfuzzy import gaussmf, trimf, trapmf, sigmf, gbellmf, gauss2mf


class MembershipFunction(ABC):
    def __init__(self, params: list[float] | None = None):
        self.params = params if params is not None and len(params) == self.num_params else (
            np.random.rand(self.num_params).tolist()
        )

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass

    @abstractmethod
    def update_params(self, x: np.ndarray, error: np.ndarray, learning_rate: float):
        pass

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        pass


class GaussianMembershipFunction(MembershipFunction):
    @property
    def num_params(self) -> int:
        return 2

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return gaussmf(x, self.params[0], self.params[1])

    def update_params(self, x: np.ndarray, error: np.ndarray, learning_rate: float):
        mean, sigma = self.params

        membership = self.evaluate(x)

        # partial derivatives of the gaussian function y = exp(-(x-mean)^2/(2*sigma^2))
        # dy/dmean = (x-mean)/(sigma^2) * y
        # dy/dsigma = ((x-mean)^2/(sigma^3)) * y
        # calculating the gradient
        grad_mean = np.sum(error * membership * (x - mean) / (sigma ** 2), axis=0)
        grad_sigma = np.sum(error * membership * ((x - mean) ** 2) / (sigma ** 3), axis=0)

        self.params[0] -= learning_rate * grad_mean
        self.params[1] -= learning_rate * grad_sigma

        # Ensure that sigma is not too small
        self.params[1] = max(self.params[1], 1e-5)


class Gaussian2MembershipFunction(MembershipFunction):
    @property
    def num_params(self):
        return 4

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return gauss2mf(x, self.params[0], self.params[1], self.params[2], self.params[3])

    def update_params(self, x: np.ndarray, error: np.ndarray, learning_rate: float):
        mean1, sigma1, mean2, sigma2 = self.params

        membership = self.evaluate(x)

        # partial derivatives of the gaussian function y = exp(-(x-mean)^2/(2*sigma^2))
        # dy/dmean = (x-mean)/(sigma^2) * y
        # dy/dsigma = ((x-mean)^2/(sigma^3)) * y
        # calculating the gradient
        grad_mean1 = np.sum(error * membership * (x - mean1) / (sigma1 ** 2), axis=0)
        grad_sigma1 = np.sum(error * membership * ((x - mean1) ** 2) / (sigma1 ** 3), axis=0)
        grad_mean2 = np.sum(error * membership * (x - mean2) / (sigma2 ** 2), axis=0)
        grad_sigma2 = np.sum(error * membership * ((x - mean2) ** 2) / (sigma2 ** 3), axis=0)

        self.params[0] -= learning_rate * grad_mean1
        self.params[1] -= learning_rate * grad_sigma1
        self.params[2] -= learning_rate * grad_mean2
        self.params[3] -= learning_rate * grad_sigma2

        # Ensure that sigmas are not too small
        self.params[1] = max(self.params[1], 1e-5)
        self.params[3] = max(self.params[3], 1e-5)


class TrapezoidMembershipFunction(MembershipFunction):
    def __init__(self, params: list[float] | None = None):
        super().__init__(params=params)
        self.params = sorted(self.params)

    @property
    def num_params(self):
        return 4

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return trapmf(x, self.params)

    def update_params(self, x: np.ndarray, error: np.ndarray, learning_rate: float):
        a, b, c, d = self.params

        # partial derivatives of the trapezoidal function:
        # f(x) = 0, x <= a
        # f(x) = (x-a)/(b-a), a <= x <= b
        # f(x) = 1, b <= x <= c
        # f(x) = (d-x)/(d-c), c <= x <= d
        # f(x) = 0, x >= d

        # initialize gradients
        d_a = np.zeros_like(x)
        d_b = np.zeros_like(x)
        d_c = np.zeros_like(x)
        d_d = np.zeros_like(x)

        # computation of gradients in the relevant regions
        mask_ab = (x >= a) & (x <= b)
        mask_cd = (x >= c) & (x <= d)

        if b > a:
            d_a[mask_ab] = -(x[mask_ab] - b) / ((b - a) ** 2)
            d_b[mask_ab] = (x[mask_ab] - a) / ((b - a) ** 2)

        if d > c:
            d_c[mask_cd] = -(d - x[mask_cd]) / ((d - c) ** 2)
            d_d[mask_cd] = (x[mask_cd] - c) / ((d - c) ** 2)

        self.params[0] -= learning_rate * np.sum(error * d_a)
        self.params[1] -= learning_rate * np.sum(error * d_b)
        self.params[2] -= learning_rate * np.sum(error * d_c)
        self.params[3] -= learning_rate * np.sum(error * d_d)

        self.params = np.sort(self.params)


class TriangleMembershipFunction(MembershipFunction):
    def __init__(self, params: list[float] | None = None):
        super().__init__(params=params)
        self.params = np.ndarray(sorted(self.params))

    @property
    def num_params(self):
        return 3

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return trimf(x, self.params)

    def update_params(self, x: np.ndarray, error: np.ndarray, learning_rate: float):
        a, b, c = self.params

        d_a = np.zeros_like(x)
        d_b = np.zeros_like(x)
        d_c = np.zeros_like(x)

        mask_ab = (x >= a) & (x <= b)
        mask_bc = (x > b) & (x <= c)

        if b > a:
            d_a[mask_ab] = -1 / (b - a)
            d_b[mask_ab] = (x[mask_ab] - a) / ((b - a) ** 2)

        if c > b:
            d_b[mask_bc] = (c - x[mask_bc]) / ((c - b) ** 2)
            d_c[mask_bc] = -1 / (c - b)

        self.params[0] -= learning_rate * np.sum(error * d_a)
        self.params[1] -= learning_rate * np.sum(error * d_b)
        self.params[2] -= learning_rate * np.sum(error * d_c)

        self.params = np.sort(self.params)


class SigmoidMembershipFunction(MembershipFunction):
    @property
    def num_params(self):
        return 2

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return sigmf(x, self.params[0], self.params[1])

    def update_params(self, x: np.ndarray, error: np.ndarray, learning_rate: float):
        # f(x) = 1 / (1 + exp(-a * (x - c)))
        a, c = self.params

        current_output = self.evaluate(x)

        # calculating the gradients of the sigmoid function
        # df/da = (x - c) * exp(-a * (x - c)) / (1 + exp(-a * (x - c)))^2 = (x - c) * current_output * (1 - current_output)
        # df/dc = -a * exp(-a * (x - c)) / (1 + exp(-a * (x - c)))^2 = -a * current_output * (1 - current_output)

        grad_a = np.sum(error * (x - c) * current_output * (1 - current_output))
        grad_c = np.sum(error * (-a) * current_output * (1 - current_output))

        self.params[0] -= learning_rate * grad_a
        self.params[1] -= learning_rate * grad_c


class BellMembershipFunction(MembershipFunction):
    @property
    def num_params(self):
        return 3

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return gbellmf(x, self.params[0], self.params[1], self.params[2])

    def update_params(self, x: np.ndarray, error: np.ndarray, learning_rate: float):
        # f(x) = 1 / (1 + |((x - c) / a)|^(2b))
        a, b, c = self.params

        current_output = self.evaluate(x)

        # calculating the gradients of the bell function
        # df/da = 2b * (|((x - c) / a)|^(2b) / (a * (1 + |((x - c) / a)|^(2b))^2) = -2b * (1 - current_output) * current_output / a
        # df/db = -2 * log(|(x - c) / a|) * |(x - c) / a|^(2b) / ((1 + |(x - c) / a|^(2b))^2) = -2 * current_output * (1 - current_output) * log(|(x - c) / a|)
        # df/dc = (-2b / (x - c))  * |(x - c) / a|^(2b)  * (1 / (1 + |(x - c) / a|^(2b))^2

        grad_a = np.sum(error * -2 * b * (1 - current_output) * current_output / a)
        grad_b = np.sum(error * -2 * current_output * (1 - current_output) * np.log(np.abs((x - c) / a)))
        grad_c = np.sum(error * -2 * b * (x - c) * (1 - current_output ) * current_output)

        self.params[0] -= learning_rate * grad_a
        self.params[1] -= learning_rate * grad_b
        self.params[2] -= learning_rate * grad_c

        # Ensure that a and b are not too small
        self.params[0] = max(self.params[0], 1e-5)
        self.params[1] = max(self.params[1], 1e-5)
