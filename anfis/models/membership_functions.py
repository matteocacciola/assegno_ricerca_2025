from abc import ABC, abstractmethod
import numpy as np
from skfuzzy import gaussmf, trimf, trapmf, sigmf, gbellmf, gauss2mf

np.seterr(divide="ignore", invalid="ignore")


class MembershipFunction(ABC):
    def __init__(self, params: list[float] | None = None):
        self.params = list(params) if params is not None and len(params) == self.num_params else (
            np.random.rand(self.num_params).tolist()
        )
        self._epsilon = 1e-9

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass

    @abstractmethod
    def update_params(self, x: np.ndarray, delta_premise: np.ndarray, learning_rate: float):
        pass

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        pass

    def get_params(self) -> list[float]:
        return self.params.copy()


class GaussianMembershipFunction(MembershipFunction):
    @property
    def num_params(self) -> int:
        return 2 # [mean, sigma]

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        mean, sigma = self.params
        safe_sigma = np.maximum(np.abs(sigma), self._epsilon)
        return gaussmf(x, mean, safe_sigma)

    def update_params(self, x: np.ndarray, delta_premise: np.ndarray, learning_rate: float):
        mean, sigma = self.params
        safe_sigma = float(np.maximum(np.abs(sigma), self._epsilon))
        batch_size = x.shape[0]
        if batch_size == 0: return

        membership = gaussmf(x, mean, safe_sigma)

        # partial derivatives of the gaussian function y = exp(-(x-mean)^2/(2*sigma^2))
        # dy/dmean = (x-mean)/(sigma^2) * y
        # dy/dsigma = ((x-mean)^2/(sigma^3)) * y
        # calculating the gradient
        diff_x_mean = x - mean
        d_mu_d_mean = membership * diff_x_mean / (safe_sigma ** 2)
        d_mu_d_sigma = membership * (diff_x_mean ** 2) / (safe_sigma ** 3)
        d_mu_d_sigma *= np.sign(sigma) if sigma != 0 else 1

        grad_mean = np.sum(delta_premise * d_mu_d_mean) / batch_size
        grad_sigma = np.sum(delta_premise * d_mu_d_sigma) / batch_size

        self.params[0] -= learning_rate * grad_mean
        self.params[1] -= learning_rate * grad_sigma

        self.params[1] = np.maximum(np.abs(self.params[1]), self._epsilon)


class Gaussian2MembershipFunction(MembershipFunction):
    @property
    def num_params(self) -> int:
        return 4

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return gauss2mf(x, self.params[0], self.params[1], self.params[2], self.params[3])

    def update_params(self, x: np.ndarray, delta_premise: np.ndarray, learning_rate: float):
        mean1, sigma1, mean2, sigma2 = self.params
        safe_sigma1 = np.maximum(np.abs(sigma1), self._epsilon)
        safe_sigma2 = np.maximum(np.abs(sigma2), self._epsilon)
        batch_size = x.shape[0]
        if batch_size == 0: return

        g1 = gaussmf(x, mean1, safe_sigma1)
        g2 = gaussmf(x, mean2, safe_sigma2)

        diff1 = x - mean1
        dg1_dmean1 = g1 * diff1 / (safe_sigma1 ** 2)
        dg1_dsigma1 = g1 * (diff1 ** 2) / (safe_sigma1 ** 3) * np.sign(sigma1) if sigma1 != 0 else 0

        diff2 = x - mean2
        dg2_dmean2 = g2 * diff2 / (safe_sigma2 ** 2)
        dg2_dsigma2 = g2 * (diff2 ** 2) / (safe_sigma2 ** 3) * np.sign(sigma2) if sigma2 != 0 else 0

        # partial derivatives of the product μ = g1 * g2
        # dμ/d_param1 = (dg1/d_param1) * g2
        # dμ/d_param2 = g1 * (dg2/d_param2)
        dmu_dmean1 = dg1_dmean1 * g2
        dmu_dsigma1 = dg1_dsigma1 * g2
        dmu_dmean2 = g1 * dg2_dmean2
        dmu_dsigma2 = g1 * dg2_dsigma2

        grad_mean1 = np.sum(delta_premise * dmu_dmean1) / batch_size
        grad_sigma1 = np.sum(delta_premise * dmu_dsigma1) / batch_size
        grad_mean2 = np.sum(delta_premise * dmu_dmean2) / batch_size
        grad_sigma2 = np.sum(delta_premise * dmu_dsigma2) / batch_size

        self.params[0] -= learning_rate * grad_mean1
        self.params[1] -= learning_rate * grad_sigma1
        self.params[2] -= learning_rate * grad_mean2
        self.params[3] -= learning_rate * grad_sigma2

        self.params[1] = np.maximum(np.abs(self.params[1]), self._epsilon)
        self.params[3] = np.maximum(np.abs(self.params[3]), self._epsilon)


class TrapezoidMembershipFunction(MembershipFunction):
    def __init__(self, params: list[float] | None = None):
        super().__init__(params=params)
        self.params = sorted(self.params)

    @property
    def num_params(self) -> int:
        return 4

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return trapmf(x, sorted(self.params))

    def update_params(self, x: np.ndarray, delta_premise: np.ndarray, learning_rate: float):
        a, b, c, d = sorted(self.params)
        batch_size = x.shape[0]
        if batch_size == 0: return

        # partial derivatives of the trapezoidal function:
        # f(x) = 0, x <= a
        # f(x) = (x-a)/(b-a), a <= x <= b
        # f(x) = 1, b <= x <= c
        # f(x) = (d-x)/(d-c), c <= x <= d
        # f(x) = 0, x >= d
        d_mu_da = np.zeros_like(x, dtype=float)
        d_mu_db = np.zeros_like(x, dtype=float)
        d_mu_dc = np.zeros_like(x, dtype=float)
        d_mu_dd = np.zeros_like(x, dtype=float)

        mask_ab = (x > a) & (x < b)
        mask_cd = (x > c) & (x < d)

        den_ba = b - a
        if den_ba > self._epsilon:
            mu_ab = (x[mask_ab] - a) / den_ba
            d_mu_da[mask_ab] = -mu_ab / den_ba
            d_mu_db[mask_ab] = -(x[mask_ab] - a) / (den_ba ** 2)

        den_dc = d - c
        if den_dc > self._epsilon:
            mu_cd = (d - x[mask_cd]) / den_dc
            d_mu_dc[mask_cd] = mu_cd / den_dc
            d_mu_dd[mask_cd] = (x - c) / (d - c)^2

        grad_a = np.sum(delta_premise * d_mu_da) / batch_size
        grad_b = np.sum(delta_premise * d_mu_db) / batch_size
        grad_c = np.sum(delta_premise * d_mu_dc) / batch_size
        grad_d = np.sum(delta_premise * d_mu_dd) / batch_size

        original_indices = np.argsort(np.argsort(self.params))
        grads = [grad_a, grad_b, grad_c, grad_d]

        for i in range(4):
             original_pos = np.where(original_indices == i)[0][0]
             self.params[original_pos] -= learning_rate * grads[i]

        self.params = sorted(self.params)


class TriangleMembershipFunction(MembershipFunction):
    def __init__(self, params: list[float] | None = None):
        super().__init__(params=params)
        self.params = sorted(list(self.params))

    @property
    def num_params(self) -> int:
        return 3

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return trimf(x, sorted(self.params))

    def update_params(self, x: np.ndarray, delta_premise: np.ndarray, learning_rate: float):
        a, b, c = sorted(self.params)
        batch_size = x.shape[0]
        if batch_size == 0: return

        d_mu_da = np.zeros_like(x, dtype=float)
        d_mu_db = np.zeros_like(x, dtype=float)
        d_mu_dc = np.zeros_like(x, dtype=float)

        mask_ab = (x > a) & (x < b)
        mask_bc = (x >= b) & (x < c)

        den_ba = b - a
        if den_ba > self._epsilon:
            d_mu_da[mask_ab] = (x[mask_ab] - b) / (den_ba ** 2)
            d_mu_db[mask_ab] = -(x[mask_ab] - a) / (den_ba ** 2)

        den_cb = c - b
        if den_cb > self._epsilon:
            d_mu_db[mask_bc] += (c - x[mask_bc]) / (den_cb ** 2)
            d_mu_dc[mask_bc] = (x[mask_bc] - b) / (den_cb ** 2)

        grad_a = np.sum(delta_premise * d_mu_da) / batch_size
        grad_b = np.sum(delta_premise * d_mu_db) / batch_size
        grad_c = np.sum(delta_premise * d_mu_dc) / batch_size

        original_indices = np.argsort(np.argsort(self.params))
        grads = [grad_a, grad_b, grad_c]
        for i in range(3):
             original_pos = np.where(original_indices == i)[0][0]
             self.params[original_pos] -= learning_rate * grads[i]

        self.params = sorted(self.params)


class SigmoidMembershipFunction(MembershipFunction):
    @property
    def num_params(self) -> int:
        return 2 # [c, gamma] according to skfuzzy

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # order for skfuzzy: center (c), slope (gamma)
        c, gamma = self.params
        return sigmf(x, c, gamma)

    def update_params(self, x: np.ndarray, delta_premise: np.ndarray, learning_rate: float):
        # f(x) = 1 / (1 + exp(-a * (x - c)))
        c, gamma = self.params
        batch_size = x.shape[0]
        if batch_size == 0: return

        # μ = 1 / (1 + exp(-gamma * (x - c)))
        membership = sigmf(x, c, gamma) # μ

        # Partial Derivatives dμ/d_param
        # dμ/dc = -gamma * exp(-gamma*(x-c)) * (-1) / (1 + exp(...))^2
        #       = gamma * (exp(-gamma*(x-c))/(1+exp(...))) * (1/(1+exp(...)))
        #       = gamma * (1 - μ) * μ
        d_mu_dc = gamma * membership * (1 - membership)

        # dμ/dgamma = -(x-c) * exp(-gamma*(x-c)) * (-1) / (1 + exp(...))^2
        #           = (x-c) * (exp(-gamma*(x-c))/(1+exp(...))) * (1/(1+exp(...)))
        #           = (x-c) * (1 - μ) * μ
        d_mu_dgamma = (x - c) * membership * (1 - membership)

        grad_c = np.sum(delta_premise * d_mu_dc) / batch_size
        grad_gamma = np.sum(delta_premise * d_mu_dgamma) / batch_size

        self.params[0] -= learning_rate * grad_c
        self.params[1] -= learning_rate * grad_gamma


class BellMembershipFunction(MembershipFunction):
    @property
    def num_params(self) -> int:
        return 3  # [a, b, c] where a is the width, b is the slope, c is the center

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        a, b, c = self.params
        safe_a = np.maximum(np.abs(a), self._epsilon)
        safe_b = np.maximum(b, self._epsilon)
        return gbellmf(x, safe_a, safe_b, c)

    def update_params(self, x: np.ndarray, delta_premise: np.ndarray, learning_rate: float):
        # f(x) = 1 / (1 + |((x - c) / a)|^(2b))
        a, b, c = self.params
        safe_a = np.maximum(np.abs(a), self._epsilon)
        safe_b = np.maximum(b, self._epsilon)
        batch_size = x.shape[0]
        if batch_size == 0: return

        # μ = 1 / (1 + |(x - c) / a|^(2b))
        membership = gbellmf(x, safe_a, safe_b, c) # μ

        common_term = membership * (1 - membership)

        # Termine |(x - c) / a|
        z_abs = np.abs((x - c) / safe_a)
        safe_log_z_abs = np.log(np.maximum(z_abs, self._epsilon))

        diff_x_c = x - c
        safe_diff_x_c = np.where(diff_x_c == 0, self._epsilon, diff_x_c)

        # Partial Derivatives dμ/d_param
        # dμ/da = (2b/a) * μ * (1 - μ) --- Multiplied by sign(a) if a is negative
        d_mu_da = (2 * safe_b / safe_a) * common_term * np.sign(a) if a != 0 else 0

        # dμ/db = -2 * μ * (1 - μ) * ln(|z|)
        d_mu_db = -2 * common_term * safe_log_z_abs

        d_mu_dc = (-2 * safe_b / safe_diff_x_c) * common_term

        grad_a = np.sum(delta_premise * d_mu_da) / batch_size
        grad_b = np.sum(delta_premise * d_mu_db) / batch_size
        grad_c = np.sum(delta_premise * d_mu_dc) / batch_size

        self.params[0] -= learning_rate * grad_a
        self.params[1] -= learning_rate * grad_b
        self.params[2] -= learning_rate * grad_c

        self.params[0] = np.maximum(np.abs(self.params[0]), self._epsilon)
        self.params[1] = np.maximum(self.params[1], self._epsilon)