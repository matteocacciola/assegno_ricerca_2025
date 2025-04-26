import time
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from services.factories import MembershipFunctionFactory


class ANFIS:
    def __init__(
        self,
        input_data: np.ndarray,
        output_data: np.ndarray,
        n_mfs: int,
        mf_type: str,
        membership_functions: List[List] | None = None
    ):
        """
        Initialize the ANFIS model.

        Parameters:
            input_data (numpy.ndarray): Training input data (features).
            output_data (numpy.ndarray): Output data (targets).
            n_mfs (int): Number of membership functions per input
            mf_type (str): Type of membership function to use.
            membership_functions (list[list[MembershipFunction]]): Membership functions for each input.
        """
        if n_mfs < 1:
            raise ValueError("Number of membership functions must be at least 1")

        self.input_data = input_data
        self.output_data = output_data
        _, n_inputs = input_data.shape
        self.n_mfs = n_mfs
        self.mf_type = mf_type
        self.n_rules = self.n_mfs ** n_inputs

        self.rule_indices = [self._decode_rule_index(n_inputs, rule_idx) for rule_idx in range(self.n_rules)]

        self.membership_functions = membership_functions
        if self._check_number_of_mfs(n_inputs) is False:
            self._init_mfs(n_inputs)

        self.rule_params = np.random.rand(self.n_rules, n_inputs + 1)  # a1...an, b

        self.errors = []
        self.predicted_values = []
        self.elapsed_time = 0

    def _check_number_of_mfs(self, n_inputs: int) -> bool:
        n_membership_functions = 0 if self.membership_functions is None else (
            len(self.membership_functions) * len(self.membership_functions[0])
        )
        return n_membership_functions == self.n_mfs * n_inputs

    def _init_mfs(self, n_inputs: int):
        factory = MembershipFunctionFactory()

        self.membership_functions = [
            [factory.create_membership_function(self.mf_type) for _ in range(self.n_mfs)]
            for _ in range(n_inputs)
        ]

    def override_mfs(self, membership_functions: List[List]) -> "ANFIS":
        """
        Override the membership functions for the ANFIS model.

        Args:
            membership_functions (list[list[MembershipFunction]]): Membership functions for each input.
        """
        old_mfs = self.membership_functions
        self.membership_functions = membership_functions
        n_inputs = len(membership_functions)

        if self._check_number_of_mfs(n_inputs) is False:
            self.membership_functions = old_mfs
            raise ValueError("The number of membership functions does not match the number of rules.")

        return self

    def _evaluate_memberships(self, inputs: np.ndarray) -> List[np.ndarray]:
        """Evaluate all membership functions over input X."""
        mf_values = []

        _, n_inputs = inputs.shape
        for i in range(n_inputs):
            mf_values_i = np.column_stack([
                self.membership_functions[i][j].evaluate(inputs[:, i])
                for j in range(self.n_mfs)
            ])
            mf_values.append(mf_values_i)
        return mf_values

    def _compute_rule_strengths(self, inputs: np.ndarray, mf_values: List[np.ndarray]) -> np.ndarray:
        """Compute firing strength for each rule."""
        n_samples, n_inputs = inputs.shape
        rule_strengths = np.ones((n_samples, self.n_rules))

        for rule_idx, indices in enumerate(self.rule_indices):  # ← reuse precomputed indices
            for i in range(n_inputs):
                rule_strengths[:, rule_idx] *= mf_values[i][:, indices[i]]

        return rule_strengths

    def _decode_rule_index(self, n_inputs: int, rule_idx: int) -> List[int]:
        """Decode rule index into MF indices for each input."""
        indices = []
        for _ in range(n_inputs):
            indices.append(rule_idx % self.n_mfs)
            rule_idx //= self.n_mfs
        return indices[::-1]

    def _normalize(self, w: np.ndarray) -> np.ndarray:
        """Normalize firing strengths."""
        w_sum = np.sum(w, axis=1, keepdims=True)
        w_sum = np.where(w_sum == 0, 1.0, w_sum)  # Avoid division by zero
        return w / w_sum

    def _compute_rule_outputs(self, inputs: np.ndarray) -> np.ndarray:
        """Compute rule consequent outputs."""
        n_samples, _ = inputs.shape

        outputs = np.zeros((n_samples, self.n_rules))
        for i in range(self.n_rules):
            linear_part = np.sum(self.rule_params[i, :-1] * inputs, axis=1)
            outputs[:, i] = linear_part + self.rule_params[i, -1]
        return outputs

    def forward_pass(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Forward pass to compute output."""
        mf_values = self._evaluate_memberships(inputs)
        w = self._compute_rule_strengths(inputs, mf_values)
        normalized_w = self._normalize(w)
        rule_outputs = self._compute_rule_outputs(inputs)

        output = np.sum(normalized_w * rule_outputs, axis=1)
        return output, normalized_w, mf_values

    def train(self, epochs: int = 1000, learning_rate: float = 0.01, tolerance: float = 1e-6):
        """
        Train the ANFIS model using hybrid learning: Backpropagation + The Least Squares.

        Args:
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for gradient descent.
            tolerance (float): Convergence criterion.
        """
        start_time = time.time()

        n_samples, n_inputs = self.input_data.shape
        for epoch in range(epochs):
            predicted_output, normalized_w, mf_values = self.forward_pass(self.input_data)
            error = self.output_data - predicted_output

            mean_error = np.mean(np.abs(error))
            self.errors.append(mean_error)
            self.predicted_values.append(predicted_output)

            if mean_error < tolerance:
                print(f"Converged at epoch {epoch}, Error: {mean_error}")
                break

            # Update consequent parameters via Least Squares
            for i in range(self.n_rules):
                w = normalized_w[:, i]
                inputs_aug = np.column_stack([self.input_data, np.ones(n_samples)])
                gradient = -2 * np.dot((error * w), inputs_aug) / n_samples
                self.rule_params[i] -= learning_rate * gradient

            # Update membership function parameters
            for i in range(n_inputs):
                for j in range(self.n_mfs):
                    self.membership_functions[i][j].update_params(self.input_data[:, i], error, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Error: {mean_error}")

        self.elapsed_time = time.time() - start_time

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predict outputs for new data.

        Args:
            inputs (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted output.
        """
        preds, _, _ = self.forward_pass(inputs)
        return preds

    def plot_errors(self):
        """Plot training error over epochs."""
        plt.plot(self.errors)
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Error")
        plt.title("Training Errors")
        plt.grid()
        plt.show()

    def plot_predictions(self):
        """Plot true vs predicted values."""
        preds = self.predict(self.input_data)
        plt.plot(self.output_data, label="Actual")
        plt.plot(preds, label="Predicted")
        plt.legend()
        plt.title("Actual vs Predicted")
        plt.xlabel("Samples")
        plt.ylabel("Output")
        plt.grid()
        plt.show()

    def plot_membership_functions(self):
        """Plot all membership functions."""
        _, n_inputs = self.input_data.shape
        for i in range(n_inputs):
            x = np.linspace(np.min(self.input_data[:, i]), np.max(self.input_data[:, i]), 100)
            for mf in self.membership_functions[i]:
                y = mf.evaluate(x)
                plt.plot(x, y)
            plt.title(f"Input {i + 1} Membership Functions")
            plt.xlabel("Input Value")
            plt.ylabel("Membership Degree")
            plt.grid()
            plt.show()
