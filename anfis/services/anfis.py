import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from services.factories import MembershipFunctionFactory, FuzzificationFactory


class ANFIS:
    def __init__(
        self,
        input_data: np.ndarray,
        output_data: np.ndarray,
        n_rules: int,
        n_inputs: int,
        mf_type: str,
        membership_functions: List[List] | None = None
    ):
        """
        Initialize the ANFIS model.

        Parameters:
            input_data (numpy.ndarray): Training input data (features).
            output_data (numpy.ndarray): Output data (targets).
            n_rules (int): Number of fuzzy rules.
            n_inputs (int): Number of input features.
            mf_type (str): Type of membership function to use.
            membership_functions (list[list[MembershipFunction]]): Membership functions for each input.
        """
        if n_rules < 1:
            raise ValueError("Number of rules must be at least 1")

        if n_inputs < 1:
            raise ValueError("Number of inputs must be at least 1")

        self.input_data = input_data
        self.output_data = output_data
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.mf_type = mf_type
        self.membership_functions = membership_functions

        n_membership_functions = 0 if self.membership_functions is None else (
                len(self.membership_functions) * len(self.membership_functions[0])
        )
        if n_membership_functions != self.n_rules * self.n_inputs:
            self.init_membership_functions()

        self.rule_params = np.random.rand(self.n_rules, 2)  # Linear rule parameters [a, b] for output: z = a*x + b

        self.errors = []
        self.predicted_values = []
        self.elapsed_time = 0

    def init_membership_functions(self):
        factory = MembershipFunctionFactory()
        self.membership_functions = [
            [factory.create_membership_function(self.mf_type) for _ in range(self.n_rules)] for _ in range(self.n_inputs)
        ]

    def fuzzification(self, inputs: np.ndarray) -> np.ndarray:
        """
        Fuzzify the inputs using Gaussian membership functions.

        Args:
            inputs (numpy.ndarray): Input data to be fuzzified.

        Returns:
            numpy.ndarray: Fuzzified inputs.
        """
        factory = FuzzificationFactory()

        fuzzified_inputs = [factory.create_fuzzifier(self.mf_type).fuzzify(inputs[:, i]) for i in range(self.n_inputs)]
        return np.array(fuzzified_inputs).T

    def rule_evaluation(self, fuzzified_inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the rule firing strengths based on fuzzified inputs.

        Args:
            fuzzified_inputs (numpy.ndarray): Fuzzified inputs.

        Returns:
            numpy.ndarray: Rule firing strengths.
        """
        w = []
        for i in range(self.n_rules):
            firing_strength = 1
            for j in range(self.n_inputs):
                firing_strength *= self.membership_functions[i][j].evaluate(fuzzified_inputs[:, j])
            w.append(firing_strength)
        return np.array(w)

    def normalize(self, w: np.ndarray) -> np.ndarray:
        """
        Normalize the firing strengths of the rules.

        Args:
            w (numpy.ndarray): Rule firing strengths.

        Returns:
            numpy.ndarray: Normalized rule firing strengths.
        """
        return w / np.sum(w)

    def defuzzify(self, normalized_w: np.ndarray, fuzzified_inputs: np.ndarray) -> np.ndarray:
        """
        Defuzzify the output using the weighted sum of rule consequents.

        Args:
            normalized_w (numpy.ndarray): Normalized rule firing strengths.
            fuzzified_inputs (numpy.ndarray): Fuzzified inputs.

        Returns:
            numpy.ndarray: Defuzzified output.
        """
        output = 0
        for i in range(self.n_rules):
            a, b = self.rule_params[i]  # Linear rule parameters
            output += normalized_w[i] * (a * fuzzified_inputs[:, 0] + b)
        return output

    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network to compute the output.

        Args:
            inputs (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted output.
        """
        fuzzified_inputs = self.fuzzification(inputs)
        w = self.rule_evaluation(fuzzified_inputs)
        normalized_w = self.normalize(w)
        return self.defuzzify(normalized_w, fuzzified_inputs)

    def train(self, epochs: int | None = 1000, learning_rate: float | None = 0.01, tolerance: float | None = 1e-6):
        """
        Train the ANFIS model using hybrid learning: Backpropagation + Least Squares.

        Args:
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for gradient descent.
            tolerance (float): Convergence criterion.
        """
        start_time = time.time()
        for epoch in range(epochs):
            predicted_output = self.forward_pass(self.input_data)
            error = self.output_data - predicted_output

            mean_error = np.mean(np.abs(error))
            np.append(self.errors, mean_error)
            np.append(self.predicted_values, predicted_output)
            if mean_error < tolerance:
                self.elapsed_time = time.time() - start_time
                print(f'Converged at epoch {epoch}, Error: {np.mean(np.abs(error))}')
                break

            for i in range(self.n_rules):
                # Update rule parameters using Least Squares Estimation (LSE)
                rule_output = self.rule_params[i, 0] * self.input_data[:, 0] + self.rule_params[i, 1]
                self.rule_params[i, 0] -= learning_rate * np.sum(error * rule_output)
                self.rule_params[i, 1] -= learning_rate * np.sum(error)

                # Update membership function parameters using gradient descent
                for j in range(self.n_inputs):
                    self.membership_functions[j][i].update_params(self.input_data[:, j], error, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            inputs (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted output.
        """
        return self.forward_pass(inputs)

    def plot_errors(self):
        """Plot the training errors."""
        plt.plot(self.errors)
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Error")
        plt.title("Training Errors")
        plt.show()

    def plot_predictions(self):
        """Plot the predicted values."""
        plt.plot(self.output_data, label="Actual")
        plt.plot(self.predicted_values, label="Predicted")
        plt.xlabel("Samples")
        plt.ylabel("Values")
        plt.title("Actual vs Predicted Values")
        plt.legend()
        plt.show()

    def plot_membership_functions(self):
        """Plot the membership functions."""
        for i in range(self.n_inputs):
            for j in range(self.n_rules):
                mf = self.membership_functions[i][j]
                x = np.linspace(np.min(self.input_data[:, i]), np.max(self.input_data[:, i]), 100)
                y = mf.evaluate(x)
                plt.plot(x, y, label=f"MF {i+1}, Rule {j+1}")
        plt.xlabel("Input")
        plt.ylabel("Membership Value")
        plt.title("Membership Functions")
        plt.legend()
        plt.show()
