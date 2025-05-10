import time
from typing import List, Tuple, Generator, Callable
import numpy as np
import matplotlib.pyplot as plt

from helpers import logs
from models import PredictionParser
from services.factories import MembershipFunctionFactory


class ANFIS:
    def __init__(
        self,
        n_inputs: int,
        n_mfs: int,
        mf_type: str,
        now: str,
        membership_functions: List[List] | None = None,
        prediction_parser: PredictionParser | None = None,
    ):
        """
        Initialize the ANFIS model.

        Parameters:
            n_inputs (int): Number of inputs.
            n_mfs (int): Number of membership functions per input.
            mf_type (str): Type of membership function to use.
            now (str): Current time string for logging.
            membership_functions (list[list[MembershipFunction]]): Membership functions for each input.
            prediction_parser (PredictionParser): Parser for predictions, with the function to apply and the number of classes.
        """
        if n_mfs < 1:
            raise ValueError("Number of membership functions must be at least 1")

        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.mf_type = mf_type
        self.n_rules = self.n_mfs ** n_inputs
        self.now = now

        self.prediction_parser = prediction_parser

        self.rule_indices = [self._decode_rule_index(n_inputs, rule_idx) for rule_idx in range(self.n_rules)]

        self.membership_functions = membership_functions
        if self._check_number_of_mfs(n_inputs) is False:
            self._init_mfs(n_inputs)

        self.rule_params = np.random.rand(self.n_rules, n_inputs + 1)  # a1...an, b

        self.errors_epoch = []
        self.min_input_data = None
        self.max_input_data = None
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
        """Decode the rule index into MF indices for each input."""
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
        """Forward pass to compute output and intermediate values needed for backprop."""
        mf_values_batch = self._evaluate_memberships(inputs)  # Output Layer 1
        w_batch = self._compute_rule_strengths(inputs, mf_values_batch)  # Output Layer 2
        normalized_w_batch = self._normalize(w_batch)  # Output Layer 3
        rule_outputs_batch = self._compute_rule_outputs(inputs)  # Output Layer 4

        predicted_output_batch = np.sum(normalized_w_batch * rule_outputs_batch, axis=1)  # Output Layer 5
        predicted_output_batch = np.nan_to_num(predicted_output_batch)  # NaN if w_sum was 0

        if self.prediction_parser is not None:
            predicted_output_batch = self.prediction_parser.parser(
                predicted_output_batch, self.prediction_parser.n_classes
            )

        return predicted_output_batch, normalized_w_batch, mf_values_batch

    def train(
        self,
        data_generator_factory: Callable[[], Generator[Tuple[np.ndarray, np.ndarray], None, None]],
        epochs: int,
        learning_rate_consequent: float,
        learning_rate_premise: float,
        batches_per_epoch: int,
        tolerance: float = 1e-6,
        min_improvement: float | None = None,
        patience: int | None = None,
    ) -> None:
        """
        Train the ANFIS model using mini-batch gradient descent with backpropagation.

        Args:
            data_generator_factory: A callable that returns a generator yielding batches of input and output data.
            epochs: Number of training epochs.
            learning_rate_consequent: Learning rate for consequent parameters (Layer 4).
            learning_rate_premise: Learning rate for premise parameters (Layer 1 MFs).
            batches_per_epoch: Number of batches per epoch (needed for reporting and error averaging).
            tolerance: Convergence criterion on the average error per epoch.
            min_improvement: Optional. If provided, specifies the minimum relative improvement in error required between epochs (as a fraction). Early stopping is disabled if None.
            patience: Optional. If min_improvement is provided, specifies the number of consecutive epochs without significant improvement before stopping.
        """
        start_time = time.time()
        self.errors_epoch = []  # Store absolute mean error for each epoch

        if min_improvement is not None and patience is None:
            raise ValueError("If `min_improvement` is specified, `patience` must also be specified")
        no_improvement_count = 0

        logs(f"nn_{self.now}", [f"Starting the training: {epochs} epochs..."])

        for epoch in range(epochs):
            epoch_batch_errors = []
            batch_count = 0

            logs(f"nn_{self.now}", [f"Epoch {epoch + 1}/{epochs}: Batch generation."])

            # Iterate over the batches provided by the generator for this epoch
            # The generator must be resettable or recreated for each epoch
            for input_batch, output_batch in data_generator_factory():
                batch_size = input_batch.shape[0]
                if batch_size == 0:
                    continue

                batch_min = np.min(input_batch, axis=0)
                batch_max = np.max(input_batch, axis=0)

                self.min_input_data = (
                    batch_min if self.min_input_data is None else np.minimum(self.min_input_data, batch_min)
                )
                self.max_input_data = (
                    batch_max if self.max_input_data is None else np.maximum(self.max_input_data, batch_max)
                )

                # --- Forward Pass over the Batch ---
                predicted_output_batch, normalized_w_batch, mf_values_batch = self.forward_pass(input_batch)

                # --- Error over the Batch ---
                error_batch = output_batch - predicted_output_batch
                epoch_batch_errors.extend(np.abs(error_batch))  # Collect errors of the del batch

                # --- Updating Parameters (Backward Pass over the Batch) ---

                # 1. Update Consequent Parameters (rule_params) - Mini-batch GD
                #    Your current implementation already computes a gradient. Adapt it
                #    to use the batch data and batch size.
                inputs_aug_batch = np.column_stack([input_batch, np.ones(batch_size)])
                for i in range(self.n_rules):
                    w = normalized_w_batch[:, i]
                    # Calculate the gradient for the current batch ONLY
                    gradient = -2 * np.dot((error_batch * w), inputs_aug_batch) / batch_size
                    self.rule_params[i] -= learning_rate_consequent * gradient

                # 2. Updating Membership Functions - Mini-batch GD
                #    This is the tricky part. Your MFs's `update_params`
                #    method MUST be modified to accept the batch data and the batch error
                #    and perform a gradient descent step based on the chain rule
                #    computed ONLY on that batch.
                for i in range(self.n_inputs):
                    for j in range(self.n_mfs):
                        self.membership_functions[i][j].update_params(
                            input_batch[:, i], error_batch, learning_rate_premise
                        )

                batch_count += 1
                logs(
                    f"nn_{self.now}",
                    [f" Epoch {epoch + 1}, Batch {batch_count}/{batches_per_epoch}: input_shape={input_batch.shape}, output_shape={output_batch.shape}"],
                )
                if batch_count >= batches_per_epoch:
                    break

            mean_epoch_error = float(np.mean(epoch_batch_errors))
            logs(f"nn_{self.now}", [f"Epoch {epoch + 1}/{epochs}, Absolute Mean Error: {mean_epoch_error:.6f}"])
            self.errors_epoch.append(mean_epoch_error)

            # Check for convergence based on an absolute error threshold
            if mean_epoch_error < tolerance:
                logs(f"nn_{self.now}", [f"Convergence reached at the epoch {epoch + 1}, Error: {mean_epoch_error:.6f}"])
                break

            # Check for improvement only if min_improvement was specified
            if min_improvement is not None and epoch > 0:
                relative_improvement = (self.errors_epoch[-2] - mean_epoch_error) / self.errors_epoch[-2]

                if relative_improvement < min_improvement:
                    no_improvement_count += 1
                    logs(
                        f"nn_{self.now}",
                        [f"Insufficient improvement: {relative_improvement:.6f} < {min_improvement}. No improvement count: {no_improvement_count}/{patience}"]
                    )

                    if no_improvement_count >= patience:
                        logs(
                            f"nn_{self.now}",
                            [f"Early stopping after {epoch + 1} epochs due to {patience} consecutive epochs without significant improvement"]
                        )
                        break
                else:
                    # Reset counter if there was a significant improvement
                    no_improvement_count = 0
                    logs(
                        f"nn_{self.now}",
                        [f"Significant improvement: {relative_improvement:.6f} >= {min_improvement}. Resetting counter."]
                    )

        self.elapsed_time = time.time() - start_time
        logs(f"nn_{self.now}", [f"Training completed in {self.elapsed_time:.2f} seconds."])

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

    def plot_errors(self, save_path: str):
        """Plot training error over epochs."""
        plt.plot(self.errors_epoch)
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Error")
        plt.title("Training Errors")
        plt.grid()
        plt.show()
        plt.savefig(save_path)

    def plot_predictions(self, input_data: np.ndarray, output_data: np.ndarray, save_path: str):
        """Plot true vs predicted values."""
        preds = self.predict(input_data)
        plt.plot(output_data, label="Actual")
        plt.plot(preds, label="Predicted")
        plt.legend()
        plt.title("Actual vs Predicted")
        plt.xlabel("Samples")
        plt.ylabel("Output")
        plt.grid()
        plt.show()
        plt.savefig(save_path)

    def plot_membership_functions(self, save_path: str):
        """Plot all membership functions."""
        # create a multiplot
        plt.suptitle("ANFIS Membership Functions")
        for i in range(self.n_inputs):
            plt.subplot(self.n_inputs, 1, i + 1)
            x = np.linspace(self.min_input_data[i], self.max_input_data[i], 100)
            for mf in self.membership_functions[i]:
                y = mf.evaluate(x)
                plt.plot(x, y)
            plt.title(f"ANFIS Input {i + 1} Membership Functions")
            plt.xlabel("Input Feature")
            plt.ylabel("Membership Degree")
            plt.grid()
        plt.show()
        plt.savefig(save_path)
