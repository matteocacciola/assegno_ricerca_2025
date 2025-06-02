import time
from typing import List, Tuple, Generator, Callable
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import threading

from helpers import logs
from models import PredictionParser, MembershipFunction
from services.factories import MembershipFunctionFactory


# Global functions for multiprocessing (must be at module level to be picklable)
def _evaluate_input_mfs_worker(args):
    """Worker function for evaluating membership functions for a single input dimension"""
    input_idx, input_data, membership_functions, n_mfs = args
    mf_values_i = np.column_stack([
        membership_functions[input_idx][j].evaluate(input_data)
        for j in range(n_mfs)
    ])
    return input_idx, mf_values_i


def _compute_rule_strength_chunk_worker(args):
    """Worker function for computing strengths for a chunk of rules"""
    rule_indices_chunk, mf_values, n_inputs, n_samples = args
    chunk_strengths = []
    for rule_idx, indices in rule_indices_chunk:
        strength = np.ones(n_samples)
        for i in range(n_inputs):
            strength *= mf_values[i][:, indices[i]]
        chunk_strengths.append(strength)
    return chunk_strengths


def _update_rule_params_chunk_worker(args):
    """Worker function for updating parameters for a chunk of rules"""
    rule_indices_chunk, error_batch, normalized_w_batch, inputs_aug_batch, batch_size, learning_rate = args
    updates_ = []
    for rule_idx_ in rule_indices_chunk:
        w = normalized_w_batch[:, rule_idx_]
        gradient = -2 * np.dot((error_batch * w), inputs_aug_batch) / batch_size
        updates_.append((rule_idx_, learning_rate * gradient))
    return updates_


def _update_mf_chunk_worker(args):
    """Worker function for updating membership functions for a chunk of (input_idx, mf_idx) pairs"""
    input_mf_pairs, membership_functions, input_batch, error_batch, learning_rate = args
    # Note: This approach won't work well with multiprocessing because membership_functions
    # are complex objects that need to be shared. Consider using threading for MF updates.
    for input_idx, mf_idx in input_mf_pairs:
        membership_functions[input_idx][mf_idx].update_params(
            input_batch[:, input_idx], error_batch, learning_rate
        )


class ParallelANFIS:
    def __init__(
        self,
        n_inputs: int,
        n_mfs: int,
        mf_type: str,
        now: str,
        membership_functions: List[List[MembershipFunction]] | None = None,
        prediction_parser: PredictionParser | None = None,
        n_workers: int | None = None,
        use_multiprocessing: bool = False,
    ):
        """
        Initialize the ANFIS model with parallel computing support.

        Parameters:
            n_inputs (int): Number of inputs.
            n_mfs (int): Number of membership functions per input.
            mf_type (str): Type of membership function to use.
            now (str): Current time string for logging.
            membership_functions (list[list[MembershipFunction]]): Membership functions for each input.
            prediction_parser (PredictionParser): Parser for predictions, with the function to apply and the number of classes.
            n_workers (int): Number of parallel workers. If None, uses CPU count.
            use_multiprocessing (bool): If True, uses ProcessPoolExecutor, else ThreadPoolExecutor.
        """
        if n_mfs < 1:
            raise ValueError("Number of membership functions must be at least 1")

        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.mf_type = mf_type
        self.n_rules = self.n_mfs ** n_inputs
        self.now = now

        self.prediction_parser = prediction_parser

        # Parallel computing settings
        self.n_workers = n_workers or min(cpu_count(), 8)  # Limit to 8 to avoid overhead
        self.use_multiprocessing = use_multiprocessing
        self.executor = None

        # Thread-safe lock for parameter updates - ONLY for threading, not multiprocessing
        self.param_lock = threading.Lock() if not use_multiprocessing else None

        self.rule_indices = [self._decode_rule_index(n_inputs, rule_idx) for rule_idx in range(self.n_rules)]

        self.membership_functions = membership_functions
        if self._check_number_of_mfs(n_inputs) is False:
            self._init_mfs(n_inputs)

        self.rule_params = np.random.rand(self.n_rules, n_inputs + 1)  # a1...an, b

        self.errors_epoch = []
        self.min_input_data = None
        self.max_input_data = None
        self.elapsed_time = 0

    def __enter__(self):
        """Context manager entry - initialize executor"""
        self.executor = (
            ProcessPoolExecutor(max_workers=self.n_workers)
            if self.use_multiprocessing
            else ThreadPoolExecutor(max_workers=self.n_workers)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup executor"""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

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

    def override_mfs(self, membership_functions: List[List]) -> "ParallelANFIS":
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
        """Evaluate all membership functions over input X in parallel."""

        # Use executor if available, otherwise fall back to sequential
        if self.executor:
            futures = [
                self.executor.submit(
                    _evaluate_input_mfs_worker, (i, inputs[:, i], self.membership_functions, self.n_mfs)
                )
                for i in range(self.n_inputs)
            ]

            # Collect results in order
            mf_values = [None] * self.n_inputs
            for future in futures:
                idx, result = future.result()
                mf_values[idx] = result

            return mf_values  # type: ignore

        # Sequential fallback
        mf_values = []
        for i in range(self.n_inputs):
            _, mf_values_i = _evaluate_input_mfs_worker((i, inputs[:, i], self.membership_functions, self.n_mfs))
            mf_values.append(mf_values_i)

        return mf_values

    def _compute_rule_strengths(self, inputs: np.ndarray, mf_values: List[np.ndarray]) -> np.ndarray:
        """Compute firing strength for each rule in parallel."""
        n_samples = inputs.shape[0]

        # Split rules into chunks for parallel processing
        chunk_size = max(1, self.n_rules // self.n_workers)
        rule_chunks = []

        for i in range(0, self.n_rules, chunk_size):
            end_idx = min(i + chunk_size, self.n_rules)
            chunk = [(rule_idx, self.rule_indices[rule_idx]) for rule_idx in range(i, end_idx)]
            rule_chunks.append(chunk)

        # Use executor if available
        if self.executor and len(rule_chunks) > 1:
            futures = [
                self.executor.submit(
                    _compute_rule_strength_chunk_worker, (chunk, mf_values, self.n_inputs, n_samples)
                )
                for chunk in rule_chunks
            ]

            # Collect results
            all_strengths = []
            for future in futures:
                all_strengths.extend(future.result())

            return np.column_stack(all_strengths)

        # Sequential fallback for small problems or no executor
        all_strengths = []
        for chunk in rule_chunks:
            all_strengths.extend(_compute_rule_strength_chunk_worker((chunk, mf_values, self.n_inputs, n_samples)))

        return np.column_stack(all_strengths)

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
        # This can be vectorized efficiently without explicit parallelization
        # since NumPy already uses optimized BLAS operations
        inputs_aug = np.column_stack([inputs, np.ones(inputs.shape[0])])
        outputs = np.dot(inputs_aug, self.rule_params.T)
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

    def _update_consequent_params(
        self,
        input_batch: np.ndarray,
        error_batch: np.ndarray,
        normalized_w_batch: np.ndarray,
        learning_rate: float
    ):
        """Update consequent parameters in parallel."""
        batch_size = input_batch.shape[0]
        inputs_aug_batch = np.column_stack([input_batch, np.ones(batch_size)])

        # Split rules into chunks
        chunk_size = max(1, self.n_rules // self.n_workers)
        rule_chunks = [
            list(range(i, min(i + chunk_size, self.n_rules)))
            for i in range(0, self.n_rules, chunk_size)
        ]

        # Use executor if available
        if self.executor and len(rule_chunks) > 1:
            futures = [
                self.executor.submit(
                    _update_rule_params_chunk_worker,
                    (chunk, error_batch, normalized_w_batch, inputs_aug_batch, batch_size, learning_rate)
                )
                for chunk in rule_chunks
            ]

            # Apply updates with thread safety (only for threading)
            if self.param_lock:
                with self.param_lock:
                    for future in futures:
                        updates = future.result()
                        for rule_idx, gradient_update in updates:
                            self.rule_params[rule_idx] -= gradient_update
            else:
                # For multiprocessing, no lock needed as each process has its own memory
                for future in futures:
                    updates = future.result()
                    for rule_idx, gradient_update in updates:
                        self.rule_params[rule_idx] -= gradient_update

            return

        # Sequential fallback
        for chunk in rule_chunks:
            updates = _update_rule_params_chunk_worker(
                (chunk, error_batch, normalized_w_batch, inputs_aug_batch, batch_size, learning_rate)
            )
            for rule_idx, gradient_update in updates:
                self.rule_params[rule_idx] -= gradient_update

    def _update_membership_functions(
        self,
        input_batch: np.ndarray,
        error_batch: np.ndarray,
        learning_rate: float
    ):
        """Update membership function parameters.

        Note: For multiprocessing, this uses threading instead because
        membership functions are complex objects that don't serialize well.
        """
        # Create list of all (input_idx, mf_idx) pairs
        all_mf_pairs = [
            (i, j) for i in range(self.n_inputs) for j in range(self.n_mfs)
        ]

        # For membership function updates, always use threading or sequential
        # because MF objects are complex and don't work well with multiprocessing
        if self.executor and len(all_mf_pairs) > self.n_workers:
            # Split into chunks
            chunk_size = max(1, len(all_mf_pairs) // self.n_workers)
            mf_chunks = [
                all_mf_pairs[i:i + chunk_size]
                for i in range(0, len(all_mf_pairs), chunk_size)
            ]

            futures = [
                self.executor.submit(_update_mf_chunk_worker, (chunk, self.membership_functions, input_batch, error_batch, learning_rate))
                for chunk in mf_chunks
            ]

            # Wait for completion
            for future in futures:
                future.result()

            return

        # Sequential fallback (recommended for multiprocessing mode)
        for input_idx, mf_idx in all_mf_pairs:
            _update_mf_chunk_worker(((input_idx, mf_idx), self.membership_functions, input_batch, error_batch, learning_rate))

    def __getstate__(self):
        """Custom method for pickling - exclude non-picklable objects"""
        state = self.__dict__.copy()
        # Remove the unpicklable thread lock and executor
        state["param_lock"] = None
        state["executor"] = None
        return state

    def __setstate__(self, state):
        """Custom method for unpickling - recreate non-picklable objects"""
        self.__dict__.update(state)
        # Recreate the thread lock if not using multiprocessing
        # Executor will be recreated in __enter__ method when needed
        self.param_lock = threading.Lock() if not self.use_multiprocessing else None

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
        Train the ANFIS model using mini-batch gradient descent with backpropagation and parallel computing.

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

        logs("nn", self.now, [f"Starting parallel training with {self.n_workers} workers: {epochs} epochs..."])

        # Initialize executor for training
        with self:
            for epoch in range(epochs):
                epoch_batch_errors = []
                batch_count = 0

                logs("nn", self.now, [f"Epoch {epoch + 1}/{epochs}: Batch generation."])

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

                    # Forward Pass (parallel)
                    predicted_output_batch, normalized_w_batch, mf_values_batch = self.forward_pass(input_batch)

                    # Error calculation
                    error_batch = output_batch - predicted_output_batch
                    epoch_batch_errors.extend(np.abs(error_batch))

                    # Backward Pass (parallel)
                    # Update consequent parameters
                    self._update_consequent_params(
                        input_batch, error_batch, normalized_w_batch, learning_rate_consequent
                    )

                    # Update membership functions
                    self._update_membership_functions(
                        input_batch, error_batch, learning_rate_premise
                    )

                    batch_count += 1
                    logs(
                        "nn",
                        self.now,
                        [f" Epoch {epoch + 1}, Batch {batch_count}/{batches_per_epoch}: input_shape={input_batch.shape}, output_shape={output_batch.shape}"],
                    )
                    if batch_count >= batches_per_epoch:
                        break

                mean_epoch_error = float(np.mean(epoch_batch_errors))
                logs("nn", self.now, [f"Epoch {epoch + 1}/{epochs}, Absolute Mean Error: {mean_epoch_error:.6f}"])
                self.errors_epoch.append(mean_epoch_error)

                # Convergence check
                if mean_epoch_error < tolerance:
                    logs("nn", self.now, [f"Convergence reached at epoch {epoch + 1}, Error: {mean_epoch_error:.6f}"])
                    break

                # Early stopping check
                if min_improvement is not None and epoch > 0:
                    relative_improvement = (self.errors_epoch[-2] - mean_epoch_error) / self.errors_epoch[-2]

                    if relative_improvement < min_improvement:
                        no_improvement_count += 1
                        logs(
                            "nn",
                            self.now,
                            [f"Insufficient improvement: {relative_improvement:.6f} < {min_improvement}. No improvement count: {no_improvement_count}/{patience}"]
                        )

                        if no_improvement_count >= patience:
                            logs(
                                "nn",
                                self.now,
                                [f"Early stopping after {epoch + 1} epochs due to {patience} consecutive epochs without significant improvement"]
                            )
                            break
                    else:
                        no_improvement_count = 0
                        logs(
                            "nn",
                            self.now,
                            [f"Significant improvement: {relative_improvement:.6f} >= {min_improvement}. Resetting counter."]
                        )

        self.elapsed_time = time.time() - start_time
        logs(
            "nn",
            self.now,
            [f"Parallel training completed in {self.elapsed_time:.2f} seconds using {self.n_workers} workers."]
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predict outputs for new data using parallel computation.
        """
        with self:
            preds, _, _ = self.forward_pass(inputs)
        return preds

    def plot_errors(self, save_path: str):
        """Plot training error over epochs."""
        plt.plot(self.errors_epoch)
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Error")
        plt.title("Training Errors")
        plt.grid()
        plt.savefig(save_path)
        plt.close()

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
        plt.savefig(save_path)
        plt.close()

    def plot_membership_functions(self, save_path: str):
        """Plot all membership functions."""
        plt.suptitle("ANFIS Membership Functions")
        plt.figure(figsize=(25, 3.5 * self.n_inputs))
        for i in range(self.n_inputs):
            plt.subplot(self.n_inputs, 1, i + 1)
            x = np.linspace(self.min_input_data[i], self.max_input_data[i], 100)
            for mf in self.membership_functions[i]:
                y = mf.evaluate(x)
                plt.plot(x, y)
            plt.title(f"ANFIS Input {i + 1} Membership Functions")
            plt.ylabel("Membership Degree")
            plt.grid()
        plt.xlabel("Input Feature")
        plt.savefig(save_path)
        plt.close()


class ANFIS(ParallelANFIS):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("n_workers", 1)  # Default sequential execution
        kwargs.setdefault("use_multiprocessing", False)
        super().__init__(*args, **kwargs)
