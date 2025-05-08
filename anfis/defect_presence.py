import json
import math
import os
from argparse import ArgumentParser
from typing import Any, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import ConfigDict
from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from pyvolutionary import (
    best_agent,
    ContinuousMultiVariable,
    ParticleSwarmOptimization,
    ParticleSwarmOptimizationConfig,
    Task,
)
import time
import gc
import pyarrow.parquet as pq

from helpers import load_csv_data, prepare_predictions, logs, reduce_dataset, get_test_data, predict_chunks
from models import MembershipFunctionType, FractionData
from services import MembershipFunctionFactory, ANFIS

factory = MembershipFunctionFactory()

file_type = "csv"  # parquet

train_file_path = f"datasets/defect_presence/db_train.{file_type}"
train_metadata_path = "datasets/defect_presence/db_train.json"
test_file_path = f"datasets/defect_presence/db_test.{file_type}"


class ANFISOptimizedProblem(Task):
    n_mfs: int
    mf_type: str
    anfis_model: Any
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    chunk_size: int

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def objective_function(self, x: list[Any]):
        n_inputs = self.X_test.shape[1]
        x_transformed = self.transform_solution(x)
        logs("ec", [f"Run objective function with x: {x_transformed}"])
        mfs = [
            [
                factory.create_membership_function(
                    self.mf_type,
                    params=[
                        x_transformed[f"x_{i * self.n_mfs + j}"][0],
                        x_transformed[f"x_{i * self.n_mfs + j}"][1]
                    ]
                ) for j in range(self.n_mfs)
            ] for i in range(n_inputs)
        ]

        self.anfis_model.override_mfs(mfs)

        y_pred_real = predict_chunks(self.anfis_model, self.X_test, self.chunk_size)
        y_pred_ready = prepare_predictions(y_pred_real, n_classes=self.n_classes)
        logs("ec", ["Prediction finished"])

        return metrics.accuracy_score(self.y_test, y_pred_ready)


# Simulate the ANFIS model using the Feedforward Backpropagation
def simulate_by_nn(
    X_test: np.ndarray, n_inputs: int, n_mfs: int, chunk_size: int, batches_per_epoch: int
) -> Tuple[np.ndarray, float, Any]:
    def create_data_generator(file_type_: str):
        def generator_parquet():
            parquet_file = pq.ParquetFile(train_file_path)
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                batch_df = batch.to_pandas().astype(np.float32).values
                input_batch = batch_df[:, :-1]
                output_batch = batch_df[:, -1]
                yield input_batch, output_batch
        def generator_csv():
            for batch_df in pd.read_csv(train_file_path, chunksize=chunk_size, dtype=np.float32):
                input_batch = batch_df.iloc[:, :-1].values
                output_batch = batch_df.iloc[:, -1].values.flatten()
                yield input_batch, output_batch

        return generator_csv if file_type_ == "csv" else generator_parquet

    # Create the ANFIS model
    anfis_model = ANFIS(n_inputs, n_mfs, str(MembershipFunctionType.GAUSSIAN))

    data_generator_factory = create_data_generator(file_type)
    anfis_model.train(
        data_generator_factory(),
        epochs=1000,
        learning_rate_consequent=0.005,
        learning_rate_premise=0.001,
        batches_per_epoch=batches_per_epoch,
        tolerance=1e-5
    )

    logs("nn", ["Prediction started"])
    # predict the test data, in chunks
    y_pred = predict_chunks(anfis_model, X_test, chunk_size)
    logs("nn", ["Prediction finished"])

    return y_pred, anfis_model.elapsed_time, anfis_model.errors_epoch


# Simulate the ANFIS model using Evolutionary Computation
# You can replace the PSO with any other algorithm implemented in the library.
def simulate_by_ec(
    X_test: np.ndarray, y_test: np.ndarray,n_inputs: int, n_mfs: int, mf_type: str, chunk_size: int
) -> Tuple[np.ndarray, float, Any]:
    logs("ec", ["Training started..."])

    anfis_model = ANFIS(n_inputs, n_mfs, mf_type)

    # we have two parameters for each membership function: the mean and the standard deviation
    task = ANFISOptimizedProblem(
        n_mfs=n_mfs,
        mf_type=str(MembershipFunctionType.GAUSSIAN),
        anfis_model=anfis_model,
        X_test=X_test,
        y_test=y_test,
        n_classes=len(np.unique(y_test)),
        chunk_size=chunk_size,
        variables=[
            ContinuousMultiVariable(lower_bounds=[0.01, 10], upper_bounds=[1, 100], name=f"x_{i}")
            for i in range(n_mfs * n_inputs)
        ],
        minmax="max",
    )

    configuration = ParticleSwarmOptimizationConfig(
        population_size=200,
        fitness_error=10e-4,
        max_cycles=100,
        c1=0.1,
        c2=0.1,
        w=[0.35, 1],
    )

    start_time = time.time()
    result = ParticleSwarmOptimization(configuration, debug=True).optimize(task)
    elapsed_time = time.time() - start_time

    logs("ec", ["Training finished."])

    best = best_agent(result.evolution[-1].agents, task.minmax)
    best_params = task.transform_solution(best.position)

    logs("ec", [
        "Evolutionary algorithm: Particle Swarm Optimization"
        f"Best parameters: {best_params}",
        f"Best accuracy: {best.cost}"
    ])

    best_params = list(best_params.values())

    anfis_model.override_mfs([
        [
            factory.create_membership_function(
                mf_type, params=[best_params[i * n_mfs + j][0], best_params[i * n_mfs + j][1]]
            ) for j in range(n_mfs)
        ] for i in range(n_inputs)
    ])

    logs("ec", ["Prediction started"])
    y_predict = predict_chunks(anfis_model, X_test, chunk_size)
    logs("ec", ["Prediction finished"])

    return y_predict, elapsed_time, result


def prepare_db():
    dataset = load_csv_data(
        "datasets/defect_presence",
        fraction_data=FractionData(fraction=args.fraction, fraction_callback=reduce_dataset)
    ).to_numpy()

    x = dataset[:, :-1]
    y = dataset[:, -1]
    y = y.reshape(-1, 1)
    del dataset
    gc.collect()

    print(f"Loaded {x.shape[0]} samples with {x.shape[1]} features.")

    # Scaling
    x_scaled = StandardScaler().fit_transform(x)
    print(f"Scaled data shape: {x_scaled.shape}")

    del x
    gc.collect()

    # Perform PCA to reduce dimensionality
    # x_pca = PCA(n_components=0.95, svd_solver="full").fit_transform(x_scaled)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

    n_rows, n_inputs = X_train.shape  # Number of samples and number of input features
    chunk_size = math.ceil(0.1 * n_rows)  # Number of samples per batch
    batches_per_epoch = math.ceil(n_rows / chunk_size)
    metadata = {
        "n_rows": n_rows,
        "n_inputs": n_inputs,
        "n_outputs": y_test.shape[1],
        "chunk_size": chunk_size,
        "batches_per_epoch": batches_per_epoch,
        "n_classes": len(np.unique(y)),
        "n_mfs": 3,  # Number of membership functions per input
    }

    del x_scaled, y
    gc.collect()

    df_train = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)).astype(np.float32)
    if file_type == "csv":
        df_train.to_csv(train_file_path, index=False)
    else:
        df_train.to_parquet(train_file_path, index=False)
    del df_train, X_train, y_train
    gc.collect()

    df_test = pd.DataFrame(np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)).astype(np.float32)
    if file_type == "csv":
        df_test.to_csv(test_file_path, index=False)
    else:
        df_test.to_parquet(test_file_path, index=False)
    del df_test, X_test, y_test
    gc.collect()

    with open(train_metadata_path, "w") as f:
        json.dump(metadata, f)

    print(f"Saved training and testing data to {'CSV' if file_type == 'csv' else 'Parquet'} files.")


def parse_arguments():
    parser = ArgumentParser()

    # argument for the solver; it can be either 'nn' or 'ec'
    parser.add_argument(
        "-s",
        "--solver",
        default="nn",
        help="Solver to use for the optimization problem.",
    )

    parser.add_argument(
        "-f",
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use for training and testing.",
    )

    args_ = parser.parse_args()
    if args_.solver not in ["nn", "ec"]:
        raise ValueError("Invalid solver. Use 'nn' or 'ec'.")

    return args_


def main():
    os.makedirs("results", exist_ok=True)

    with open(train_metadata_path, "r") as f:
        train_metadata = json.load(f)

    if file_type not in ["csv", "parquet"]:
        raise ValueError("Invalid file type. Use 'csv' or 'parquet'.")

    X_test, y_test = get_test_data(file_type, test_file_path)

    if args.solver == "nn":
        y_predict, elapsed_time, errors = simulate_by_nn(
            X_test,
            train_metadata["n_inputs"],
            train_metadata["n_mfs"],
            train_metadata["chunk_size"],
            train_metadata["batches_per_epoch"],
        )
    else:
        y_predict, elapsed_time, result = simulate_by_ec(
            X_test,
            y_test,
            train_metadata["n_inputs"],
            train_metadata["n_mfs"],
            str(MembershipFunctionType.GAUSSIAN),
            train_metadata["chunk_size"],
        )
        errors = result.rates

    logs(args.solver, [
        f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_predict)}",
        f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_predict)}",
        f"Root Mean Squared Error: {metrics.root_mean_squared_error(y_test, y_predict)}",
        f"R2 Score: {metrics.r2_score(y_test, y_predict)}",
        f"Time to Convergence: {len(errors)}",
        f"Final Error: {errors[-1]}",
        f"Elapsed Time: {elapsed_time}",
    ])

    # Plot the results
    plt.scatter(y_test, y_predict, color="blue")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linewidth=2)
    plt.xlabel("True Defect Category")
    plt.ylabel("Predicted Defect Category")
    plt.title("ANFIS Regression - True vs Predicted Defect Categories")
    plt.savefig(f"results/anfis_results_{args.solver}.png")


if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists("datasets/defect_presence/db_train.json"):
        # Prepare the database
        # This is a placeholder for the actual database preparation code
        print("Preparing the database...")
        prepare_db()
        print(f"Database prepared and stored into `{train_file_path}` and `{test_file_path}` files.")

    main()
