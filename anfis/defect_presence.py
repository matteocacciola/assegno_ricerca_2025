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
    EarlyStopping,
)
import time
import gc
import pyarrow.parquet as pq

from helpers import (
    load_csv_data,
    prepare_predictions,
    logs,
    get_data,
    predict_chunks,
    plot_confusion_matrix,
    save_anfis_model,
    save_results,
)
from models import MembershipFunctionType, FractionData, PredictionParser
from services import MembershipFunctionFactory, ANFIS

mf_factory = MembershipFunctionFactory()
mf_type = str(MembershipFunctionType.GAUSSIAN)

file_type = "csv"  # parquet

train_file_path = f"datasets/defect_presence/db_train.{file_type}"
train_metadata_path = "datasets/defect_presence/db_train.json"
test_file_path = f"datasets/defect_presence/db_test.{file_type}"

now = time.strftime("%Y%m%d%H%M%S")


def plot_errors(log_file_path: str):
    # open the file `log_file_path`
    # read the file and search the lines with the format "Epoch xxx/1000, Absolute Mean Error: yyy" where xxx is an integer and yyy is a float
    # get the yyy value and add it to a list
    # plot the list with the x axis as the epoch number and the y axis as the yyy value
    # save the plot as a png file

    with open(log_file_path, 'r') as f:
        lines = f.readlines()
        errors = []
        epochs = []
        for line in lines:
            if "Epoch" in line and "Absolute Mean Error" in line:
                parts = line.split(",")
                epoch_part = parts[0].split(" ")[1]
                epoch_part = epoch_part.split("/")[0]
                error_part = parts[1].split(":")[1].strip()
                epochs.append(int(epoch_part))
                errors.append(float(error_part))

    plt.plot(epochs, errors)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error")
    plt.title("Training Errors")
    plt.grid()
    plt.savefig("results/anfis_nn_errors_20250510143431.png")
    plt.close()


class ANFISOptimizedProblem(Task):
    n_mfs: int
    mf_type: str
    anfis_model: Any
    X_train: np.ndarray
    y_train: np.ndarray
    n_classes: int
    chunk_size: int

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def objective_function(self, x: list[Any]):
        n_inputs = self.X_train.shape[1]
        x_transformed = self.transform_solution(x)
        mfs = [
            [
                mf_factory.create_membership_function(
                    self.mf_type,
                    params=[
                        x_transformed[f"x_{i * self.n_mfs + j}"][0],
                        x_transformed[f"x_{i * self.n_mfs + j}"][1]
                    ]
                ) for j in range(self.n_mfs)
            ] for i in range(n_inputs)
        ]

        self.anfis_model.override_mfs(mfs)

        y_pred = predict_chunks(self.anfis_model, self.X_train, self.chunk_size)
        return metrics.accuracy_score(self.y_train, y_pred)


# Simulate the ANFIS model using the Feedforward Backpropagation
def simulate_by_nn(
    X_test: np.ndarray, n_inputs: int, n_mfs: int, n_classes: int, chunk_size: int, batches_per_epoch: int
) -> Tuple[np.ndarray, float, Any, Any]:
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

    anfis_model = ANFIS(
        n_inputs,
        n_mfs,
        mf_type,
        now,
        prediction_parser=PredictionParser(parser=prepare_predictions, n_classes=n_classes),
    )

    data_generator_factory = create_data_generator(file_type)
    anfis_model.train(
        data_generator_factory,
        epochs=1000,
        learning_rate_consequent=0.005,
        learning_rate_premise=0.001,
        batches_per_epoch=batches_per_epoch,
        tolerance=1e-2,
        min_improvement=1e-4,
        patience=10,
    )

    logs("nn", now, ["Prediction started"])
    # predict the test data, in chunks
    y_pred = predict_chunks(anfis_model, X_test, chunk_size)
    logs("nn", now, ["Prediction finished"])

    # plots
    anfis_model.plot_membership_functions(save_path=f"results/anfis_nn_mfs_{now}.png")
    anfis_model.plot_errors(save_path=f"results/anfis_nn_errors_{now}.png")

    return y_pred, anfis_model.elapsed_time, anfis_model.errors_epoch, anfis_model


# Simulate the ANFIS model using Evolutionary Computation
# You can replace the PSO with any other algorithm implemented in the library.
def simulate_by_ec(
    X_test: np.ndarray, n_inputs: int, n_mfs: int, n_classes: int, chunk_size: int
) -> Tuple[np.ndarray, float, Any, Any]:
    logs("ec", now, ["Training started..."])

    anfis_model = ANFIS(
        n_inputs,
        n_mfs,
        mf_type,
        now,
        prediction_parser=PredictionParser(parser=prepare_predictions, n_classes=n_classes),
    )

    X_train, y_train = get_data(file_type, train_file_path)

    # we have two parameters for each membership function: the mean and the standard deviation
    task = ANFISOptimizedProblem(
        n_mfs=n_mfs,
        mf_type=str(MembershipFunctionType.GAUSSIAN),
        anfis_model=anfis_model,
        X_train=X_train,
        y_train=y_train,
        n_classes=n_classes,
        chunk_size=chunk_size,
        variables=[
            ContinuousMultiVariable(lower_bounds=[-1, 0], upper_bounds=[1, 1.5], name=f"x_{i}")
            for i in range(n_mfs * n_inputs)
        ],
        minmax="max",
    )

    configuration = ParticleSwarmOptimizationConfig(
        population_size=100,
        fitness_error=1e-4,
        max_cycles=100,
        c1=0.1,
        c2=0.1,
        w=[0.35, 1],
        early_stopping=EarlyStopping(patience=5, min_delta=0.01),
    )

    start_time = time.time()
    result = ParticleSwarmOptimization(configuration, debug=True).optimize(task)
    elapsed_time = time.time() - start_time

    logs("ec", now, ["Training finished."])

    best = best_agent(result.evolution[-1].agents, task.minmax)
    best_params = task.transform_solution(best.position)

    logs("ec", now, [
        "Evolutionary algorithm: Particle Swarm Optimization"
        f"Best parameters: {best_params}",
        f"Best accuracy: {best.cost}"
    ])

    best_params = list(best_params.values())

    anfis_model.override_mfs([
        [
            mf_factory.create_membership_function(
                mf_type, params=[best_params[i * n_mfs + j][0], best_params[i * n_mfs + j][1]]
            ) for j in range(n_mfs)
        ] for i in range(n_inputs)
    ])

    logs("ec", now, ["Prediction started"])
    y_predict = predict_chunks(anfis_model, X_test, chunk_size)
    logs("ec", now, ["Prediction finished"])

    return y_predict, elapsed_time, result, anfis_model


def prepare_db():
    # fraction the combined_data so that the percentages of the output classes are preserved
    def reduce_dataset(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
        stratify_col = df.columns[-1]
        sampled_df, _ = train_test_split(
            df, test_size=(1 - fraction), stratify=df[stratify_col], random_state=42
        )

        return sampled_df.drop_duplicates().reset_index(drop=True)

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
    chunk_size = math.ceil(0.05 * n_rows)  # Number of samples per batch
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

    df_train = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1), dtype=np.float32)
    if file_type == "csv":
        df_train.to_csv(train_file_path, index=False)
    else:
        df_train.to_parquet(train_file_path, index=False)
    del df_train, X_train, y_train
    gc.collect()

    df_test = pd.DataFrame(np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1), dtype=np.float32)
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

    X_test, y_test = get_data(file_type, test_file_path)

    if args.solver == "nn":
        y_predict, elapsed_time, errors, anfis_model = simulate_by_nn(
            X_test,
            train_metadata["n_inputs"],
            train_metadata["n_mfs"],
            train_metadata["n_classes"],
            train_metadata["chunk_size"],
            train_metadata["batches_per_epoch"],
        )
    else:
        y_predict, elapsed_time, result, anfis_model = simulate_by_ec(
            X_test,
            train_metadata["n_inputs"],
            train_metadata["n_mfs"],
            train_metadata["n_classes"],
            train_metadata["chunk_size"],
        )
        errors = result.rates

    save_anfis_model(anfis_model, y_predict, errors, args.solver, now)
    plot_confusion_matrix(y_test, y_predict, args.solver, now, ["No defect", "Defect"])
    save_results(y_test, y_predict, args.solver, now, errors, elapsed_time)


if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists("datasets/defect_presence/db_train.json"):
        # Prepare the database
        # This is a placeholder for the actual database preparation code
        print("Preparing the database...")
        prepare_db()
        print(f"Database prepared and stored into `{train_file_path}` and `{test_file_path}` files.")

    main()
