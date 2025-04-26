import os
from argparse import ArgumentParser
from typing import Any, Tuple
import numpy as np
from matplotlib import pyplot as plt
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

from helpers import load_csv_data, prepare_predictions
from models import MembershipFunctionType
from services import ANFIS, MembershipFunctionFactory


# Simulate the ANFIS model using the Feedforward Backpropagation
def simulate_by_nn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_mfs: int
) -> Tuple[np.ndarray, float, Any]:
    anfis_model = ANFIS(X_train, y_train, n_mfs, str(MembershipFunctionType.GAUSSIAN))
    anfis_model.train(epochs=1000, learning_rate=0.01, tolerance=1e-4)

    # Predict the house prices on the test set
    y_pred = anfis_model.predict(X_test)

    return y_pred, anfis_model.elapsed_time, anfis_model.errors


# Simulate the ANFIS model using Evolutionary Computation
def simulate_by_ec(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_mfs: int
) -> Tuple[np.ndarray, float, Any]:
    # You can replace the PSO with any other algorithm implemented in the library.
    class ANFISOptimizedProblem(Task):
        def objective_function(self, x: list[Any]):
            x_transformed = self.transform_solution(x)
            mfs = [
                [
                    factory.create_membership_function(
                        mf_type,
                        params=[
                            x_transformed[f"x_{i * n_mfs + j}"][0],
                            x_transformed[f"x_{i * n_mfs + j}"][1]
                        ]
                    ) for j in range(n_mfs)
                ] for i in range(n_inputs)
            ]

            y_pred_real = anfis_model.override_mfs(mfs).predict(X_test)
            y_pred_ready = prepare_predictions(y_pred_real, n_classes=len(np.unique(y_test)))
            return metrics.accuracy_score(y_test, y_pred_ready)

    n_inputs = X_train.shape[1]  # Number of input features
    n_params = 2 # Number of parameters for each membership function (mean and std deviation)

    factory = MembershipFunctionFactory()
    mf_type = str(MembershipFunctionType.GAUSSIAN)

    anfis_model = ANFIS(X_train, y_train, n_mfs, mf_type)

    # we have two parameters for each membership function: the mean and the standard deviation
    task = ANFISOptimizedProblem(
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

    best = best_agent(result.evolution[-1].agents, task.minmax)
    best_params = task.transform_solution(best.position)

    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best.cost}")

    best_params = list(best_params.values())
    membership_functions = [
        [
            factory.create_membership_function(
                mf_type, params=[best_params[i * n_mfs + j][0], best_params[i * n_mfs + j][1]]
            ) for j in range(n_mfs)
        ] for i in range(n_inputs)
    ]

    y_predict = anfis_model.override_mfs(membership_functions).predict(X_test)

    return y_predict, elapsed_time, result


def parse_arguments():
    parser = ArgumentParser()

    # argument for the solver; it can be either 'nn' or 'ec'
    parser.add_argument(
        '-s',
        "--solver",
        default="nn",
        help="Solver to use for the optimization problem.",
    )

    args = parser.parse_args()
    if args.solver not in ["nn", "ec"]:
        raise ValueError("Invalid solver. Use 'nn' or 'ec'.")

    return args


def main(args):
    # Load the dataset
    x, y = load_csv_data("datasets/defect_presence", to_numpy=True)

    # Scaling
    x_scaled = StandardScaler().fit_transform(x)

    # Perform PCA to reduce dimensionality
    # x_pca = PCA(n_components=0.95, svd_solver="full").fit_transform(x_scaled)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # Initialize the ANFIS model
    n_mfs = 3  # Number of membership functions

    if args.solver == "nn":
        y_predict, elapsed_time, errors = simulate_by_nn(X_train, y_train, X_test, n_mfs)
    else:
        y_predict, elapsed_time, result = simulate_by_ec(X_train, y_train, X_test, y_test, n_mfs)
        errors = result.rates

    print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_predict))
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_predict))
    print("Root Mean Squared Error:", metrics.root_mean_squared_error(y_test, y_predict))
    print("R2 Score:", metrics.r2_score(y_test, y_predict))
    print("Time to Convergence:", len(errors))
    print("Final Error:", errors[-1])
    print("Elapsed Time:", elapsed_time)

    os.makedirs("results", exist_ok=True)

    # Plot the results
    plt.scatter(y_test, y_predict, color="blue")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linewidth=2)
    plt.xlabel("True Defect Category")
    plt.ylabel("Predicted Defect Category")
    plt.title("ANFIS Regression - True vs Predicted Defect Categories")
    plt.savefig(f"results/anfis_results_{args.solver}.png")


if __name__ == "__main__":
    main(parse_arguments())