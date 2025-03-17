# Optimization of the ANFIS model using Evolutionary Computation

from typing import Any
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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

from helpers import load_mat_file
from models import MembershipFunctionType
from services import ANFIS, MembershipFunctionFactory


class ANFISOptimizedProblem(Task):
    def objective_function(self, x: list[Any]):
        x_transformed = self.transform_solution(x)
        mfs = [
            [
                factory.create_membership_function(
                    mf_type,
                    params=[
                        x_transformed[f"x_{i * n_rules + j}"][0],
                        x_transformed[f"x_{i * n_rules + j}"][1]
                    ]
                ) for j in range(n_rules)
            ] for i in range(n_inputs)
        ]

        y_pred = ANFIS(X_train, y_train, n_rules, n_inputs, mf_type, membership_functions=mfs).predict(X_test_std)
        return metrics.accuracy_score(y_test, y_pred)


data_path = "datasets/TQA_long_basso1.mat"
data = load_mat_file(data_path, {
    "param": ["f1", "f2", "gain", "phi0", "dt", "dx", "Ns0", "Nacq", "T_des", "T_eff"],
    "signals": ["s", "y", "psi1", "psi2", "psi3", "psi4", "h1", "h2", "h3", "h4"]
})

input_data = np.vstack(data["signals"]["y"]).T
pca_input_data = PCA(n_components=0.95, svd_solver="full")
output_data = defect_category

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(pca_input_data, output_data, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Initialize the ANFIS model
n_rules = 4  # Simple rule base
n_inputs = X_train.shape[1]  # Number of input features

factory = MembershipFunctionFactory()
mf_type = str(MembershipFunctionType.GAUSSIAN)

task = ANFISOptimizedProblem(
    variables=[
        ContinuousMultiVariable(lower_bound=[0.01, 10], upper_bound=[1, 100], name=f"x_{i}") for i in range(n_rules * n_inputs)
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
result = ParticleSwarmOptimization(configuration).optimize(task)
elapsed_time = time.time() - start_time

best = best_agent(result.evolution[-1].agents, task.minmax)

print(f"Best parameters: {task.transform_solution(best.position)}")
print(f"Best accuracy: {best.cost}")

membership_functions = [
    [
        factory.create_membership_function(
            mf_type, params=[best.position[i * n_rules + j][0], best.position[i * n_rules + j][1]]
        ) for j in range(n_rules)
    ] for i in range(n_inputs)
]

anfis_model = ANFIS(X_train, y_train, n_rules, n_inputs, mf_type, membership_functions=membership_functions)
y_predict = anfis_model.predict(X_test_std)

print("Mean Absolute Error:", np.mean(np.abs(y_test - y_predict)))
print("Mean Squared Error:", np.mean((y_test - y_predict) ** 2))
print("Root Mean Squared Error:", np.sqrt(np.mean((y_test - y_predict) ** 2)))
print("R2 Score:", 1 - np.sum((y_test - y_predict) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
print("Time to Convergence:", len(result.rates))
print("Final Error:", result.rates[-1])
print("Elapsed Time:", elapsed_time)

# Plot the results
plt.scatter(y_test, y_predict, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linewidth=2)
plt.xlabel("True Defect Category")
plt.ylabel("Predicted Defect Category")
plt.title("ANFIS Regression - True vs Predicted Defect Categories")
plt.show()

#
# You can replace the PSO with any other algorithm implemented in the library.
