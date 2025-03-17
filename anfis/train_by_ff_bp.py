# Training by feedforward backpropagation

import numpy as np
from matplotlib import pyplot as plt
from sklearn import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from helpers import load_mat_file
from models import MembershipFunctionType
from services import ANFIS

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

# Initialize and train ANFIS model
n_rules = 4  # Simple rule base
n_inputs = X_train.shape[1]  # Number of input features
anfis_model = ANFIS(X_train_std, y_train, n_rules, n_inputs, str(MembershipFunctionType.GAUSSIAN))
anfis_model.train(epochs=1000, learning_rate=0.01, tolerance=1e-4)

# Predict the house prices on the test set
y_pred = anfis_model.predict(X_test_std)

print("Mean Absolute Error:", np.mean(np.abs(y_test - y_pred)))
print("Mean Squared Error:", np.mean((y_test - y_pred) ** 2))
print("Root Mean Squared Error:", np.sqrt(np.mean((y_test - y_pred) ** 2)))
print("R2 Score:", 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
print("Time to Convergence:", len(anfis_model.errors))
print("Final Error:", anfis_model.errors[-1])
print("Elapsed Time:", anfis_model.elapsed_time)

# Plot the results
plt.scatter(y_test, y_pred, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linewidth=2)
plt.xlabel("True Defect Category")
plt.ylabel("Predicted Defect Category")
plt.title("ANFIS Regression - True vs Predicted Defect Categories")
plt.show()
