import gc
import os
from typing import Dict, List, Any, Tuple
import pandas as pd
from scipy.io import loadmat
import numpy as np
import pyarrow.parquet as pq

from models import FractionData


def load_mat_file(path_file: str, keys: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Load a .mat file and return a pandas DataFrame

    Args:
        path_file (str): The path to the .mat file
        keys (Dict[str, List[str]]): A dictionary with the keys as the names of the contained data and the values as the
        names of the data to be extracted from the .mat file and converted to a Pandas DataFrame

    Returns:
        A dictionary containing the data extracted from the .mat file
    """
    data_set = loadmat(path_file)

    result = {}
    for main_key, sub_keys in keys.items():
        result[main_key] = {sub_key: data_set[main_key][0][0][i].flatten() for i, sub_key in enumerate(sub_keys)}
    return result


def load_csv_data(path_folder: str, fraction_data: FractionData | None = None) -> pd.DataFrame:
    """
    Load all the CSV files from `path_folder` and return a pandas DataFrame composing all the data

    Args:
        path_folder (str): The path to the folder containing the CSV files
        fraction_data (FractionData | None): A FractData object containing the fraction of data to be used and a callback function

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data extracted from the CSV files
    """
    combined_data = None
    for file_name in os.listdir(path_folder):
        if not file_name.endswith(".csv"):
            continue
        print(f"Loading {file_name}...")

        df = pd.read_csv(os.path.join(path_folder, file_name), dtype=np.float32).dropna()
        combined_data = pd.concat(
            [combined_data, df], ignore_index=True
        ) if combined_data is not None else df
        del df
        gc.collect()

    if fraction_data is None:
        return combined_data

    print(f"Applying fraction: {fraction_data.fraction}")
    return fraction_data.fraction_callback(combined_data, fraction_data.fraction).astype(np.float32)


def prepare_predictions(y_pred: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """
    Prepares predictions for classification metrics:
    - If binary classification (n_classes=2), apply 0.5 threshold.
    - If multiclass, apply rounding.

    Args:
        y_pred (np.ndarray): The predicted values.
        n_classes (int | None): The number of classes. If None, it will be inferred from y_pred.

    Returns:
        np.ndarray: The processed predictions.
    """
    if n_classes is None:
        n_classes = len(np.unique(y_pred))

    # Assume binary classification (threshold at 0.5) or multiclass (round to the nearest integer)
    return (y_pred > 0.5).astype(int) if n_classes == 2 else np.round(y_pred).astype(int)


def logs(file_suffix: str, phrases: List[str]):
    os.makedirs("results", exist_ok=True)
    with open(f"results/anfis_results_{file_suffix}.txt", "a") as f:
        for phrase in phrases:
            print(phrase)
            f.write(phrase + "\n")


def get_test_data(file_type: str, test_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if file_type == "csv":
        dataset_test = pd.read_csv(test_file_path, dtype=np.float32)
        X_test = dataset_test.iloc[:, :-1].values
        y_test = dataset_test.iloc[:, -1].values.flatten()
    else:
        parquet_file = pq.ParquetFile(test_file_path)
        dataset_test = parquet_file.read().to_pandas().astype(np.float32).to_numpy()
        X_test = dataset_test[:, :-1]
        y_test = dataset_test[:, -1]

    y_test = y_test.reshape(-1, 1)
    return X_test, y_test


def predict_chunks(anfis_model: Any, X_test: np.ndarray, chunk_size: int) -> np.ndarray:
    y_pred = np.array([])
    for i in range(0, len(X_test), chunk_size):
        batch = X_test[i: i + chunk_size]
        y_pred_batch = anfis_model.predict(batch)
        y_pred = np.concatenate((y_pred, y_pred_batch), axis=0) if y_pred.size else y_pred_batch

    return y_pred
