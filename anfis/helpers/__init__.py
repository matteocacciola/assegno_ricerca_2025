import os
from typing import Dict, List, Any, Tuple
import pandas as pd
from scipy.io import loadmat
import numpy as np


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


def load_csv_data(
    path_folder: str,
    inputs: List[str] | None = None,
    output: str | None = None,
    to_numpy: bool = False,
    fraction: float | None = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all the CSV files from `path_folder` and return a pandas DataFrame composing all the data

    Args:
        path_folder (str): The path to the folder containing the CSV files
        inputs (List[str]): A list of the names of the input columns to be extracted from the CSV files
        output (str): The name of the output column to be extracted from the CSV files
        to_numpy (bool): If True, convert the DataFrame to a numpy array
        fraction (float | None): If not None, the fraction of the data to be used. If None, all the data will be used

    Returns:
        A tuple containing two pandas DataFrames: the first one with the input data and the second one with the output data
    """
    data_frames = []
    if inputs is not None and output is None:
        raise ValueError("If inputs is provided, output must be provided too.")

    columns = (inputs if inputs else []) + ([output] if output else [])

    for file_name in os.listdir(path_folder):
        if not file_name.endswith(".csv"):
            continue
        print(f"Loading {file_name}...")

        df = pd.read_csv(os.path.join(path_folder, file_name))
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=1)

        data_frames.append(df[columns] if columns else df)

    combined_data = pd.concat(data_frames, ignore_index=True)
    combined_data = combined_data.dropna().drop_duplicates().reset_index(drop=True)

    # If inputs is None, check the output:
    # - if it is None, then split all the columns but the last one for the inputs, and the last one for the output
    # - if it is not None, then split the inputs as all the columns but the one identified by the output
    if inputs is None:
        if output is None:
            inputs = combined_data.columns[:-1].tolist()
            output = combined_data.columns[-1]
        else:
            inputs = [col for col in combined_data.columns if col != output]

    if to_numpy:
        return combined_data[inputs].to_numpy(), combined_data[output].to_numpy()

    return combined_data[inputs], combined_data[output]


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

    if n_classes == 2:
        # Assume binary classification: threshold at 0.5
        y_pred_processed = (y_pred > 0.5).astype(int)
    else:
        # Assume multiclass: round to nearest integer
        y_pred_processed = np.round(y_pred).astype(int)

    return y_pred_processed
