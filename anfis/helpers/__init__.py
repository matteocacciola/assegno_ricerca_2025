from typing import Dict, List, Any
from scipy.io import loadmat


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
