import numpy as np

def error_norms(filepath: str):
    # Load data from npz file
    data = np.load(filepath)

    # Fetch error pr data point
    error = data["error_n"]

    # L2 error norm
    l2_error = np.sqrt(np.sum(error** 2))

    # Maximum norm
    max_error = np.max(np.abs(error))

    return l2_error, max_error

