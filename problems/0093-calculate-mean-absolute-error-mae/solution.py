import numpy as np

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between two arrays.

    Parameters:
        y_true (numpy.ndarray): Array of true values
        y_pred (numpy.ndarray): Array of predicted values

    Returns:
        float: Mean Absolute Error
    """
    # Your code here
    y_pred = np.array(y_pred,dtype=float)
    y_true = np.array(y_true,dtype=float)
    return float(np.mean(np.abs(y_true-y_pred)))