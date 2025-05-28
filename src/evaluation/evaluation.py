"""
evaluation.py

Module for computing regression evaluation metrics.
Reusable across training, testing, or inference validation.
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error


def evaluate_regression(y_true, y_pred) -> dict:
    """
    Compute standard regression metrics.

    Parameters:
    - y_true: true target values
    - y_pred: predicted target values

    Returns:
    - dict of evaluation metrics (mse, rmse, mae, r2)
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),  # squared=False
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }
