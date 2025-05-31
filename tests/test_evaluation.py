import numpy as np
from evaluation import evaluate_regression

def test_evaluate_regression_perfect_prediction():
    y_true = [100, 200, 300]
    y_pred = [100, 200, 300]  # perfect match

    metrics = evaluate_regression(y_true, y_pred)

    assert metrics["mse"] == 0
    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert metrics["r2"] == 1

def test_evaluate_regression_with_errors():
    y_true = [100, 200, 300]
    y_pred = [110, 190, 310]

    metrics = evaluate_regression(y_true, y_pred)

    # Check keys
    for key in ["mse", "rmse", "mae", "r2"]:
        assert key in metrics

    # Check that metrics are > 0
    assert metrics["mse"] > 0
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0

    # r2 should be < 1 for imperfect prediction
    assert metrics["r2"] < 1

def test_evaluate_regression_handles_numpy_arrays():
    y_true = np.array([5, 15, 25])
    y_pred = np.array([5, 14, 27])

    metrics = evaluate_regression(y_true, y_pred)

    assert isinstance(metrics, dict)
    assert all(isinstance(v, float) for v in metrics.values())
