import pytest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock
import warnings

# Add the parent directory to path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try different possible import paths based on your project structure
    from src.evaluation.evaluation import evaluate_regression
    MODULE_PATH = 'src.evaluation.evaluation'
except ImportError:
    try:
        from src.evaluation import evaluate_regression
        MODULE_PATH = 'src.evaluation'
    except ImportError:
        try:
            from evaluation.evaluation import evaluate_regression
            MODULE_PATH = 'evaluation.evaluation'
        except ImportError:
            try:
                from evaluation import evaluate_regression
                MODULE_PATH = 'evaluation'
            except ImportError:
                # If evaluation.py is directly in src/
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
                try:
                    from evaluation import evaluate_regression
                    MODULE_PATH = 'evaluation'
                except ImportError:
                    raise ImportError("Could not import evaluate_regression. Please check the module location.")

print(f"Successfully imported evaluate_regression from {MODULE_PATH}")


class TestEvaluateRegression:
    """Test cases for evaluate_regression function"""
    
    def test_perfect_predictions(self):
        """Test metrics when predictions are perfect (y_true == y_pred)"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        metrics = evaluate_regression(y_true, y_pred)
        
        # Perfect predictions should yield perfect metrics
        assert metrics["mse"] == 0.0, f"Expected MSE=0, got {metrics['mse']}"
        assert metrics["rmse"] == 0.0, f"Expected RMSE=0, got {metrics['rmse']}"
        assert metrics["mae"] == 0.0, f"Expected MAE=0, got {metrics['mae']}"
        assert metrics["r2"] == 1.0, f"Expected R2=1, got {metrics['r2']}"
        
        # Verify all expected keys are present
        expected_keys = {"mse", "rmse", "mae", "r2"}
        assert set(metrics.keys()) == expected_keys, f"Missing keys: {expected_keys - set(metrics.keys())}"
    
    def test_worst_case_predictions(self):
        """Test metrics with deliberately bad predictions"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Perfectly inverted
        
        metrics = evaluate_regression(y_true, y_pred)
        
        # All metrics should indicate poor performance
        assert metrics["mse"] > 0, f"Expected MSE > 0, got {metrics['mse']}"
        assert metrics["rmse"] > 0, f"Expected RMSE > 0, got {metrics['rmse']}"
        assert metrics["mae"] > 0, f"Expected MAE > 0, got {metrics['mae']}"
        assert metrics["r2"] < 1.0, f"Expected R2 < 1, got {metrics['r2']}"
        
        # MSE should equal RMSE squared
        assert abs(metrics["mse"] - metrics["rmse"]**2) < 1e-10, \
            f"MSE should equal RMSE^2: {metrics['mse']} vs {metrics['rmse']**2}"
    
    def test_known_values_simple(self):
        """Test with known simple values where we can calculate metrics manually"""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])  # Each prediction is 0.5 higher
        
        metrics = evaluate_regression(y_true, y_pred)
        
        # Manual calculations:
        # MSE = ((0.5)^2 + (0.5)^2 + (0.5)^2) / 3 = 0.75 / 3 = 0.25
        # RMSE = sqrt(0.25) = 0.5
        # MAE = (0.5 + 0.5 + 0.5) / 3 = 0.5
        
        assert abs(metrics["mse"] - 0.25) < 1e-10, f"Expected MSE=0.25, got {metrics['mse']}"
        assert abs(metrics["rmse"] - 0.5) < 1e-10, f"Expected RMSE=0.5, got {metrics['rmse']}"
        assert abs(metrics["mae"] - 0.5) < 1e-10, f"Expected MAE=0.5, got {metrics['mae']}"
        
        # R2 calculation: y_true mean = 2.0, TSS = 2.0, RSS = 0.75, R2 = 1 - 0.75/2.0 = 0.625
        assert abs(metrics["r2"] - 0.625) < 1e-10, f"Expected R2=0.625, got {metrics['r2']}"
    
    def test_mathematical_relationships(self):
        """Test mathematical relationships between metrics"""
        np.random.seed(42)
        y_true = np.random.normal(10, 5, 100)
        y_pred = y_true + np.random.normal(0, 2, 100)  # Add some noise
        
        metrics = evaluate_regression(y_true, y_pred)
        
        # RMSE should be the square root of MSE
        assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-10, \
            f"RMSE should equal sqrt(MSE): {metrics['rmse']} vs {np.sqrt(metrics['mse'])}"
        
        # All metrics should be non-negative except R2 (which can be negative)
        assert metrics["mse"] >= 0, f"MSE should be non-negative, got {metrics['mse']}"
        assert metrics["rmse"] >= 0, f"RMSE should be non-negative, got {metrics['rmse']}"
        assert metrics["mae"] >= 0, f"MAE should be non-negative, got {metrics['mae']}"
        
        # R2 should be <= 1 for reasonable predictions
        assert metrics["r2"] <= 1.0, f"R2 should be <= 1, got {metrics['r2']}"
    
    def test_single_data_point(self):
        """Test with single data point"""
        y_true = np.array([5.0])
        y_pred = np.array([4.0])
        
        # Suppress the expected warning about R^2 with single data point
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = evaluate_regression(y_true, y_pred)
        
        # With single point: MSE = (5-4)^2 = 1, RMSE = 1, MAE = 1
        assert metrics["mse"] == 1.0, f"Expected MSE=1, got {metrics['mse']}"
        assert metrics["rmse"] == 1.0, f"Expected RMSE=1, got {metrics['rmse']}"
        assert metrics["mae"] == 1.0, f"Expected MAE=1, got {metrics['mae']}"
        
        # R2 is undefined for single point, sklearn returns NaN
        assert isinstance(metrics["r2"], (float, np.floating)), f"R2 should be float, got {type(metrics['r2'])}"
        assert np.isnan(metrics["r2"]), f"Expected R2=NaN for single point, got {metrics['r2']}"
    
   
    
    def test_negative_values(self):
        """Test with negative values in predictions and true values"""
        y_true = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y_pred = np.array([-1.5, -0.5, 0.5, 1.5, 2.5])
        
        metrics = evaluate_regression(y_true, y_pred)
        
        # All metrics should be computed correctly with negative values
        assert isinstance(metrics["mse"], (float, np.floating)), f"MSE should be float, got {type(metrics['mse'])}"
        assert isinstance(metrics["rmse"], (float, np.floating)), f"RMSE should be float, got {type(metrics['rmse'])}"
        assert isinstance(metrics["mae"], (float, np.floating)), f"MAE should be float, got {type(metrics['mae'])}"
        assert isinstance(metrics["r2"], (float, np.floating)), f"R2 should be float, got {type(metrics['r2'])}"
        
        # Verify the mathematical relationship still holds
        assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-10
    
    def test_large_values(self):
        """Test with large numerical values"""
        y_true = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
        y_pred = np.array([1.1e6, 2.1e6, 3.1e6, 4.1e6, 5.1e6])
        
        metrics = evaluate_regression(y_true, y_pred)
        
        # Should handle large values without overflow
        assert np.isfinite(metrics["mse"]), f"MSE should be finite, got {metrics['mse']}"
        assert np.isfinite(metrics["rmse"]), f"RMSE should be finite, got {metrics['rmse']}"
        assert np.isfinite(metrics["mae"]), f"MAE should be finite, got {metrics['mae']}"
        assert np.isfinite(metrics["r2"]), f"R2 should be finite, got {metrics['r2']}"
        
        # Each prediction is off by 1e5, so MAE should be 1e5
        assert abs(metrics["mae"] - 1e5) < 1e-5, f"Expected MAE=1e5, got {metrics['mae']}"
    
    def test_small_values(self):
        """Test with very small numerical values"""
        y_true = np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
        y_pred = np.array([1.1e-6, 2.1e-6, 3.1e-6, 4.1e-6, 5.1e-6])
        
        metrics = evaluate_regression(y_true, y_pred)
        
        # Should handle small values without underflow
        assert np.isfinite(metrics["mse"]), f"MSE should be finite, got {metrics['mse']}"
        assert np.isfinite(metrics["rmse"]), f"RMSE should be finite, got {metrics['rmse']}"
        assert np.isfinite(metrics["mae"]), f"MAE should be finite, got {metrics['mae']}"
        assert np.isfinite(metrics["r2"]), f"R2 should be finite, got {metrics['r2']}"
        
        # Each prediction is off by 1e-7, so MAE should be 1e-7
        assert abs(metrics["mae"] - 1e-7) < 1e-12, f"Expected MAE=1e-7, got {metrics['mae']}"
    
    def test_different_array_types(self):
        """Test with different input array types (list, numpy, pandas)"""
        # Test data
        y_true_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred_list = [1.1, 2.1, 3.1, 4.1, 5.1]
        
        y_true_np = np.array(y_true_list)
        y_pred_np = np.array(y_pred_list)
        
        y_true_pd = pd.Series(y_true_list)
        y_pred_pd = pd.Series(y_pred_list)
        
        # All should produce the same results
        metrics_list = evaluate_regression(y_true_list, y_pred_list)
        metrics_np = evaluate_regression(y_true_np, y_pred_np)
        metrics_pd = evaluate_regression(y_true_pd, y_pred_pd)
        
        # Compare results
        for key in ["mse", "rmse", "mae", "r2"]:
            assert abs(metrics_list[key] - metrics_np[key]) < 1e-10, \
                f"List vs NumPy mismatch for {key}: {metrics_list[key]} vs {metrics_np[key]}"
            assert abs(metrics_np[key] - metrics_pd[key]) < 1e-10, \
                f"NumPy vs Pandas mismatch for {key}: {metrics_np[key]} vs {metrics_pd[key]}"
    
    def test_empty_arrays(self):
        """Test behavior with empty arrays"""
        y_true = np.array([])
        y_pred = np.array([])
        
        # Should raise an error or handle gracefully
        with pytest.raises((ValueError, IndexError)):
            evaluate_regression(y_true, y_pred)
    
    def test_mismatched_lengths(self):
        """Test behavior with mismatched array lengths"""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])  # Different length
        
        # Should raise an error
        with pytest.raises(ValueError):
            evaluate_regression(y_true, y_pred)
    
    def test_nan_values(self):
        """Test behavior with NaN values"""
        y_true = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # sklearn metrics raise ValueError for NaN values
        with pytest.raises(ValueError, match="Input contains NaN"):
            evaluate_regression(y_true, y_pred)
    
    def test_clean_data_after_nan_removal(self):
        """Test that function works correctly when NaN values are removed beforehand"""
        # Simulate data where NaN values have been properly handled upstream
        y_true = np.array([1.0, 2.0, 4.0, 5.0])  # NaN removed
        y_pred = np.array([1.1, 2.1, 4.1, 5.1])  # Corresponding prediction removed
        
        metrics = evaluate_regression(y_true, y_pred)
        
        # Should work normally with clean data
        assert all(np.isfinite(v) for v in metrics.values()), \
            "All metrics should be finite with clean data"
        assert metrics["mse"] > 0
        assert metrics["rmse"] > 0 
        assert metrics["mae"] > 0
    
    def test_infinite_values(self):
        """Test behavior with infinite values"""
        y_true = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # sklearn metrics raise ValueError for infinite values
        with pytest.raises(ValueError, match="Input contains infinity"):
            evaluate_regression(y_true, y_pred)
    
    def test_clean_data_after_infinite_removal(self):
        """Test that function works correctly when infinite values are removed beforehand"""
        # Simulate data where infinite values have been properly handled upstream
        y_true = np.array([1.0, 2.0, 4.0, 5.0])  # inf removed
        y_pred = np.array([1.1, 2.1, 4.1, 5.1])  # Corresponding prediction removed
        
        metrics = evaluate_regression(y_true, y_pred)
        
        # Should work normally with clean data
        assert all(np.isfinite(v) for v in metrics.values()), \
            "All metrics should be finite with clean data"
        assert metrics["mse"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
    
    def test_data_validation_best_practices(self):
        """Test demonstrating best practices for data validation before evaluation"""
        # Original data with problematic values
        y_true_raw = np.array([1.0, 2.0, np.nan, 4.0, np.inf, 6.0])
        y_pred_raw = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
        
        # Best practice: Clean data before evaluation
        valid_mask = np.isfinite(y_true_raw) & np.isfinite(y_pred_raw)
        y_true_clean = y_true_raw[valid_mask]
        y_pred_clean = y_pred_raw[valid_mask]
        
        # Should work with cleaned data
        metrics = evaluate_regression(y_true_clean, y_pred_clean)
        
        # Verify we have valid results
        assert all(np.isfinite(v) for v in metrics.values()), \
            "All metrics should be finite after data cleaning"
        assert len(y_true_clean) == 4, "Should have 4 valid data points"  # 1.0, 2.0, 4.0, 6.0
        assert metrics["mse"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
    
    def test_zero_values(self):
        """Test with all zero values"""
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Suppress potential warnings about zero variance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = evaluate_regression(y_true, y_pred)
        
        # Perfect zero predictions
        assert metrics["mse"] == 0.0, f"Expected MSE=0, got {metrics['mse']}"
        assert metrics["rmse"] == 0.0, f"Expected RMSE=0, got {metrics['rmse']}"
        assert metrics["mae"] == 0.0, f"Expected MAE=0, got {metrics['mae']}"
        # R2 with perfect prediction (even with zero variance) returns 1.0 in sklearn
        assert isinstance(metrics["r2"], (float, np.floating))
        assert metrics["r2"] == 1.0, f"Expected R2=1.0 for perfect prediction, got {metrics['r2']}"
    
    def test_sklearn_consistency(self):
        """Test that results match direct sklearn function calls"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
        
        np.random.seed(123)
        y_true = np.random.normal(0, 1, 50)
        y_pred = y_true + np.random.normal(0, 0.1, 50)
        
        metrics = evaluate_regression(y_true, y_pred)
        
        # Compare with direct sklearn calls
        sklearn_mse = mean_squared_error(y_true, y_pred)
        sklearn_rmse = root_mean_squared_error(y_true, y_pred)
        sklearn_mae = mean_absolute_error(y_true, y_pred)
        sklearn_r2 = r2_score(y_true, y_pred)
        
        assert abs(metrics["mse"] - sklearn_mse) < 1e-10, \
            f"MSE mismatch: {metrics['mse']} vs sklearn {sklearn_mse}"
        assert abs(metrics["rmse"] - sklearn_rmse) < 1e-10, \
            f"RMSE mismatch: {metrics['rmse']} vs sklearn {sklearn_rmse}"
        assert abs(metrics["mae"] - sklearn_mae) < 1e-10, \
            f"MAE mismatch: {metrics['mae']} vs sklearn {sklearn_mae}"
        assert abs(metrics["r2"] - sklearn_r2) < 1e-10, \
            f"R2 mismatch: {metrics['r2']} vs sklearn {sklearn_r2}"
    
    def test_return_type_and_structure(self):
        """Test that return type and structure are correct"""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        
        metrics = evaluate_regression(y_true, y_pred)
        
        # Should return a dictionary
        assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"
        
        # Should have exactly 4 keys
        expected_keys = {"mse", "rmse", "mae", "r2"}
        assert set(metrics.keys()) == expected_keys, \
            f"Expected keys {expected_keys}, got {set(metrics.keys())}"
        
        # All values should be numeric
        for key, value in metrics.items():
            assert isinstance(value, (float, np.floating)), \
                f"Metric {key} should be float, got {type(value)}: {value}"
    
    def test_large_dataset_performance(self):
        """Test with larger dataset to ensure reasonable performance"""
        np.random.seed(456)
        n_samples = 10000
        y_true = np.random.normal(100, 20, n_samples)
        y_pred = y_true + np.random.normal(0, 5, n_samples)
        
        # Should complete without issues
        metrics = evaluate_regression(y_true, y_pred)
        
        # Verify all metrics are computed
        assert all(isinstance(v, (float, np.floating)) for v in metrics.values())
        assert all(np.isfinite(v) for v in metrics.values())
    
    def test_edge_case_r2_negative(self):
        """Test case where R2 can be negative (predictions worse than mean)"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Predictions that are worse than just predicting the mean
        y_pred = np.array([10.0, -5.0, 15.0, -10.0, 20.0])
        
        metrics = evaluate_regression(y_true, y_pred)
        
        # R2 should be negative in this case
        assert metrics["r2"] < 0, f"Expected negative R2, got {metrics['r2']}"
        
        # Other metrics should still be positive
        assert metrics["mse"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0


class TestEvaluateRegressionIntegration:
    """Integration tests for evaluate_regression function"""
    
    def test_realistic_scenario(self):
        """Test with realistic house price prediction scenario"""
        # Simulate house prices in thousands
        np.random.seed(789)
        true_prices = np.random.lognormal(mean=5, sigma=0.5, size=100) * 100  # ~$150k-800k range
        
        # Simulate model predictions with some error
        prediction_error = np.random.normal(0, 0.1, 100)
        predicted_prices = true_prices * (1 + prediction_error)
        
        metrics = evaluate_regression(true_prices, predicted_prices)
        
        # For a reasonable model, we expect:
        assert 0.7 < metrics["r2"] < 1.0, f"R2 should be reasonably high, got {metrics['r2']}"
        assert metrics["rmse"] < np.std(true_prices), f"RMSE should be less than standard deviation"
        
        # Relative errors should be reasonable
        relative_rmse = metrics["rmse"] / np.mean(true_prices)
        assert relative_rmse < 0.2, f"Relative RMSE should be < 20%, got {relative_rmse:.3f}"
    
    def test_different_scales(self):
        """Test that metrics scale appropriately with different value ranges"""
        # Test with different scales but same relative errors
        base_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        base_pred = np.array([1.1, 2.2, 3.3, 4.4, 5.5])  # 10% relative error each
        
        scales = [1, 10, 100, 1000]
        r2_values = []
        relative_rmse_values = []
        
        for scale in scales:
            y_true = base_true * scale
            y_pred = base_pred * scale
            
            metrics = evaluate_regression(y_true, y_pred)
            r2_values.append(metrics["r2"])
            relative_rmse_values.append(metrics["rmse"] / np.mean(y_true))
        
        # R2 should be scale-invariant (approximately the same)
        for i in range(1, len(r2_values)):
            assert abs(r2_values[i] - r2_values[0]) < 1e-10, \
                f"R2 should be scale-invariant, but {r2_values[i]} != {r2_values[0]}"
        
        # Relative RMSE should also be scale-invariant
        for i in range(1, len(relative_rmse_values)):
            assert abs(relative_rmse_values[i] - relative_rmse_values[0]) < 1e-10, \
                f"Relative RMSE should be scale-invariant"


# Fixtures for common test data
@pytest.fixture
def sample_regression_data():
    """Sample regression data for testing"""
    np.random.seed(42)
    n_samples = 50
    x = np.linspace(0, 10, n_samples)
    y_true = 2 * x + 1 + np.random.normal(0, 1, n_samples)  # Linear with noise
    y_pred = 2.1 * x + 0.9 + np.random.normal(0, 0.8, n_samples)  # Slightly different slope/intercept
    return y_true, y_pred


@pytest.fixture
def perfect_prediction_data():
    """Perfect prediction data for testing"""
    y_values = np.array([10.5, 20.3, 30.7, 40.1, 50.9])
    return y_values, y_values.copy()


def test_with_sample_data(sample_regression_data):
    """Test using sample regression data fixture"""
    y_true, y_pred = sample_regression_data
    
    metrics = evaluate_regression(y_true, y_pred)
    
    # Should have reasonable performance
    assert 0.5 < metrics["r2"] < 1.0, f"R2 should be reasonable, got {metrics['r2']}"
    assert metrics["mse"] > 0
    assert metrics["rmse"] > 0 
    assert metrics["mae"] > 0


def test_with_perfect_data(perfect_prediction_data):
    """Test using perfect prediction data fixture"""
    y_true, y_pred = perfect_prediction_data
    
    metrics = evaluate_regression(y_true, y_pred)
    
    # Should be perfect metrics
    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


# Test runner configuration
if __name__ == "__main__":
    # Configure for verbose testing
    pytest.main(["-v", __file__])