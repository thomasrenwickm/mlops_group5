import pandas as pd
import pytest
from unittest import mock
from inference import run_inference

@mock.patch("inference.pd.read_csv")
@mock.patch("inference.pickle.load")
@mock.patch("inference.pd.read_json")
@mock.patch("inference.pd.DataFrame.to_csv")
@mock.patch("inference.engineer_features")
@mock.patch("inference.transform_with_pipeline")
def test_run_inference_pipeline_mocks_everything(
    mock_transform,
    mock_engineer,
    mock_to_csv,
    mock_read_json,
    mock_pickle_load,
    mock_read_csv,
):
    # Setup mocks
    df_input = pd.DataFrame({"dummy": [1, 2, 3]})
    df_features = df_input.copy()
    df_transformed = pd.DataFrame({
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [0.4, 0.5, 0.6]
    })

    model_mock = mock.Mock()
    model_mock.predict.return_value = [100, 200, 300]

    # Return values for mocks
    mock_read_csv.return_value = df_input
    mock_engineer.return_value = df_features
    mock_transform.return_value = df_transformed
    mock_read_json.return_value = pd.Series(["feature1", "feature2"])
    mock_pickle_load.side_effect = [mock.Mock(), model_mock]

    # Create a minimal config and run
    dummy_config_path = "tests/dummy_config.yaml"
    dummy_input = "tests/mock_input.csv"
    dummy_output = "tests/mock_output.csv"

    with mock.patch("builtins.open", mock.mock_open(read_data="""
    data_source:
      new_data_path: mock_input.csv
    artifacts:
      model: mock_model.pkl
      preprocessing_pipeline: mock_pipeline.pkl
      selected_features: mock_features.json
      inference_output: mock_output.csv
    """)):
        run_inference(dummy_input, dummy_config_path, dummy_output)

    # Validate predictions saved
    mock_to_csv.assert_called_once()
    model_mock.predict.assert_called_once_with(df_transformed[["feature1", "feature2"]])
