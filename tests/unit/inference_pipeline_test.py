import os
import pytest
import pandas as pd
from myapp.config.config_manager import AppConfig, ConfigManager
from myapp.utils.logger import CustomLogger
from myapp.pipelines.main_pipeline import MainPipeline
from myapp.services.inference_pipeline import InferencePipeline
from myapp.utils.filepaths import resolve_project_path


# Initialize config and logger once
@pytest.fixture(scope="module")
def test_config() -> AppConfig:
    config = ConfigManager().appconfig
    # Override paths for testing if needed
    return config

@pytest.fixture(scope="module")
def test_logger() -> CustomLogger:
    return CustomLogger(__name__).get_logger()

def test_inference_pipeline_runs_successfully(test_config, test_logger):
    pipeline = InferencePipeline(config=test_config, logger=test_logger)
    # Run the pipeline
    prediction = pipeline.run()
    
    # Check that predictions is a DataFrame and has expected structure
    assert isinstance(prediction, pd.DataFrame), "Predictions should be a DataFrame"
    assert "prediction" in prediction.columns, "'prediction' column missing in output"

    # Check that DataFrame predictions are not empty
    assert not prediction.empty, "Predictions DataFrame is empty"
    assert len(prediction) > 0, "No predictions were made"

   # Optional: Check that predictions are numeric
    assert pd.api.types.is_numeric_dtype(prediction["prediction"]), "Predictions should be numeric"
    assert all(prediction["prediction"].apply(lambda x: isinstance(x, (int, float)))), "All predictions should be int or float"
    print(f"Sample predictions:\n{prediction.head()}")  
    
def test_resolve_project_path():
    resolved_path = resolve_project_path("data/raw")
    assert os.path.isabs(resolved_path)
    assert resolved_path.endswith("data/raw")
    
    
    
if __name__ == "__main__":
    config = ConfigManager().appconfig
    logger = CustomLogger(__name__).get_logger()
    test_inference_pipeline_runs_successfully(config, logger)    
    test_resolve_project_path()
    
