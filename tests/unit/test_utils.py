import pandas as pd
from myapp.pipelines.stage_04_data_preprocessing import DataPreprocessingPipeline
from myapp.pipelines.stage_05_feature_engineering import FeatureEngineeringPipeline
from myapp.config.config_manager import ConfigManager
from myapp.utils.logger import CustomLogger


# Initialize config and logger once
config = ConfigManager().appconfig
logger = CustomLogger(__name__).get_logger()

# -------------------------
# Fixtures / Sample Data
# -------------------------

def get_sample_data():
    return pd.DataFrame({
        "datetime": ["2025-10-01 00:00:00", "2025-10-01 01:00:00"],
        "aep_mw": [100, 150],
    })


# -------------------------
# Unit Tests
# -------------------------

def test_preprocessing_pipeline_returns_expected_columns():
    sample_data = get_sample_data()
    preprocessor = DataPreprocessingPipeline(config)

    output = preprocessor.run(sample_data)

    assert not output.empty, "Preprocessed DataFrame is empty"
    assert "datetime" in output.columns, "'datetime' column missing"
    assert "aep_mw" in output.columns, "'aep_mw' column missing"
    assert "datetime_dayofweek" in output.columns, "'datetime_dayofweek' not generated"
    assert output["aep_mw"].max() <= 1.0, "'aep_mw' not scaled correctly"
    assert output["aep_mw"].min() >= 0.0, "'aep_mw' not scaled correctly"


def test_feature_engineering_pipeline_adds_expected_columns():
    sample_data = get_sample_data()
    preprocessor = DataPreprocessingPipeline(config)
    preprocessed_data = preprocessor.run(sample_data)

    feature_engineer = FeatureEngineeringPipeline(config)
    output = feature_engineer.run(preprocessed_data)

    assert not output.empty, "Feature engineered DataFrame is empty"
    assert "is_weekend" in output.columns, "'is_weekend' not added"
    assert "is_holiday" in output.columns, "'is_holiday' not added"
    assert set(output["is_weekend"].unique()).issubset({0, 1}), "'is_weekend' contains invalid values"
    assert set(output["is_holiday"].unique()).issubset({0, 1}), "'is_holiday' contains invalid values"


def test_preprocessing_pipeline_handles_empty_input():
    empty_data = pd.DataFrame(columns=["datetime", "aep_mw"])
    preprocessor = DataPreprocessingPipeline(config)

    output = preprocessor.run(empty_data)

    assert output.empty, "Expected empty output for empty input"


def test_pipeline_end_to_end_integration():
    sample_data = get_sample_data()

    preprocessor = DataPreprocessingPipeline(config)
    preprocessed = preprocessor.run(sample_data)

    feature_engineer = FeatureEngineeringPipeline(config)
    fe_output = feature_engineer.run(preprocessed)

    assert not fe_output.empty, "Final output is empty"
    assert "is_weekend" in fe_output.columns, "Final output missing 'is_weekend'"
    assert "datetime_dayofweek" in fe_output.columns, "Final output missing 'datetime_dayofweek'"


# -------------------------
# Optional: Manual Test Runner
# -------------------------

if __name__ == "__main__":
    try:
        test_preprocessing_pipeline_returns_expected_columns()
        print("✅ test_preprocessing_pipeline_returns_expected_columns passed")

        test_feature_engineering_pipeline_adds_expected_columns()
        print("✅ test_feature_engineering_pipeline_adds_expected_columns passed")

        test_preprocessing_pipeline_handles_empty_input()
        print("✅ test_preprocessing_pipeline_handles_empty_input passed")

        test_pipeline_end_to_end_integration()
        print("✅ test_pipeline_end_to_end_integration passed")

    except AssertionError as e:
        print("❌ Test failed:", e)
