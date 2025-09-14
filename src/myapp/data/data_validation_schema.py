from typing import Dict
import pandas as pd


class DataValidationSchema:
    """Data schema definition for validation."""

    required_columns: Dict[str, str] = {
        "datetime": "datetime64[ns]",
        "aep_mw": "float64",
    }

    rename_map: Dict[str, str] = {
        "Datetime": "datetime",
        "AEP_MW": "aep_mw",
    }
    
    nullable_columns = set()
    
    value_constraints: Dict[str, Dict[str, float]] = {
        "aep_mw": {"min": 0.0}
    }
    
    default_values = {}
    
    column_descriptions: Dict[str, str] = {
        "datetime": "Timestamp of the observation",
        "aep_mw": "Actual energy production in megawatts",
    }
    
    @staticmethod
    def validate_datetime_monotonic(df: pd.DataFrame) -> None:
        """Check if the datetime column is strictly increasing."""
        if not df['datetime'].is_monotonic_increasing:
            raise ValueError("Datetime column must be sorted in ascending order.")
        
    @staticmethod
    def validate_no_duplicates(df: pd.DataFrame) -> None:
        # Check duplicates on 'datetime' column since it should be unique in time series
        if df['datetime'].duplicated().any():
            raise ValueError("Duplicate rows found based on 'datetime' column")

    custom_validations = {
        "datetime_monotonic": validate_datetime_monotonic.__func__,
        "no_duplicates": validate_no_duplicates.__func__,
    }
