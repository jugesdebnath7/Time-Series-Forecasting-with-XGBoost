from typing import Dict, Optional
import pandas as pd
from myapp.utils.logger import CustomLogger


class ValidationSchema:
    """Schema definition for data validation."""

    logger = CustomLogger(module_name=__name__).get_logger()

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

    @classmethod
    def validate_datetime_monotonic(cls, df: pd.DataFrame) -> None:
        """Check if the datetime column is strictly increasing."""
        if not df['datetime'].is_monotonic_increasing:
            cls.logger.error("Datetime column is not monotonic increasing.")
            raise ValueError("Datetime column must be sorted in ascending order.")
        cls.logger.info("Datetime monotonic validation passed.")

    @classmethod
    def validate_no_duplicates(cls, df: pd.DataFrame) -> None:
        # Check duplicates on 'datetime' column since it should be unique in time series
        if df['datetime'].duplicated().any():
            cls.logger.error("Duplicate rows found in datetime column.")
            raise ValueError("Duplicate rows found based on 'datetime' column")
        cls.logger.info("No duplicates validation passed.")

    custom_validations = {
        "datetime_monotonic": validate_datetime_monotonic,
        "no_duplicates": validate_no_duplicates,
    }

    @classmethod
    def validate_dataframe(
        cls,
        df: pd.DataFrame,
        copy: bool = False,
    ) -> pd.DataFrame:
        if copy:
            df = df.copy()

        cls.logger.info("Starting dataframe validation.")

        # Rename columns
        df = df.rename(columns=cls.rename_map)
        cls.logger.debug(f"Columns renamed using map: {cls.rename_map}")

        # Check required columns
        missing_cols = [col for col in cls.required_columns if col not in df.columns]
        if missing_cols:
            cls.logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate dtypes
        for col, expected_dtype in cls.required_columns.items():
            if col in df.columns:
                if not pd.api.types.is_dtype_equal(df[col].dtype, expected_dtype):
                    msg = f"Column '{col}' has dtype '{df[col].dtype}', expected '{expected_dtype}'"
                    cls.logger.error(msg)
                    raise ValueError(msg)

        # Value constraints
        for col, constraints in cls.value_constraints.items():
            if col in df.columns:
                if "min" in constraints and (df[col] < constraints["min"]).any():
                    msg = f"Column '{col}' has values below minimum {constraints['min']}."
                    cls.logger.error(msg)
                    raise ValueError(msg)

        # Custom validations
        for name, validation_fn in cls.custom_validations.items():
            cls.logger.info(f"Running custom validation: {name}")
            validation_fn(df)

        cls.logger.info("Dataframe validation completed successfully.")

        return df
