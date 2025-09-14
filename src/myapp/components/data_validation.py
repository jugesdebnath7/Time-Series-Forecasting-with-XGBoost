import pandas as pd
import numpy as np
from typing import Optional, Union, Generator
from myapp.utils.logger import CustomLogger
from myapp.data.data_validation_schema import DataValidationSchema


class DataValidator:
    """Validates dataframes or generators of dataframes against the defined schema."""

    def __init__(
        self, 
        logger: Optional[CustomLogger] = None
    ) -> None:
        self.logger = logger or CustomLogger(name=__name__).get_logger()
        self.schema = DataValidationSchema()

    def validate(
        self,
        data: Union[pd.DataFrame, np.ndarray, Generator[pd.DataFrame, None, None]]
    ) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
        """
        Validate the data (DataFrame, ndarray, or Generator of DataFrames).

        Args:
            data: The data to validate.

        Returns:
            The validated data (same type as input).

        Raises:
            ValueError: If data is invalid or unsupported type.
        """
        if isinstance(data, pd.DataFrame):
            return self._validate_dataframe(data)

        elif isinstance(data, np.ndarray):
            self._validate_ndarray(data)
            return data

        elif hasattr(data, "__iter__") and not isinstance(data, (pd.DataFrame, np.ndarray)):
            return self._validate_generator(data)

        else:
            self.logger.error(f"Unsupported data type: {type(data)}")
            raise ValueError("Unsupported data type for validation.")

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Validating pandas DataFrame...")

        # Step 1: Rename columns
        df = df.rename(columns=self.schema.rename_map)

        # Step 2: Check required columns
        missing_cols = [col for col in self.schema.required_columns if col not in df.columns]
        if missing_cols:
            msg = f"Missing required columns: {missing_cols}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Step 3: Validate column data types
        for col, expected_dtype in self.schema.required_columns.items():
            if col in df.columns:
                if not pd.api.types.is_dtype_equal(df[col].dtype, expected_dtype):
                    self.logger.warning(
                        f"Column '{col}' has dtype '{df[col].dtype}', expected '{expected_dtype}'"
                    )

        # Step 4: Value constraints (e.g., non-negative)
        for col, constraints in self.schema.value_constraints.items():
            if col in df.columns:
                if "min" in constraints:
                    if (df[col] < constraints["min"]).any():
                        raise ValueError(f"Column '{col}' has values below minimum {constraints['min']}.")

        # Step 5: Custom validations from schema
        for name, validation_fn in self.schema.custom_validations.items():
            try:
                validation_fn(df)
                self.logger.info(f"Custom validation passed: {name}")
            except Exception as e:
                self.logger.error(f"Validation failed - {name}: {e}")
                raise

        return df

    def _validate_ndarray(self, arr: np.ndarray) -> None:
        self.logger.info("Validating numpy ndarray...")
        if np.isnan(arr).any():
            self.logger.warning("Array contains NaN values.")

    def _validate_generator(
        self,
        data_gen: Generator[pd.DataFrame, None, None]
    ) -> Generator[pd.DataFrame, None, None]:
        self.logger.info("Validating data generator (lazy loading)...")

        for chunk in data_gen:
            yield self._validate_dataframe(chunk)
