from typing import Generator, Union
import pandas as pd
import numpy as np
import collections.abc
from typing import Optional
from myapp.utils.logger import CustomLogger
from myapp.data.validation_schema import ValidationSchema


class DataValidator:
    """Validates dataframes or generators of dataframes against the defined schema."""

    def __init__(
        self, 
        logger: Optional[CustomLogger] = None
    ) -> None:
        self.logger = logger or CustomLogger(module_name=__name__).get_logger()
        self.schema = ValidationSchema()

    def validate(
        self, data: Union[pd.DataFrame, np.ndarray, Generator[pd.DataFrame, None, None]]
    ) -> Union[pd.DataFrame, np.ndarray, Generator[pd.DataFrame, None, None]]:
        if isinstance(data, pd.DataFrame):
            return self._validate_dataframe(data)

        elif isinstance(data, np.ndarray):
            self._validate_ndarray(data)
            return data

        elif isinstance(data, collections.abc.Generator):
            return self._validate_generator(data)

        else:
            self.logger.error(f"Unsupported data type: {type(data)}")
            raise ValueError("Unsupported data type for validation.")

    def _validate_dataframe(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.info("Validating pandas DataFrame...")
        validated_df = self.schema.validate_dataframe(df, copy=True)
        return validated_df

    def _validate_generator(
        self, 
        gen: Generator[pd.DataFrame, None, None]
    ) -> Generator[pd.DataFrame, None, None]:
        self.logger.info("Validating generator of DataFrames...")
        for chunk in gen:
            yield self.schema.validate_dataframe(chunk, copy=False)

    def _validate_ndarray(self, arr: np.ndarray) -> None:
        self.logger.info("Validating numpy ndarray...")
        if np.isnan(arr).any():
            self.logger.warning("Array contains NaN values.")
