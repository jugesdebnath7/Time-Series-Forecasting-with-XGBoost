from typing import Generator, Union
import pandas as pd
import numpy as np
import collections.abc
from typing import Optional
from myapp.utils.logger import CustomLogger
from myapp.schemas.preprocessing_schema import PreprocessingSchema


class DataPreprocessor:
    """Preprocesses dataframes or generators of dataframes based on the defined schema."""

    def __init__(
        self,
        logger: Optional[CustomLogger] = None
    ) -> None:
        self.logger = logger or CustomLogger(module_name=__name__).get_logger()
        self.schema = PreprocessingSchema()

    def preprocess(
        self,
        data: Union[pd.DataFrame, np.ndarray, Generator[pd.DataFrame, None, None]]
    ) -> Union[pd.DataFrame, np.ndarray, Generator[pd.DataFrame, None, None]]:
        if isinstance(data, pd.DataFrame):
            return self._preprocess_dataframe(data)

        elif isinstance(data, np.ndarray):
            self.logger.info("Received ndarray, no preprocessing defined.")
            return data

        elif isinstance(data, collections.abc.Iterator):
            return self._preprocess_generator(data)

        else:
            self.logger.error(f"Unsupported data type: {type(data)}")
            raise ValueError("Unsupported data type for preprocessing.")

    def _preprocess_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.info("Preprocessing pandas DataFrame...")
        processed_df = self.schema.preprocess_dataframe(df, copy=True)
        return processed_df

    def _preprocess_generator(
        self,
        gen: Generator[pd.DataFrame, None, None]
    ) -> Generator[pd.DataFrame, None, None]:
        self.logger.info("Preprocessing generator of DataFrames...")
        for chunk in gen:
            yield self.schema.preprocess_dataframe(chunk)
