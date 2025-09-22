from typing import Generator, Union
import pandas as pd
import numpy as np
import collections.abc
from typing import Optional
from myapp.utils.logger import CustomLogger
from myapp.schemas.feature_engineering_schema import FeatureEngineeringSchema


class FeatureEngineering:
    """
    Engineered features from the existing features to create new features for more robust predictions.
    
    """
    def __init__(
        self, 
        logger: Optional[CustomLogger] = None
    ) -> None:
        self.logger = logger or CustomLogger(module_name=__name__).get_logger()
        self.schema = FeatureEngineeringSchema()

    def features_engineered(
        self,
        data: Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]
    ) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
        if isinstance(data, pd.DataFrame):
            cleaned_df = self.schema._create_features(data)
            return cleaned_df

        elif hasattr(data, "__iter__"):
            return self._features_engineered_generator(data)

        else:
            self.logger.error("Unsupported data type for cleaning.")
            raise TypeError("DataCleaner only supports DataFrame or Generator of DataFrames.")

    def _features_engineered_generator(
        self, 
        data_gen: Generator[pd.DataFrame, None, None]
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Cleans a generator of DataFrames chunk by chunk, removes duplicates across chunks.
        """
        for chunk in data_gen:
            chunk = self.schema._create_features(chunk)
            yield chunk
