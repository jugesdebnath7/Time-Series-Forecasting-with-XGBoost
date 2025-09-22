import pandas as pd
from pandas import DataFrame
from typing import Optional, Generator, Union
from itertools import tee
import collections.abc
from myapp.utils.logger import CustomLogger
from myapp.config.config_manager import ConfigManager
from myapp.components.feature_engineering import FeatureEngineering


class FeatureEngineeringPipeline:
    """
    Feature Engineering Pipeline.

    Enhances the dataset by generating time-based, statistical, and cyclical features.
    Supports eager and lazy (chunked) processing modes.
    """

    def __init__(
        self,
        config: ConfigManager,
        logger: Optional[CustomLogger] = None
    ) -> None:
        self.config = config
        self.logger = logger or CustomLogger(module_name=__name__).get_logger()

    def run(
        self, 
        data: Union[DataFrame, Generator[DataFrame, None, None]]
    ) -> Union[DataFrame, Generator[DataFrame, None, None]]:
        """
        Run the feature engineering pipeline.

        Args:
            data: Input data, either as a DataFrame or a generator of DataFrames.

        Returns:
            Data with engineered features (same type as input).

        Raises:
            Exception if feature engineering fails.
        """
        try:
            self.logger.info("Starting feature engineering pipeline")

            features = FeatureEngineering(logger=self.logger)
            self.logger.debug(f"Data type for feature engineering: {type(data)}")

            engineered_features = features.features_engineered(data)

            # Peek into the generator safely
            if isinstance(engineered_features, collections.abc.Iterator):
                engineered_features, preview = tee(engineered_features)
                try:
                    first_chunk = next(preview)
                    self.logger.info(f"First chunk preview:\n{first_chunk.head()}")
                except StopIteration:
                    self.logger.warning("No data returned by feature engineering.")
                self.logger.info("Feature engineering completed successfully (generator mode).")
            else:
                self.logger.info(f"Data preview:\n{engineered_features.head()}")
                self.logger.info("Feature engineering completed successfully (DataFrame mode).")

            return engineered_features

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}", exc_info=True)
            raise
