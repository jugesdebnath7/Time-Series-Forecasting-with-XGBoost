import pandas as pd
from pandas import DataFrame
from pathlib import Path
from typing import Optional, Union, Generator
from itertools import tee
import collections.abc
from myapp.config.config_manager import ConfigManager
from myapp.utils.logger import CustomLogger
from myapp.components.data_preprocessing import DataPreprocessor


class DataPreprocessingPipeline:
    """
    Data Preprocessing Pipeline.

    Applies feature scaling, type casting, time formatting, or other domain-specific preprocessing.

    Supports both eager (DataFrame) and lazy (Generator) execution.
    """

    def __init__(
        self,
        config: ConfigManager,
        logger: Optional[CustomLogger] = None
    ):
        self.config = config
        self.logger = logger or CustomLogger(module_name=__name__).get_logger()

    def run(
        self, 
        data: Union[DataFrame, Generator[DataFrame, None, None]]
    ) -> Union[DataFrame, Generator[DataFrame, None, None]]:
        """
        Run the data preprocessing pipeline.

        Args:
            data: The data to preprocess, either as a DataFrame or a generator of DataFrames.

        Returns:
            Preprocessed data (same type as input).

        Raises:
            Exception if preprocessing fails.
        """
        try:
            self.logger.info("Starting data preprocessing pipeline")

            preprocessor = DataPreprocessor(logger=self.logger)

            self.logger.debug(f"Data type for preprocessing: {type(data)}")
            preprocessed_data = preprocessor.preprocess(data)

            # Preview chunk safely
            if isinstance(preprocessed_data, collections.abc.Iterator):
                preprocessed_data, preview = tee(preprocessed_data)
                try:
                    first_chunk = next(preview)
                    self.logger.info(f"First chunk preview:\n{first_chunk.head()}")
                except StopIteration:
                    self.logger.warning("No data returned by preprocessor.")
            else:
                self.logger.info(f"Data preview:\n{preprocessed_data.head()}")

            self.logger.info("Data preprocessing completed successfully.")
            return preprocessed_data

        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}", exc_info=True)
            raise
