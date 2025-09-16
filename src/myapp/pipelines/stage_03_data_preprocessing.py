import pandas as pd
from pandas import DataFrame
from pathlib import Path
from typing import Optional, Union, Generator
from myapp.config.config_manager import ConfigManager
from myapp.utils.logger import CustomLogger
from myapp.components.data_preprocessing import DataPreprocessor


class DataPreprocessingPipeline:
    """
    Data Preprocessing Pipeline.

    Preprocesses data files from the configured directory in various supported formats.
    Supports eager and lazy loading modes depending on configuration.
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

            self.logger.info("Data preprocessing completed successfully.")
            return preprocessed_data

        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}", exc_info=True)
            raise