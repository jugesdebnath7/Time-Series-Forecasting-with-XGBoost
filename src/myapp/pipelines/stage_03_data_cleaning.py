import pandas as pd
from pandas import DataFrame
from typing import Optional, Generator, Union
from itertools import tee
import collections.abc
from myapp.utils.logger import CustomLogger
from myapp.config.config_manager import ConfigManager
from myapp.components.data_cleaning import DataCleaner


class DataCleaningPipeline:
    """
    Cleans the input data using the configured cleaning logic.

    Supports both eager (DataFrame) and lazy (Generator) execution.
    Handles duplicates, nulls, or domain-specific inconsistencies as defined in `DataCleaner`.
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
        Run the data cleaning pipeline.

        Args:
            data: The data to clean, either as a DataFrame or a generator of DataFrames.

        Returns:
            Cleaned data (same type as input).

        Raises:
            Exception if cleaning fails.
        """
        try:
            self.logger.info("Starting data cleaning pipeline")

            cleaner = DataCleaner(logger=self.logger)

            self.logger.debug(f"Data type for cleaning: {type(data)}")
            cleaned_data = cleaner.clean(data)

            # Preview safely
            if isinstance(cleaned_data, collections.abc.Iterator):
                cleaned_data, preview = tee(cleaned_data)
                try:
                    first_chunk = next(preview)
                    self.logger.info(f"First chunk preview:\n{first_chunk.head()}")
                except StopIteration:
                    self.logger.warning("No data returned by cleaner.")
            else:
                self.logger.info(f"Data preview:\n{cleaned_data.head()}")

            self.logger.info("Data cleaning completed successfully.")
            return cleaned_data

        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}", exc_info=True)
            raise
