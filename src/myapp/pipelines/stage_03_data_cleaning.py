import pandas as pd
from pandas import DataFrame
from myapp.utils.logger import CustomLogger
from myapp.utils.column_mappings import get_rename_map
from myapp.config.config_manager import ConfigManager
from myapp.components.data_cleaning import DataCleaner
from typing import Optional, Generator, Union, Dict


class DataCleaningPipeline:
    
    def __init__(
        self,
        config: ConfigManager,
        rename_map: Optional[Dict[str, str]] = None,
        logger: Optional[CustomLogger] = None
    ) -> None:
        self.config = config
        self.rename_map = rename_map or get_rename_map()
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

            cleaner = DataCleaner(
                logger=self.logger,
                rename_map=self.rename_map
            )
            
            self.logger.debug(f"Data type for cleaning: {type(data)}")
            cleaned_data = cleaner.clean(data)
            
            # Temporary debug: peek into the data to verify ingestion and flow
            if isinstance(cleaned_data, Generator):
                try:
                    first_chunk = next(cleaned_data)
                    self.logger.info(f"First chunk preview:\n{first_chunk.head()}")
                    # Re-create the generator so downstream stages can consume it fresh
                    cleaned_data = cleaner.clean(data)
                except StopIteration:
                    self.logger.warning("No data returned by ingestion.")
            else:
                self.logger.info(f"Data preview:\n{cleaned_data.head()}")
                

            self.logger.info("Data cleaning completed successfully.")
            return cleaned_data

        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}", exc_info=True)
            raise
