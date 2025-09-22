import pandas as pd
from pandas import DataFrame
from pathlib import Path
from typing import Optional, Union, Generator
from itertools import tee
import collections.abc
from myapp.config.config_manager import ConfigManager
from myapp.utils.logger import CustomLogger
from myapp.components.data_validation import DataValidator


class DataValidationPipeline:
    """
    Data Validation Pipeline.

    Validates in-memory data or streamed data from files.
    Supports both eager and lazy loading modes (DataFrame or generator).
    Delegates validation logic to the `DataValidator` component.
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
        Run the data validation pipeline.

        Args:
            data: The data to validate, either as a DataFrame or a generator of DataFrames.

        Returns:
            Validated data (same type as input).

        Raises:
            Exception if validation fails.
        """
        try:
            self.logger.info("Starting data validation pipeline")

            validator = DataValidator(logger=self.logger)
            self.logger.debug(f"Data type for validation: {type(data)}")

            validated_data = validator.validate(data)

            if isinstance(validated_data, collections.abc.Iterator):
                # Use tee to preview without exhausting original
                validated_data, preview_data = tee(validated_data)
                try:
                    first_chunk = next(preview_data)
                    self.logger.info(f"First chunk preview:\n{first_chunk.head()}")
                except StopIteration:
                    self.logger.warning("No data returned by validator.")
            else:
                self.logger.info(f"Data preview:\n{validated_data.head()}")

            self.logger.info("Data validation completed successfully.")
            return validated_data

        except Exception as e:
            self.logger.error(f"Data validation failed: {e}", exc_info=True)
            raise
