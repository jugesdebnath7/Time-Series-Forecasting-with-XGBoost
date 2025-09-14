import pandas as pd
from pandas import DataFrame
from pathlib import Path
from typing import Optional, Union, Generator
from myapp.config.config_manager import ConfigManager
from myapp.utils.logger import CustomLogger
from myapp.components.data_validation import DataValidator


class DataValidationPipeline:
    """
    Data Validation Pipeline.

    Validates data files from the configured directory in various supported formats.
    Supports eager and lazy loading modes depending on configuration.
    """

    def __init__(
        self,
        config: ConfigManager,
        logger: Optional[CustomLogger] = None
    ):
        self.config = config
        self.logger = logger or CustomLogger(name=__name__).get_logger()

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

            self.logger.info("Data validation completed successfully.")
            return validated_data

        except Exception as e:
            self.logger.error(f"Data validation failed: {e}", exc_info=True)
            raise
