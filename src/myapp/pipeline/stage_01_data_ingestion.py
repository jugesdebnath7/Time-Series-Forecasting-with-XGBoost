import pandas as pd
from pandas import DataFrame
from pathlib import Path
from typing import Optional, Union, Generator
from logging import Logger
from myapp.config.config_manager import ConfigManager
from myapp.utils.logger import CustomLogger
from myapp.components.data_ingestion import Ingestion


class DataIngestionPipeline:
    """
    Data Ingestion Pipeline.

    Reads data files from the configured directory in various supported formats.
    Supports eager and lazy loading modes depending on configuration.
    """

    def __init__(
        self,
        config: ConfigManager,
        logger: Optional[Logger] = None
    ):
        self.config = config
        self.logger = logger or CustomLogger(name=__name__).get_logger()

    def run(self) -> Union[DataFrame, Generator[DataFrame, None, None]]:
        """
        Run the data ingestion pipeline.

        Returns:
            DataFrame or generator of DataFrames (if lazy loading enabled)

        Raises:
            Exception if ingestion fails or no files found.
        """
        try:
            file_type = self._detect_file_type()
            self.logger.info(f"Starting data ingestion pipeline with file type: {file_type}")

            ingestion = Ingestion(
                data_path=self.config.paths.raw,
                file_type=file_type,
                lazy=self.config.data.lazy,
                chunk_size=self.config.data.chunk_size,
                logger=self.logger
            )

            data = ingestion.ingest_data()
            self.logger.info("Data ingestion completed successfully.")
            return data

        except Exception as e:
            self.logger.error(f"Data ingestion failed: {e}", exc_info=True)
            raise

    def _detect_file_type(self) -> str:
        """
        Detect the file type by scanning the raw data directory.

        Returns:
            str: Detected file type.

        Raises:
            ValueError: If no supported file types are found.
        """
        supported_types = ['csv', 'json', 'xlsx', 'parquet']
        raw_path = Path(self.config.paths.raw)

        for file_type in supported_types:
            if any(raw_path.glob(f"*.{file_type}")):
                self.logger.info(f"Detected file type: {file_type}")
                return file_type

        error_msg = "No supported file types found in the data directory."
        self.logger.error(error_msg)
        raise ValueError(error_msg)
