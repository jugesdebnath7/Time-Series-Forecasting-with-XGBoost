import pandas as pd
import collections.abc
from pandas import DataFrame
from pathlib import Path
from typing import Optional, Union, Generator
from itertools import tee
from myapp.config.config_manager import ConfigManager
from myapp.utils.logger import CustomLogger
from myapp.components.data_ingestion import DataIngestion


class DataIngestionPipeline:
    """
    Ingests raw data from disk in a configurable and format-agnostic way.

    Supports lazy (chunked) and eager (full load) modes depending on configuration.
    Automatically detects supported file types (CSV, JSON, XLSX, Parquet).
    """

    def __init__(
        self,
        config: ConfigManager,
        logger: Optional[CustomLogger] = None
    ) -> None:
        self.config = config
        self.logger = logger or CustomLogger(module_name=__name__).get_logger()

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

            ingestion_engine = DataIngestion(
                data_path=self.config.paths.raw,
                file_type=file_type,
                lazy=self.config.data.lazy,
                chunk_size=self.config.data.chunk_size,
                logger=self.logger
            )

            raw_data = ingestion_engine.ingest_data()

            # Safe preview using tee (does not consume original generator)
            if isinstance(raw_data, collections.abc.Iterator):
                raw_data, preview = tee(raw_data)
                try:
                    first_chunk = next(preview)
                    self.logger.info(f"First chunk preview:\n{first_chunk.head()}")
                except StopIteration:
                    self.logger.warning("No data returned by ingestor.")
            else:
                self.logger.info(f"Data preview:\n{raw_data.head()}")

            self.logger.info("Data ingestion completed successfully.")
            return raw_data

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
