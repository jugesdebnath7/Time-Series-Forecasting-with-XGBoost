from pathlib import Path
import pandas as pd
from typing import Union, Generator, Callable, Optional
from myapp.utils.logger import CustomLogger


class DataIngestion:
    """
    Ingest data files (csv, json, parquet, xlsx) from a directory with eager or lazy loading.
    Lazy loading currently supports only 'csv' files.
    
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        file_type: str,
        lazy: bool = False,
        chunk_size: int = 100_000,
        logger: Optional[CustomLogger] = None
    ) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.exists() or not self.data_path.is_dir():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist or is not a directory.")
        
        self.file_type = file_type.lower()
        self.lazy = lazy
        self.chunk_size = chunk_size
        self.logger = logger or CustomLogger(module_name=__name__).get_logger()
        self.reader: Callable = self._get_reader()

    def _get_reader(self) -> Callable:
        if self.file_type == 'csv':
            return pd.read_csv
        elif self.file_type == 'json':
            return pd.read_json
        elif self.file_type == 'parquet':
            return pd.read_parquet
        elif self.file_type == 'xlsx':
            return pd.read_excel
        else:
            self.logger.error(f"Unsupported file type: {self.file_type}")
            raise ValueError(f"Unsupported file type: {self.file_type}")

    def ingest_data(self) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
        return self._lazy_ingest() if self.lazy else self._eager_ingest()

    def _eager_ingest(self) -> pd.DataFrame:
        files = list(self.data_path.glob(f"*.{self.file_type}"))
        self.logger.info(f"Eagerly ingesting data from {files}")

        if not files:
            self.logger.warning(f"No {self.file_type} files found in {self.data_path}")
            raise FileNotFoundError(f"No {self.file_type} files found in {self.data_path}")

        dfs = []
        for file in files:
            try:
                self.logger.info(f"Reading file eagerly: {file}")
                dfs.append(self.reader(file))
            except Exception as e:
                self.logger.error(f"Failed to read {file}: {e}", exc_info=True)

        if not dfs:
            self.logger.warning(f"No valid {self.file_type} files read successfully.")
            return pd.DataFrame()

        combined_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Data ingested successfully with shape {combined_df.shape}")
        return combined_df

    def _lazy_ingest(self) -> Generator[pd.DataFrame, None, None]:
        if self.file_type != 'csv':
            self.logger.error(f"Lazy loading not supported for {self.file_type}.")
            raise NotImplementedError(f"Lazy loading not supported for {self.file_type}")

        files = list(self.data_path.glob(f"*.{self.file_type}"))
        self.logger.info(f"Lazy ingestion (chunk size={self.chunk_size}) from files: {files}")

        if not files:
            self.logger.warning(f"No {self.file_type} files found in {self.data_path}")
            raise FileNotFoundError(f"No {self.file_type} files found in {self.data_path}")

        for file in files:
            try:
                self.logger.info(f"Reading file lazily in chunks: {file}")
                for chunk in self.reader(file, chunksize=self.chunk_size):
                    if not chunk.empty:
                        self.logger.debug(f"Yielding chunk with shape {chunk.shape} from {file}")
                        yield chunk
            except Exception as e:
                self.logger.error(f"Failed to read {file}: {e}", exc_info=True)
                continue
