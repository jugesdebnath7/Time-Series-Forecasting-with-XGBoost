import pandas as pd
from typing import Optional, Union, Generator, List, Set
from myapp.utils.logger import CustomLogger


class DataCleaner:
    def __init__(
        self,
        datetime_cols: Optional[List[str]] = None,
        rename_map: Optional[dict] = None,
        sort_by: Optional[str] = None,
        drop_duplicates: bool = True,
        logger: Optional[CustomLogger] = None,
    ):
        self.datetime_cols = datetime_cols or ["datetime"]
        self.rename_map = rename_map or {}
        self.sort_by = sort_by or self.datetime_cols[0]
        self.drop_duplicates = drop_duplicates
        self.logger = logger or CustomLogger(module_name=__name__).get_logger()

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.rename_map:
            try:
                df = df.rename(columns=self.rename_map)
                self.logger.info(f"Renamed columns as per map: {self.rename_map}")
            except Exception as e:
                self.logger.error(f"Failed to rename columns: {e}", exc_info=True)
        return df

    def _convert_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.datetime_cols:
            if col not in df.columns:
                self.logger.error(f"Datetime column '{col}' not found in DataFrame.")
                continue
            try:
                original_non_null = df[col].notnull().sum()
                df[col] = pd.to_datetime(df[col], errors="coerce")
                converted_non_null = df[col].notnull().sum()
                if converted_non_null < original_non_null:
                    self.logger.warning(
                        f"Datetime conversion coerced {original_non_null - converted_non_null} invalid values to NaT in column '{col}'."
                    )
                else:
                    self.logger.info(f"Converted column '{col}' to datetime.")
            except Exception as e:
                self.logger.error(f"Error converting column '{col}' to datetime: {e}", exc_info=True)
        return df

    def _sort_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sort_by and self.sort_by in df.columns:
            try:
                df = df.sort_values(by=self.sort_by).reset_index(drop=True)
                self.logger.info(f"Sorted dataframe by '{self.sort_by}'.")
            except Exception as e:
                self.logger.error(f"Error sorting dataframe by '{self.sort_by}': {e}", exc_info=True)
        else:
            if self.sort_by:
                self.logger.warning(f"Sort column '{self.sort_by}' not found in DataFrame; skipping sorting.")
        return df

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.drop_duplicates:
            before = df.shape[0]
            try:
                df = df.drop_duplicates()
                after = df.shape[0]
                dropped = before - after
                self.logger.info(f"Dropped {dropped} duplicate rows.")
            except Exception as e:
                self.logger.error(f"Error dropping duplicates: {e}", exc_info=True)
        return df

    def _apply_cleaning_steps(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._rename_columns(df)
        df = self._convert_datetime(df)
        df = self._sort_dataframe(df)
        df = self._drop_duplicates(df)
        return df

    def clean(
        self,
        data: Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]
    ) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
        if isinstance(data, pd.DataFrame):
            cleaned_df = self._apply_cleaning_steps(data)
            cleaned_df = self._remove_internal_duplicates(cleaned_df)
            return cleaned_df

        elif hasattr(data, "__iter__"):
            return self._clean_generator(data)

        else:
            self.logger.error("Unsupported data type for cleaning.")
            raise TypeError("DataCleaner only supports DataFrame or Generator of DataFrames.")

    def _remove_internal_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates based on datetime_cols within a single DataFrame
        """
        key_cols = self.datetime_cols
        if all(col in df.columns for col in key_cols):
            before = df.shape[0]
            df = df.drop_duplicates(subset=key_cols)
            after = df.shape[0]
            dropped = before - after
            if dropped > 0:
                self.logger.warning(f"Removed {dropped} duplicate rows based on columns {key_cols} within DataFrame.")
        else:
            missing_cols = [col for col in key_cols if col not in df.columns]
            self.logger.error(f"Cannot remove duplicates - missing columns: {missing_cols}")
        return df

    def _clean_generator(
        self, 
        data_gen: Generator[pd.DataFrame, None, None]
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Cleans a generator of DataFrames chunk by chunk, removes duplicates across chunks.
        """
        seen_keys: Set = set()
        key_cols = self.datetime_cols
        for chunk in data_gen:
            chunk = self._apply_cleaning_steps(chunk)

            # Remove duplicates within chunk first
            chunk = self._remove_internal_duplicates(chunk)

            # Remove duplicates across chunks based on key columns
            if not all(col in chunk.columns for col in key_cols):
                missing_cols = [col for col in key_cols if col not in chunk.columns]
                self.logger.error(f"Chunk missing columns for duplicate removal: {missing_cols}. Yielding chunk as is.")
                yield chunk
                continue

            # Build boolean mask for rows not seen before
            def row_key(row):
                # Create a tuple key of all datetime column values for uniqueness
                return tuple(row[col] for col in key_cols)

            mask = chunk.apply(lambda row: row_key(row) not in seen_keys, axis=1)

            filtered_chunk = chunk.loc[mask].copy()

            # Update seen keys
            new_keys = set(filtered_chunk.apply(row_key, axis=1))
            seen_keys.update(new_keys)

            dropped = chunk.shape[0] - filtered_chunk.shape[0]
            if dropped > 0:
                self.logger.warning(f"Removed {dropped} duplicate rows across chunks based on columns {key_cols}.")

            yield filtered_chunk
