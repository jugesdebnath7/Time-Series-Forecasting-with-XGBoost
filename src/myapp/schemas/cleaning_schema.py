import pandas as pd
import numpy as np
from typing import Dict, Callable, Optional, List
from myapp.utils.logger import CustomLogger
from myapp.utils.column_mappings import get_rename_map


class CleaningSchema:
    """
    Schema definition for data cleaning.
    """
    #=== Setup ====
    logger = CustomLogger(module_name=__name__).get_logger()
    
    # Schema-wide configurations
    datetime_columns: List[str] = ["datetime"]
    sort_by_column: Optional[str] = "datetime"
    drop_dupes: bool = True
    rename_map: Dict[str, str] = get_rename_map()
    
    # === 1. Global Data Cleaning Steps ===
    @classmethod
    def rename_columns(
        cls, 
        df:pd.DataFrame
    ) -> pd.DataFrame:
        cls.logger.debug("Renaming columns using provided mapping.")
        return df.rename(columns=cls.rename_map)
    
    @classmethod
    def convert_datetime_columns(
        cls,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        for col in cls.datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                cls.logger.debug(f"Converted column '{col}' to datetime.")
            else:
                cls.logger.warning(f"Datetime column '{col}' not found in DataFrame.")
        return df
    
    @classmethod
    def sort_dataframe(
        cls,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        if cls.sort_by_column and cls.sort_by_column in df.columns:
             cls.logger.debug(f"Sorting DataFrame by '{cls.sort_by_column}'")
             return df.sort_values(by=cls.sort_by_column).reset_index(drop=True)
        cls.logger.debug("Sort column not found or not specified; skipping sorting")
        return df
    
    @classmethod
    def drop_duplicates(
        cls, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        if cls.drop_dupes:
            before = df.shape[0]
            try:
                df = df.drop_duplicates()
                after = df.shape[0]
                cls.logger.info(f"Dropped {before - after} duplicate rows.")
            except Exception as e:
                cls.logger.error(f"Error dropping duplicates: {e}", exc_info=True)
        return df
    
    @classmethod
    def remove_internal_duplicates(
        cls, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        key_cols = cls.datetime_columns
        if all(col in df.columns for col in key_cols):
            before = df.shape[0]
            try:
                df = df.drop_duplicates(subset=key_cols)
                after = df.shape[0]
                cls.logger.info(f"Removed {before - after} internal duplicates based on {key_cols}.")
            except Exception as e:
                cls.logger.error(f"Error dropping internal duplicates: {e}", exc_info=True)
        else:
            missing = [c for c in key_cols if c not in df.columns]
            cls.logger.warning(f"Missing columns for internal duplicate removal: {missing}")        
        return df
    
     # === 2. Column-Specific Cleaning Strategies ===
    @classmethod
    def fill_mean(
        cls, 
        df: pd.DataFrame, 
        column: str
    ) -> pd.DataFrame:
        cls.logger.debug(f"Filling missing values in '{column}' with mean")
        df[column] = df[column].fillna(df[column].mean())
        return df

    @classmethod
    def fill_median(
        cls, 
        df: pd.DataFrame, 
        column: str
    ) -> pd.DataFrame:
        cls.logger.debug(f"Filling missing values in '{column}' with median")
        df[column] = df[column].fillna(df[column].median())
        return df

    @classmethod
    def fill_mode(
        cls, 
        df: pd.DataFrame, 
        column: str
    ) -> pd.DataFrame:
        cls.logger.debug(f"Filling missing values in '{column}' with mode")
        df[column] = df[column].fillna(df[column].mode()[0])
        return df
    
    @classmethod
    def fill_ffill(
        cls, 
        df: pd.DataFrame, 
        column: str
    ) -> pd.DataFrame:
        cls.logger.debug(f"Forward-filling missing values in '{column}'")
        df[column] = df[column].fillna(method="ffill")
        return df

    @classmethod
    def fill_bfill(
        cls, 
        df: pd.DataFrame, 
        column: str
    ) -> pd.DataFrame:
        cls.logger.debug(f"Backward-filling missing values in '{column}'")
        df[column] = df[column].fillna(method="bfill")
        return df
    
    @classmethod
    def detect_outliers_iqr(
        cls,
        df: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        cls.logger.debug(f"Removing outliers in '{column}' using IQR method")
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[column] = df[column].where(df[column].between(lower, upper))
        return df
    
    @classmethod
    def detect_outliers_zscore(
        cls,
        df: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        mean = df[column].mean()
        std = df[column].std()
        df[column] = df[column].where((df[column] - mean).abs() <= 3 * std) 
        return df
    
    # === 3. Strategy Maps for Dynamic Execution ===
    missing_value_strategies: Dict[str, Callable[[pd.DataFrame, str], pd.DataFrame]] = {
        "mean": fill_mean.__func__, 
        "median": fill_median.__func__,
        "mode": fill_mode.__func__,
        "ffill": fill_ffill.__func__,
        "bfill": fill_bfill.__func__,
    }
    
    outlier_detection_strategies: Dict[str, Callable[[pd.DataFrame, str], pd.DataFrame]] = {
        "iqr": detect_outliers_iqr.__func__,
        "zscore": detect_outliers_zscore.__func__,
    }
    
    # === 4. Column Cleaning Plan ===
    column_cleaning_plan: Dict[str, Dict[str, Optional[str]]] = {
        "aep_mw": {
            "missing_value": "mean",
            "outlier_detection": "iqr",
        },
        "datetime": {
            "missing_value": None,
            "outlier_detection": None,
        },
    }
    
    # === 5. Schema-Aware Cleaning Execution ===
    @classmethod
    def clean_dataframe(
        cls,
        df: pd.DataFrame,
        copy: bool = False
    ) -> pd.DataFrame:
        cls.logger.info("Starting full data cleaning pipeline")
        
        if copy:
            df = df.copy()
            
        # Global steps
        df = cls.rename_columns(df)
        df = cls.convert_datetime_columns(df)
        df = cls.sort_dataframe(df)
        df = cls.drop_duplicates(df)
        df = cls.remove_internal_duplicates(df)   
        
        # Column-wise cleaning: First apply outlier detection
        for column, steps in cls.column_cleaning_plan.items():
            outlier_key = steps.get("outlier_detection")
            if outlier_key:
                strategy_func = cls.outlier_detection_strategies.get(outlier_key)
                if strategy_func:
                    cls.logger.debug(f"Applying 'outlier_detection' strategy '{outlier_key}' on column '{column}'")
                    df = strategy_func(cls, df, column)
        
        # Then apply missing value filling
        for column, steps in cls.column_cleaning_plan.items():
            mv_key = steps.get("missing_value")
            if mv_key:
                strategy_func = cls.missing_value_strategies.get(mv_key)
                if strategy_func:
                    cls.logger.debug(f"Applying 'missing_value' strategy '{mv_key}' on column '{column}'")
                    df = strategy_func(cls, df, column)

        cls.logger.info("Data cleaning completed successfully")
        return df
