import pandas as pd
import numpy as np
from typing import Dict, Callable, Optional
from myapp.utils.logger import CustomLogger


class PreprocessingSchema:
    """Schema definition for data preprocessing."""

    logger = CustomLogger(module_name=__name__).get_logger()

    # === Preprocessing strategy functions ===
    @classmethod
    def normalize_minmax(
        cls, 
        df: pd.DataFrame, 
        column: str
    ) -> None:
        cls.logger.debug(f"Normalizing column '{column}' with min-max scaling")
        min_val, max_val = df[column].min(), df[column].max()
        if min_val != max_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
        else:
            cls.logger.debug(f"Column '{column}' has constant value; skipping min-max normalization")

    @classmethod
    def standardize_zscore(
        cls,
        df: pd.DataFrame, 
        column: str
    ) -> None:
        cls.logger.debug(f"Standardizing column '{column}' with z-score")
        mean = df[column].mean()
        std = df[column].std()
        df[column] = (df[column] - mean) / std

    @classmethod
    def log_transform(
        cls, 
        df: pd.DataFrame, 
        column: str
    ) -> None:
        cls.logger.debug(f"Applying log transform on column '{column}'")
        df[column] = df[column].apply(lambda x: np.log(x) if pd.notnull(x) and x > 0 else None)

    @classmethod
    def one_hot_encode(
        cls, 
        df: pd.DataFrame, 
        column: str
    ) -> None:
        cls.logger.debug(f"One-hot encoding column '{column}'")
        dummies = pd.get_dummies(df[column], prefix=column)
        df.drop(columns=[column], inplace=True)
        df[dummies.columns] = dummies

    @classmethod
    def label_encode(
        cls, 
        df: pd.DataFrame, 
        column: str
    ) -> None:
        cls.logger.debug(f"Label encoding column '{column}'")
        df[column] = df[column].astype('category').cat.codes

    @classmethod      
    def extract_datetime_features(
        cls,
        df: pd.DataFrame,
        column: str
    ) -> None:
        cls.logger.debug(f"Extracting datetime features from column '{column}'")
        df[column] = pd.to_datetime(df[column], errors='coerce')
        df[f"{column}_year"] = df[column].dt.year
        df[f"{column}_month"] = df[column].dt.month
        df[f"{column}_day"] = df[column].dt.day
        df[f"{column}_hour"] = df[column].dt.hour
        df[f"{column}_minute"] = df[column].dt.minute
        df[f"{column}_second"] = df[column].dt.second
        df[f"{column}_dayofweek"] = df[column].dt.dayofweek 
        
        

    # === Mapping for dynamic execution ===
    scaling_strategies: Dict[str, Callable[[pd.DataFrame, str], None]] = {
        "minmax": normalize_minmax.__func__,
        "zscore": standardize_zscore.__func__,
    }

    encoding_strategies: Dict[str, Callable[[pd.DataFrame, str], None]] = {
        "onehot": one_hot_encode.__func__,
        "label": label_encode.__func__,
    }

    transformation_strategies: Dict[str, Callable[[pd.DataFrame, str], None]] = {
        "log": log_transform.__func__,
        "datetime_features": extract_datetime_features.__func__,
    }

    feature_extraction_strategies: Dict[str, Callable[[pd.DataFrame, str], None]] = {
        "datetime_features": extract_datetime_features.__func__,
    }

    # === Column-specific preprocessing plan ===
    column_preprocessing_plan: Dict[str, Dict[str, Optional[str]]] = {
        "aep_mw": {
            "scaling": "minmax",
            "encoding": None,
            "transformation": None,
            "feature_extraction": None,
        },
        "datetime": {
            "scaling": None,
            "encoding": None,
            "transformation": None,
            "feature_extraction": "datetime_features",
        },
    }

    @classmethod
    def get_preprocessing_steps(
        cls, 
        column: str
    ) -> Dict[str, Optional[str]]:
        """Retrieve the preprocessing steps for a specific column."""
        return cls.column_preprocessing_plan.get(column, {})

    @classmethod
    def preprocess_dataframe(
        cls, 
        df: pd.DataFrame, 
        copy: bool = False
    ) -> pd.DataFrame:
        cls.logger.info("Starting preprocessing dataframe")
        if copy:
            df = df.copy()

        for column, steps in cls.column_preprocessing_plan.items():
            cls.logger.debug(f"Preprocessing column '{column}' with steps {steps}")
            for step_name, strategy_key in steps.items():
                if not strategy_key:
                    continue

                strategy_map = getattr(cls, f"{step_name}_strategies", None)
                if not strategy_map:
                    cls.logger.warning(f"No strategy map found for step '{step_name}'")
                    continue

                strategy_func = strategy_map.get(strategy_key)
                if strategy_func:
                    cls.logger.debug(f"Applying {step_name} strategy '{strategy_key}' on column '{column}'")
                    strategy_func(cls, df, column)
                else:
                    cls.logger.warning(f"No strategy function found for key '{strategy_key}' in step '{step_name}'")

        cls.logger.info("Completed preprocessing dataframe")
        return df
