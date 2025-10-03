import pandas as pd
import numpy as np
from typing import Dict, Callable, Optional
from myapp.utils.logger import CustomLogger


class PreprocessingSchema:
    """Schema definition for data preprocessing without scaling."""

    logger = CustomLogger(module_name=__name__).get_logger()

    # === Preprocessing strategy functions (no scaling) ===
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
    def extract_datetime_features(cls, df: pd.DataFrame, column: str) -> None:
        cls.logger.debug(f"Extracting datetime features from index for '{column}'")
        
        # Confirm index is datetime type, else convert
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            cls.logger.info("Converting index to datetime")
            df.index = pd.to_datetime(df.index, errors='coerce')

        df[f"{column}_year"] = df.index.year
        df[f"{column}_month"] = df.index.month
        df[f"{column}_day"] = df.index.day
        df[f"{column}_hour"] = df.index.hour
        df[f"{column}_minute"] = df.index.minute
        df[f"{column}_second"] = df.index.second
        df[f"{column}_dayofweek"] = df.index.dayofweek        

    # === Mapping for dynamic execution (no scaling) ===
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

    # === Column-specific preprocessing plan (no scaling) ===
    column_preprocessing_plan: Dict[str, Dict[str, Optional[str]]] = {
        "aep_mw": {
            "scaling": None,  # no scaling
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
        cls.logger.info("Starting preprocessing dataframe without scaling")
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

        cls.logger.info("Completed preprocessing dataframe without scaling")
        return df
