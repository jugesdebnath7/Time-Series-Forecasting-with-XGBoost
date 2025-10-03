# myapp/schemas/feature_engineering_schema.py

import pandas as pd
import numpy as np
import holidays
from myapp.utils.logger import CustomLogger


class FeatureEngineeringSchema:
    """
    Schema for applying engineered features to time-series energy data.

    Assumes:
    - DataFrame has datetime index of type datetime64[ns].
    - Datetime feature columns (e.g., datetime_year, datetime_month, etc.) are pre-extracted.
    """

    logger = CustomLogger(module_name=__name__).get_logger()

    @classmethod
    def _add_holiday_flags(
        cls, 
        df: pd.DataFrame
    ) -> None:
        cls.logger.debug("Adding holiday and weekend flags")

        # Weekend flag: Saturday (5) or Sunday (6)
        if 'datetime_dayofweek' not in df.columns:
            raise KeyError("Missing required column 'datetime_dayofweek' for holiday flags.")
        df['is_weekend'] = df['datetime_dayofweek'].isin([5, 6]).astype(int)

        # Holiday detection based on datetime index normalized to date
        if 'datetime_year' not in df.columns:
            raise KeyError("Missing required column 'datetime_year' for holiday flags.")
        unique_years = df['datetime_year'].unique()
        us_holidays = holidays.US(years=unique_years)
        df['is_holiday'] = df.index.normalize().isin(us_holidays).astype(int)

        # New Year's Eve flag
        if 'datetime_month' not in df.columns or 'datetime_day' not in df.columns:
            raise KeyError("Missing required columns 'datetime_month' or 'datetime_day' for New Year's Eve flag.")
        df['is_new_year_eve'] = ((df['datetime_month'] == 12) & (df['datetime_day'] == 31)).astype(int)

    @classmethod
    def _add_lag_features(
        cls, 
        df: pd.DataFrame
    ) -> None:
        cls.logger.debug("Adding lag features")
        df['lag_24'] = df['aep_mw'].shift(24)
        # Add more lags if needed
        # df['lag_168'] = df['aep_mw'].shift(168)

    @classmethod
    def _add_rolling_stats(
        cls, 
        df: pd.DataFrame
    ) -> None:
        cls.logger.debug("Adding rolling mean and std (24h window)")
        df['rolling_mean_24'] = df['aep_mw'].shift(1).rolling(window=24).mean()
        df['rolling_std_24'] = df['aep_mw'].shift(1).rolling(window=24).std()

    @classmethod
    def _add_cyclical_encoding(
        cls, 
        df: pd.DataFrame
    ) -> None:
        cls.logger.debug("Adding cyclical encodings for hour, dayofweek, and month")

        for col, period in [('datetime_hour', 24), ('datetime_dayofweek', 7), ('datetime_month', 12)]:
            if col not in df.columns:
                raise KeyError(f"Missing required column '{col}' for cyclical encoding.")
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)

    @classmethod
    def _add_interaction_features(
        cls, 
        df: pd.DataFrame
    ) -> None:
        cls.logger.debug("Adding interaction features between hour and holiday/weekend flags")

        if 'datetime_hour' not in df.columns or 'is_holiday' not in df.columns or 'is_weekend' not in df.columns:
            raise KeyError("Missing required columns for interaction features.")
        df['hour_is_holiday'] = df['datetime_hour'] * df['is_holiday']
        df['hour_is_weekend'] = df['datetime_hour'] * df['is_weekend']

    @classmethod
    def _add_time_of_day_flags(
        cls, 
        df: pd.DataFrame
    ) -> None:
        cls.logger.debug("Adding time of day flags")

        if 'datetime_hour' not in df.columns:
            raise KeyError("Missing required column 'datetime_hour' for time of day flags.")
        hour = df['datetime_hour']

        df['is_night'] = hour.isin(range(0, 6)).astype(int)
        df['is_morning'] = hour.isin(range(6, 12)).astype(int)
        df['is_noon'] = hour.isin(range(12, 18)).astype(int)
        df['is_evening'] = hour.isin(range(18, 24)).astype(int)

    @classmethod
    def _add_outlier_flag(
        cls, 
        df: pd.DataFrame
    ) -> None:
        cls.logger.debug("Adding outlier flag based on IQR method")

        Q1 = df['aep_mw'].quantile(0.25)
        Q3 = df['aep_mw'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df['is_outlier'] = ((df['aep_mw'] < lower) | (df['aep_mw'] > upper)).astype(int)

    @classmethod
    def _create_features(
        cls, 
        df: pd.DataFrame, 
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps to the dataframe.

        Parameters:
            df (pd.DataFrame): DataFrame with datetime index and pre-extracted datetime feature columns.
            drop_na (bool): Whether to drop rows with NA values introduced by lag and rolling window operations.

        Returns:
            pd.DataFrame: DataFrame enriched with engineered features.
        """

        cls.logger.info("Starting feature engineering process")
        df = df.copy()

        # Validate datetime index type
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise TypeError("DataFrame index must be datetime type.")

        # Validate required columns exist
        required_cols = ['aep_mw', 'datetime_year', 'datetime_month', 'datetime_day', 'datetime_hour', 'datetime_dayofweek']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns for feature engineering: {missing_cols}")

        # Apply feature engineering steps
        cls._add_holiday_flags(df)
        cls._add_lag_features(df)
        cls._add_rolling_stats(df)
        cls._add_cyclical_encoding(df)
        cls._add_interaction_features(df)
        cls._add_time_of_day_flags(df)
        cls._add_outlier_flag(df)

        if drop_na:
            cls.logger.debug("Dropping rows with NA values after lag/rolling computations")
            df.dropna(inplace=True)

        cls.logger.info("Feature engineering completed")
        return df
