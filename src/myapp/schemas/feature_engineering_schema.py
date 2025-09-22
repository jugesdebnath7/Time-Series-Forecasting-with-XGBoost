# myapp/schemas/feature_engineering_schema.py

import pandas as pd
import numpy as np
import holidays
from myapp.utils.logger import CustomLogger


class FeatureEngineeringSchema:
    """
    Schema for applying engineered features to time-series energy data.
    """
    logger = CustomLogger(module_name=__name__).get_logger()

    @classmethod
    def _add_holiday_flags(
        cls, 
        df: pd.DataFrame
    ) -> None:
        cls.logger.debug("Adding holiday and weekend flags")

        # Weekend: Saturday (5) or Sunday (6)
        df['is_weekend'] = df['datetime_dayofweek'].isin([5, 6]).astype(int)

        # Holiday detection
        us_holidays = holidays.US(years=df['datetime_year'].unique())
        df['is_holiday'] = df['datetime'].dt.normalize().isin(us_holidays).astype(int)

        # New Year's Eve flag
        df['is_new_year_eve'] = (
            (df['datetime_month'] == 12) & (df['datetime_day'] == 31)
        ).astype(int)

    @classmethod
    def _add_lag_features(
        cls, 
        df: pd.DataFrame
    ) -> None:
        cls.logger.debug("Adding lag features")
        df['lag_24'] = df['aep_mw'].shift(24)
        # Optional: Add more lag features if needed
        # df['lag_168'] = df['aep_mw'].shift(168)

    @classmethod
    def _add_rolling_stats(
        cls, 
        df: pd.DataFrame
    ) -> None:
        cls.logger.debug("Adding rolling mean and std")
        df['rolling_mean_24'] = df['aep_mw'].shift(1).rolling(window=24).mean()
        df['rolling_std_24'] = df['aep_mw'].shift(1).rolling(window=24).std()

    @classmethod
    def _add_cyclical_encoding(
        cls, 
        df: pd.DataFrame
    ) -> None:
        cls.logger.debug("Adding cyclical encodings for hour, dayofweek, month")
        df['hour_sin'] = np.sin(2 * np.pi * df['datetime_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['datetime_hour'] / 24)

        df['month_cos'] = np.cos(2 * np.pi * df['datetime_month'] / 12)

        df['dayofweek_cos'] = np.cos(2 * np.pi * df['datetime_dayofweek'] / 7)

    @classmethod
    def _add_interaction_features(
        cls, 
        df: pd.DataFrame
    ) -> None:
        cls.logger.debug("Adding interaction features")
        df['hour_is_holiday'] = df['datetime_hour'] * df['is_holiday']
        df['hour_is_weekend'] = df['datetime_hour'] * df['is_weekend']

    @classmethod
    def _add_time_of_day_flags(
        cls, 
        df: pd.DataFrame
    ) -> None:
        cls.logger.debug("Adding time of day flags")
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
        cls.logger.debug("Adding outlier flag based on IQR")
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
        Main method to apply all feature engineering steps.

        Parameters:
            df (pd.DataFrame): Time-series dataframe with datetime features and 'aep_mw' column.
            drop_na (bool): Whether to drop rows with NA from lag/rolling ops.

        Returns:
            pd.DataFrame: DataFrame with additional feature-engineered columns.
        """
        cls.logger.info("Starting feature engineering process")
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Ensure datetime column exists and is datetime type
        if 'datetime' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            raise ValueError("DataFrame must include a 'datetime' column of datetime type.")

        # Run feature engineering steps
        cls._add_holiday_flags(df)
        cls._add_lag_features(df)
        cls._add_rolling_stats(df)
        cls._add_cyclical_encoding(df)
        cls._add_interaction_features(df)
        cls._add_time_of_day_flags(df)
        cls._add_outlier_flag(df)

        if drop_na:
            cls.logger.debug("Dropping rows with NA values from lag/rolling features")
            df.dropna(inplace=True)

        cls.logger.info("Feature engineering completed")
        return df
