from myapp.config.config_loader import ConfigLoader
from typing import Optional
from pathlib import Path
from myapp.config.config_schema import (
    PathsConfig, 
    DataConfig, 
    TrainingConfig,
    HyperparameterTuningConfig, 
    LoggingConfig,
    EnvironmentConfig, 
    MetadataConfig, 
    AppConfig
)


"""Singleton Config Manager to access configuration throughout the application.
"""
class ConfigManager:
    _instance = None

    def __new__(cls, config_path: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            if config_path is None:
                config_path = Path(__file__).parents[3] / 'config' / 'config.yaml' 
            cls._instance._init(config_path)
        return cls._instance

    def _init(self, config_path):
        self.config: AppConfig = ConfigLoader(config_path).get_config()

    @property
    def paths(self) -> PathsConfig:
        return self.config.paths

    @property
    def data(self) -> DataConfig:
        return self.config.data

    @property
    def training(self) -> TrainingConfig:
        return self.config.training

    @property
    def hyperparameter_tuning(self) -> HyperparameterTuningConfig:
        return self.config.hyperparameter_tuning

    @property
    def logging(self) -> LoggingConfig:
        return self.config.logging

    @property
    def environment(self) -> EnvironmentConfig:
        return self.config.environment

    @property
    def metadata(self) -> MetadataConfig:
        return self.config.metadata

    @property
    def appconfig(self) -> AppConfig:
        return self.config
