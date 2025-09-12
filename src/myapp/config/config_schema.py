# === config_schema module ===

from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Configuration for file paths."""
    data: str
    raw: str
    processed: str
    output: str
    models: str
    logs: str
    

class DataConfig(BaseModel):
    """Configuration for data processing."""
    file_type: str
    lazy: bool
    chunk_size: int
    split_ratio: float
    shuffle: bool
    cv: int
    max_rows: int


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    random_seed: int
    early_stopping: int
    eval_metric: str
    n_jobs: int
    n_estimators: int
    verbosity: int
    tree_method: str


class HyperparameterSearchSpace(BaseModel):
    """Configuration for hyperparameter search space."""
    gamma: List[float]
    reg_alpha: List[float]
    reg_lambda: List[float]
    learning_rate: List[float]
    max_depth: List[int]
    min_child_weight: List[int]
    subsample: List[float]
    colsample_bytree: List[float]


class HyperparameterTuningConfig(BaseModel):
    """Configuration for hyperparameter tuning."""
    tuning_enabled: bool
    strategy: str
    n_iter: Optional[int]
    search_space: Optional[HyperparameterSearchSpace]
    

class FileHandlerConfig(BaseModel):
    """Configuration for file handling."""
    level: str
    filename: str
    maxBytes: int
    backupCount: int


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    app_name: str
    log_to_console: bool
    enable_mlflow: bool
    mlflow_tracking_uri: str
    log_feature_importance: bool
    save_metrics: bool
    level: str
    handlers: Dict[str, FileHandlerConfig]


class EnvironmentConfig(BaseModel):
    """Configuration for the environment."""
    mode: str
    conda_env: str
    python_version: str
    dependencies: List[str]


class MetadataConfig(BaseModel):
    """Configuration for metadata."""
    description: str
    pipeline_version: str
    model_type: str
    

class AppConfig(BaseModel):
    """Configuration for the application."""
    paths: PathsConfig
    data: DataConfig
    training: TrainingConfig
    hyperparameter_tuning: HyperparameterTuningConfig
    logging: LoggingConfig
    environment: EnvironmentConfig
    metadata: MetadataConfig
