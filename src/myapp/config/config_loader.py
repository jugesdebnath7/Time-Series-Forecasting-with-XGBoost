# === Config YAML File Reader ===
import yaml
from pathlib import Path
from typing import Optional
from pydantic import ValidationError
from myapp.config.config_schema import AppConfig


# === Custom Exceptions ===
class ConfigLoadError(Exception):
    """Raised when the configuration file cannot be loaded."""
    pass


class ConfigValidationError(Exception):
    """Raised when the configuration is invalid based on schema."""
    pass


# === Config Loader ===
class ConfigLoader:
    """
    Loads and validates the application configuration from a YAML file
    into a structured AppConfig Pydantic model.
    """

    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = (
            Path(config_file)
            if config_file
            else Path(__file__).parents[3] / 'config' / 'config.yaml'
    )

        self.config: AppConfig = self._load_and_validate_config()

    def _load_and_validate_config(self) -> AppConfig:
        """
        Load and validate the configuration YAML against the AppConfig schema.

        Returns:
            AppConfig: Parsed and validated config object.

        Raises:
            ConfigLoadError: If the file can't be read or parsed.
            ConfigValidationError: If the YAML is invalid per AppConfig schema.
        """
        if not self.config_file.exists():
            print(f"Config file not found: {self.config_file}")
            raise ConfigLoadError(f"Config file not found: {self.config_file}")

        try:
            with open(self.config_file, 'r') as file:
                raw_config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print("Failed to parse YAML")
            raise ConfigLoadError(f"YAML parsing error: {e}") from e

        try:
            validated_config = AppConfig(**raw_config)
            print("Configuration schema validation successful.")
            return validated_config
        except ValidationError as e:
            print("Configuration schema validation error")
            print(str(e))
            raise ConfigValidationError(f"Invalid configuration: {e}") from e

    def get_config(self) -> AppConfig:
        """
        Get the validated configuration.

        Returns:
            AppConfig: Validated Pydantic config object.
        """
        return self.config
