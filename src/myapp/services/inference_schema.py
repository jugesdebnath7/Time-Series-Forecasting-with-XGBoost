from typing import Optional
import os
from myapp.config.config_manager import ConfigManager
from myapp.utils.filepaths import resolve_project_path


class FilePathInputSchema:
    """Schema for inference input via static file path."""

    def __init__(self, file_path: Optional[str] = None):
        config = ConfigManager().appconfig

        if file_path is not None:
            selected_path = file_path
        else:
            # Assuming config.paths.data is a base folder path (string)
            selected_path = os.path.join(config.paths.data, "processed", "AEP_hourly.csv")  

        self.file_path: str = resolve_project_path(selected_path)

    def to_dict(self) -> dict:
        return {"file_path": self.file_path}
