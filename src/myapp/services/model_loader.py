import os
import joblib
from myapp.utils.logger import CustomLogger
from myapp.utils.filepaths import resolve_project_path


class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.logger = CustomLogger(__name__).get_logger()
    
    def load_model(self):
        # Resolve absolute artifact directory path
        artifact_path = resolve_project_path("artifacts")
        
        # Ensure artifact directory exists
        os.makedirs(artifact_path, exist_ok=True)
        
        # Construct model filename and full path inside resolved artifact directory
        model_filename = f"model_{self.config.metadata.pipeline_version}.joblib"
        model_path = os.path.join(artifact_path, model_filename)

        self.logger.info(f"Trying to load model from: {model_path}")
        self.logger.info(f"Current working directory: {os.getcwd()}")
        self.logger.info(f"File exists? {os.path.isfile(model_path)}")
        self.logger.info(f"Contents of artifact directory: {os.listdir(artifact_path)}")

        try:
            model = joblib.load(model_path)
            self.logger.info("Model loaded successfully.")
            return model

        except FileNotFoundError:
            self.logger.error(f"Model file not found at: {model_path}")
            raise

        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise
