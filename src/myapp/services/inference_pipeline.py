from typing import Optional
import pandas as pd

from myapp.config.config_schema import AppConfig
from myapp.utils.logger import CustomLogger
from myapp.services.model_loader import ModelLoader
from myapp.pipelines.main_pipeline import MainPipeline
from myapp.services.inference_schema import FilePathInputSchema


class InferencePipeline:
    def __init__(
        self,
        config: AppConfig,
        logger: CustomLogger,
        file_path: Optional[str] = None
    ) -> None:
        self.config = config
        self.logger = logger

        # Resolve file path using input schema
        self.input_schema = FilePathInputSchema(file_path)
        self.file_path = self.input_schema.file_path  # resolved absolute path
        self.logger.info(f"Using data file path: {self.file_path}")

        # Load the model
        self.model = ModelLoader(config=self.config).load_model()

        # Initialize main pipeline with file path (if supported)
        self.main_pipeline = MainPipeline(
            config=self.config,
            logger=self.logger,
            file_path=self.file_path  # Make sure MainPipeline accepts this!
        )

    def run(self) -> pd.DataFrame:
        self.logger.info("Starting Inference Pipeline")
        all_predictions = []

        try:
            for chunk_idx, chunk in enumerate(self.main_pipeline.run()):
                self.logger.info(f"Processing chunk {chunk_idx + 1}")

                model_features = list(self.model.feature_names_in_)
                missing_features = set(model_features) - set(chunk.columns)

                if missing_features:
                    raise ValueError(f"Missing features: {missing_features}")

                chunk = chunk[model_features]
                predictions = self.model.predict(chunk)

                chunk_result = pd.DataFrame({
                    "prediction": predictions
                }, index=chunk.index)

                all_predictions.append(chunk_result)

            return pd.concat(all_predictions).reset_index()

        except Exception as e:
            self.logger.error(f"Inference failed: {e}", exc_info=True)
            raise

        finally:
            self.logger.info("Inference finished.")
