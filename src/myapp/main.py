import logging
from typing import Optional
from myapp.config.config_manager import ConfigManager
from myapp.utils.logger import CustomLogger
from myapp.pipelines.stage_01_data_ingestion import DataIngestionPipeline
from myapp.pipelines.stage_02_data_validation import DataValidationPipeline


class MainPipeline:
    """
    Main Pipeline to orchestrate data ingestion and validation.
    """

    def __init__(
        self,
        config: ConfigManager,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        # Pass current module name (__name__) to CustomLogger
        self.logger = logger or CustomLogger(module_name=__name__).get_logger()

    def run(self) -> None:
        """
        Run the main pipeline consisting of data ingestion and validation.

        Raises:
            Exception if any stage fails.
        """
        try:
            self.logger.info("Starting Main Pipeline")

            # Stage 1: Data Ingestion
            ingestion_pipeline = DataIngestionPipeline(config=self.config, logger=self.logger)
            raw_data = ingestion_pipeline.run()

            # Stage 2: Data Validation
            validation_pipeline = DataValidationPipeline(config=self.config, logger=self.logger)
            validated_data = validation_pipeline.run(raw_data)

            # === Additional pipeline stages go here ===

            self.logger.info("Main Pipeline completed successfully.")

        except Exception as e:
            self.logger.error(f"Main Pipeline failed: {e}", exc_info=True)
            raise

        finally:
            self.logger.info("Main Pipeline finished.")


if __name__ == "__main__":
    config = ConfigManager().app  # or however you get your config object
    main_pipeline = MainPipeline(config=config)
    main_pipeline.run()
