import logging
import pandas as pd
from myapp.config.config_manager import ConfigManager
from myapp.config.config_schema import AppConfig
from myapp.utils.logger import CustomLogger
from myapp.pipelines.stage_01_data_ingestion import DataIngestionPipeline
from myapp.pipelines.stage_02_data_validation import DataValidationPipeline
from myapp.pipelines.stage_03_data_cleaning import DataCleaningPipeline
from myapp.pipelines.stage_04_data_preprocessing import DataPreprocessingPipeline
from myapp.pipelines.stage_05_feature_engineering import FeatureEngineeringPipeline


class MainPipeline:
    """
    Main Pipeline to orchestrate data ingestion and validation.
    """
    def __init__(
        self,
        config: AppConfig,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.logger = logger 
        
    def _ingestion_pipeline(self) -> pd.DataFrame:
        ingestion_pipeline = DataIngestionPipeline(
                config=self.config, 
                logger=self.logger
        )
        raw_data = ingestion_pipeline.run()  
        return raw_data 
        
    def _cleaning_pipeline(self, raw_data) -> pd.DataFrame:    
        cleaning_pipeline = DataCleaningPipeline(
                config=self.config, 
                logger=self.logger
        )
        cleaned_data = cleaning_pipeline.run(raw_data)
        return cleaned_data
        
    def _validation_pipeline(self, cleaned_data) -> pd.DataFrame: 
        validation_pipeline = DataValidationPipeline(
                config=self.config, 
                logger=self.logger
        )
        validated_data = validation_pipeline.run(cleaned_data) 
        return validated_data   
    
    def _preprocessing_pipeline(self, validated_data) -> pd.DataFrame:    
        preprocessing_pipeline = DataPreprocessingPipeline(
                config=self.config, 
                logger=self.logger
        )
        preprocessed_data = preprocessing_pipeline.run(validated_data)
        return preprocessed_data
        
    def _featureengineering_pipeline(self, preprocessed_data) -> pd.DataFrame: 
        engineering_pipeline = FeatureEngineeringPipeline(
                config=self.config, 
                logger=self.logger
        )
        engineered_data = engineering_pipeline.run(preprocessed_data)  
        return engineered_data
          
    # === Additional pipeline stages go here ===
    
    def run(self) -> None:
        """
        Run the main pipeline consisting of data ingestion and validation.

        Raises:
            Exception if any stage fails.
        """
        try:
            self.logger.info("Starting Main Pipeline")

            # Stage 1: Data Ingestion
            raw_data = self._ingestion_pipeline() 
            
            # Stage 2: Data Cleaning
            cleaned_data = self._cleaning_pipeline(raw_data) 
        
            # Stage 3: Data Validation
            validated_data = self._validation_pipeline(cleaned_data)
                
            # Stage 4: Data Preprocessing 
            preprocessed_data = self._preprocessing_pipeline(validated_data) 
        
            # Stage 5: Features Engineering
            engineered_data = self._featureengineering_pipeline(preprocessed_data) 
             
            # === Additional pipeline stages go here ===
            
            self.logger.info("Main Pipeline completed successfully.")

        except Exception as e:
            self.logger.error(f"Main Pipeline failed: {e}", exc_info=True)
            raise

        finally:
            self.logger.info("Main Pipeline finished.")


if __name__ == "__main__":
    logger = CustomLogger(__name__).get_logger()
    config = ConfigManager().appconfig
    
    main_pipeline = MainPipeline(
        logger=logger, 
        config=config
    )
    main_pipeline.run()
