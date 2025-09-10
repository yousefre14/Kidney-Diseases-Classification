from cnnClassifier import logger
from cnnClassifier.pipeline.Phase_00_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.Phase_01_prepare_base_model import PrepareBaseModelTrainingPipeline

PHASE_NAME = "Data Ingestion Phase"
try:
    logger.info(f">>>>>> {PHASE_NAME} started <<<<<<")
    obj= DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {PHASE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



PHASE_NAME = "Prepare Base Model Phase"
try:
    logger.info(f">>>>>> {PHASE_NAME} started <<<<<<")
    obj= PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {PHASE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e