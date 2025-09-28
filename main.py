from cnnClassifier import logger
from cnnClassifier.pipeline.Phase_00_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.Phase_01_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.Phase_02_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.Phase_03_mlflow_evalution import MLflowEvaluationPipeline

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


Phase_NAME = "Model Training Phase"
try:
    logger.info(f">>>>>> {Phase_NAME} started <<<<<<")
    obj= ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {Phase_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

Phase_NAME = "MLflow Model Evaluation Phase"
try:
    logger.info(f">>>>>> {Phase_NAME} started <<<<<<")
    obj= MLflowEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> {Phase_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e