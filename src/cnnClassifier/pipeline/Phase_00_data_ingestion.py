from cnnClassifier.config.configuration import ConfigManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

PHASE_NAME = "Data Ingestion Phase"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config= ConfigManager()
        data_ingestion_config= config.get_data_ingestion_config()
        data_ingestion= DataIngestion(config= data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {PHASE_NAME} started <<<<<<")
        obj= DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> {PHASE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e