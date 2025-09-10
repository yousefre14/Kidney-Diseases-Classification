from cnnClassifier.config.configuration import ConfigManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

PHASE_NAME = "Prepare Base Model Phase"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config= ConfigManager()
        prepare_base_model_config= config.get_prepare_base_model_config()
        prepare_base_model= PrepareBaseModel(config= prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {PHASE_NAME} started <<<<<<")
        obj= PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> {PHASE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e