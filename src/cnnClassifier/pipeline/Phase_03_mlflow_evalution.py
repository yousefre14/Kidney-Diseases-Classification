from cnnClassifier.config.configuration import ConfigManager
from cnnClassifier.components.model_evaluation import Evaluation
from cnnClassifier import logger


Stage_NAME = "MLflow Model Evaluation Stage"

class MLflowEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(config=eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>> Stage {Stage_NAME} started <<<<<<<<<<")
        obj = MLflowEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>>>> Stage {Stage_NAME} completed <<<<<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
