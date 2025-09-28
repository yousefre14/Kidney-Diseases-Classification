import os
import tensorflow as tf
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.keras
from urllib.parse import urlparse
import json
from pathlib import Path
import tempfile  # <-- needed
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json

def save_json(data, filename):
    """Save dictionary data to a JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


class Evaluation:
    def __init__(self, config: "EvaluationConfig"):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.3
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self, filename="scores.json"):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(scores, filename)

    def log_into_mlflow(self, experiment_name="EvaluationExperiment"):
        # ✅ Set DagsHub tracking URI with token
        os.environ["MLFLOW_TRACKING_USERNAME"] = "yousefre14"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "831e4988b7ed7c07fef99e112034680d2f64c41b"
        mlflow.set_tracking_uri(
            "https://831e4988b7ed7c07fef99e112034680d2f64c41b@dagshub.com/yousefre14/Kidney-Diseases-Classification.mlflow"
        )
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # ✅ Create experiment if not exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)

        # ✅ Start a run
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            # Log params and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            # ✅ Log the model as artifacts (safe for DagsHub)
            with tempfile.TemporaryDirectory() as tmpdir:
                local_model_path = f"{tmpdir}/model"
                mlflow.keras.save_model(self.model, local_model_path)
                mlflow.log_artifacts(local_model_path, artifact_path="model")

        print("✅ Metrics and model logged to DagsHub successfully!")