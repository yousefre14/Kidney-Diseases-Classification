from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.constants import *
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig)
from pathlib import Path
from cnnClassifier import logger


class ConfigManager:
    def __init__(self, 
                 config_filename: str = "configs/config.yaml", 
                 params_filename: str = "params.yaml"):
        
        # Force project root explicitly
        self.project_root = Path("/home/yousef/Kidney-Diseases-Classification")

        # Build absolute paths
        self.config_path = self.project_root / config_filename
        self.params_path = self.project_root / params_filename

        print(f"📂 Looking for config at: {self.config_path}")
        print(f"📂 Looking for params at: {self.params_path}")

        # Validate existence
        if not self.config_path.exists():
            raise FileNotFoundError(f"❌ Config file not found: {self.config_path}")
        if not self.params_path.exists():
            raise FileNotFoundError(f"❌ Params file not found: {self.params_path}")

        # Load YAMLs
        self.config = read_yaml(str(self.config_path))
        self.params = read_yaml(str(self.params_path))

        # Create artifacts directory
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
                root_dir=Path(config.root_dir),
                base_model_path= Path(config.base_model_path),
                updated_base_model_path= Path(config.updated_base_model_path),
                params_image_size=self.params.IMAGE_SIZE,
                params_learning_rate=self.params.LEARNING_RATE,
                params_include_top=self.params.INCLUDE_TOP,
                params_weights=self.params.WEIGHTS,
                params_classes=self.params.CLASSES,
        )
        return prepare_base_model_config