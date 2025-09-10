
# Ensure src/ is on sys.path
import os
import sys
import zipfile
import gdown
sys.path.append("/home/yousef/Kidney-Diseases-Classification/src")

# Debug: confirm Python can see cnnClassifier
import importlib.util, os
print("cnnClassifier path:", importlib.util.find_spec("cnnClassifier").origin)

# Now imports
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier import logger

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> str:
        '''
        fetch data from the url
        '''

        try:
            dataset_url = self.config.source_URL
            zip_download_path = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading file from :[{dataset_url}] into :[{zip_download_path}]")

            file_id = dataset_url.split('/')[-2]
            prefix = dataset_url.split('/')[-1].split('?')[0]
            gdown.download(id=file_id, output=str(zip_download_path), quiet=False)

            logger.info(f"File :[{zip_download_path}] has been downloaded successfully.")

        except Exception as e:
            raise e
        
    def extract_zip_file(self) -> None:
        '''
        unzip the downloaded file
        '''
        try:
            unzip_path = self.config.unzip_dir
            zip_file_path = self.config.local_data_file

            logger.info(f"Unzipping file :[{zip_file_path}] into dir :[{unzip_path}]")
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Unzipping completed successfully. and data is available at :[{unzip_path}]")
        except Exception as e:
            raise e    
        
