import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import configBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: path) -> ConfigBox:
    """reads yaml file and returns
        
    Args:
        path_to_yaml (str): path like input
            
    Raises:
        ValueError: if yaml file is empty 
        e: empty file
    Returns:
        ConfigBox: ConfigBox type
        """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories
    
    Args:
        path_to_directories (list): list of path of directories
        verbose (bool, optional): _description_. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: str, data: dict):
    """save dict as json file
    
    Args:
        path (str): path to json file
        data (dict): data to be saved
    """
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path: str) -> dict:
    """load json file and returns as dict
    
    Args:
        path (str): path to json file
        
    Returns:
        dict: data loaded from json file
    """
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    
    logger.info(f"json file loaded from: {path}")
    return data

@ensure_annotations
def save_bin(data: Any, path: str):
    """save binary file
    
    Args:
        data (Any): data to be saved
        path (str): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")

@ensure_annotations
def load_bin(path: str) -> Any:
    """load binary file and return the data
    
    Args:
        path (str): path to binary file
        
    Returns:
        Any: data loaded from binary file
    """
    data = joblib.load(filename=path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: str) -> str:
    """get size in KB
    
    Args:
        path (str): path to file
        
    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    logger.info(f"Size of file at {path}: {size_in_kb} KB")
    return f"~ {size_in_kb} KB"

def decodeImage(imgstring,fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
        return my_string