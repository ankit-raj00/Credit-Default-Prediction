import pandas as pd
import yaml
import os
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")

def load_config(config_path="config/config.yaml"):
    """Loads configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(raw_path):
    """
    Loads raw data from a CSV file.
    
    Args:
        raw_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    logger.info(f"Loading data from {raw_path}...")
    
    if not os.path.exists(raw_path):
         logger.error(f"File not found: {raw_path}")
         raise FileNotFoundError(f"File not found: {raw_path}")

    try:
        df = pd.read_csv(raw_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e
