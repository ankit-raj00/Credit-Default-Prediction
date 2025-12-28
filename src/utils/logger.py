import logging
import os
from datetime import datetime

def setup_logger(name="pipeline", log_dir="logs", level=logging.INFO):
    """
    Sets up a logger that writes to both a timestamped file and the console.
    
    Args:
        name (str): Name of the logger and prefix for the log file.
        log_dir (str): Directory to store log files.
        level (int): Logging level (default: logging.INFO).
        
    Returns:
        logging.Logger: Configured logger.
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # Create or get the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding multiple handlers if logger is already configured
    if logger.handlers:
        return logger
        
    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
