import logging
import os
from datetime import datetime

# Global variable to store the shared log file path for the current run
_LOG_FILE_PATH = None

def setup_logger(name="pipeline", log_dir="logs", level=logging.INFO):
    """
    Sets up a logger. The first time this is called, it creates a new log file.
    Subsequent calls will attach to the SAME log file.
    
    Args:
        name (str): Name of the logger.
        log_dir (str): Directory (only used for the first call).
        level (int): Logging level.
        
    Returns:
        logging.Logger: Configured logger.
    """
    global _LOG_FILE_PATH
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # If handlers exist, return immediately to avoid duplicates
    if logger.handlers:
        return logger

    # --- 1. SETUP SHARED LOG FILE (Once per run) ---
    if _LOG_FILE_PATH is None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Create unique filename for THIS execution
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure we don't accidentally use a sub-module name for the file if initialized late
        filename_prefix = "pipeline" 
        _LOG_FILE_PATH = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.log")

    # --- 2. ADD FILE HANDLER (To shared file) ---
    file_handler = logging.FileHandler(_LOG_FILE_PATH)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # --- 3. ADD CONSOLE HANDLER ---
    # Only add console handler if not already present (avoid dupes if simple setup)
    # Check if any existing handler is a StreamHandler
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    
    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger
