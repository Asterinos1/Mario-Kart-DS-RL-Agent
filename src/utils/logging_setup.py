import logging
import os
import sys

def setup_logging(log_file=None, level=logging.INFO):
    """Configures the logging system for training or evaluation.
    
    Sets up a root logger that prints to the console and optionally writes to a file.
    
    Args:
        log_file (str, optional): Path to a log file where logs should be appended.
        level (int): The logging level to use (e.g. logging.INFO, logging.DEBUG).
    """
    # Get root logger
    root_logger = logging.getLogger()
    
    # Remove any existing handlers to avoid duplicate logs when setup is called multiple times
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        
    root_logger.setLevel(level)
    
    # Define log format
    log_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(log_format)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
        
    # Suppress noise from third-party libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("desmume").setLevel(logging.WARNING)
    logging.getLogger("stable_baselines3").setLevel(logging.INFO)
    
    return root_logger
