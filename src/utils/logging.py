import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger that logs to the console.
    This avoids complex file-based logging and dependencies on settings.
    """
    logger = logging.getLogger(name)
    
    # Set level for this logger. INFO is a good default.
    logger.setLevel(logging.INFO)
    
    # Prevent adding handlers multiple times, which would duplicate logs.
    if not logger.handlers:
        # Create a handler that writes to standard output (the console).
        handler = logging.StreamHandler(sys.stdout)
        
        # Create a formatter to define the log message format.
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add the configured handler to the logger.
        logger.addHandler(handler)

        # Adiciona um handler para salvar os logs em um arquivo.
        file_handler = logging.FileHandler("app.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    # Propagate messages to the root logger if needed, but for this setup,
    # direct handling is sufficient.
    logger.propagate = False

    return logger
