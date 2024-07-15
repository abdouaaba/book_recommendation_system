import logging
from pathlib import Path

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    
    handler = logging.FileHandler(log_path / log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Create loggers
main_logger = setup_logger('main_logger', 'main.log')
api_logger = setup_logger('api_logger', 'api.log')
service_logger = setup_logger('service_logger', 'service.log')
