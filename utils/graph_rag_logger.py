import logging
import logging.config
import yaml
from pathlib import Path
import os
from dotenv import  load_dotenv

load_dotenv()

def setup_logging(config_path: str = "./log_config/logging_config.yaml", default_level=logging.INFO):
    """
    Setup logging configuration from YAML file.
    Auto-creates log directories if missing.
    """
    print(f'Loading logger path: {config_path}. Present working directory: {os.getcwd()}')
    config_file = Path(config_path)

    if config_file.exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f.read())

        # Ensure log directory exists
        handlers = config["logging"].get("handlers", {})
        for handler in handlers.values():
            filename = handler.get("filename")
            if filename:
                Path(filename).parent.mkdir(parents=True, exist_ok=True)

        logging.config.dictConfig(config["logging"])
    else:
        logging.basicConfig(level=default_level)
        logging.warning("Logging config file not found. Using basic config.")

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger.
    """
    if os.getenv('DISABLE_THIRD_PARTY_DEBUG_LOGS'):
        for p in os.getenv('DISABLE_THIRD_PARTY_DEBUG_LOGS').split(","):
            #print(f'disabling..{p}')
            logging.getLogger(p.strip()).setLevel(logging.WARNING)
    return logging.getLogger(name)
