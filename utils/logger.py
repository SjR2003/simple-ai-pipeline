import logging
import logging.config
import os
import yaml
from datetime import datetime

_LOGGER_CONFIG_PATH = os.path.join("configs", "logger.yaml")

_logger_instance = None

def setup_logger(name: str = "main") -> logging.Logger:
    """Initialize and configure a logger with timestamped log files.

    Args:
        name: Name of the logger (default: 'main')

    Returns:
        Configured logger instance
    """

    global _logger_instance
    if _logger_instance:
        return _logger_instance
    
    try:
        with open(_LOGGER_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for handler_config in config.get("handlers", {}).values():
            if "filename" in handler_config:
                filepath = handler_config["filename"]
                dirname = os.path.dirname(filepath)
                basename = os.path.basename(filepath)
                name_no_ext, ext = os.path.splitext(basename)
                new_filename = f"{name_no_ext}_{timestamp}{ext}"
                handler_config["filename"] = os.path.join(dirname, new_filename)
                os.makedirs(dirname, exist_ok=True)

        logging.config.dictConfig(config)
        logger = logging.getLogger(name)
        logger.info(f"✅   Logger configured from {_LOGGER_CONFIG_PATH}")

        _logger_instance = logger
        return logger

    except FileNotFoundError:
        return _fallback_logger("Logger config file not found", name)

    except Exception as e:
        return _fallback_logger(f"Failed to load logger config: {e}", name)


def _fallback_logger(reason: str, name: str) -> logging.Logger:
    """Create a fallback logger if config loading fails."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{name}_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(name)
    logger.warning(f"⚠️   {reason}, using basic fallback config.")
    return logger
