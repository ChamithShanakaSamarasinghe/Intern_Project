import logging
import sys

def setup_logger():
    """
    Named logger that always writes to stdout (works with uvicorn on Windows).
    Avoids duplicate handlers when reload is active.
    """
    name = "jw_rag"  # app logger name
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # If handlers already attached (e.g. on reload), return logger
    if logger.handlers:
        return logger

    # Stream to stdout (important)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent double output via root logger
    logger.propagate = False

    # sanity message
    logger.info("Logger initialized (jw_rag)")

    return logger