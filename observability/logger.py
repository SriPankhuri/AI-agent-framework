import logging
import sys

# Define a custom format for agent activities
LOG_FORMAT = "%(asctime)s | %(levelname)s | Agent: %(message)s"

def get_agent_logger(name="AI_Agent"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if the logger is called multiple times
    if not logger.handlers:
        # Standard output handler (Console)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Singleton instance for the framework
agent_logger = get_agent_logger()
