# Context Library
from contextlib import contextmanager

# Logging
import logging

# Context Manager
@contextmanager

# Disable all logging
def all_logging_disabled(highest_level=logging.CRITICAL):

    # Previous level
    previous_level = logging.root.manager.disable

    # Disable logging of the highest level
    logging.disable(highest_level)

    # Continue
    try:
        yield

    # Execute regardless
    finally:
        logging.disable(previous_level)
