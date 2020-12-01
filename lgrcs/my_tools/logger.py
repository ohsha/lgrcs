import sys
import logging
import logging.handlers


def initialize_logger(project, log_path, debug=True):

    logger = logging.getLogger(project)
    formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')

    handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=10**6)
    handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(stdout_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)