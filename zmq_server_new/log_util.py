import logging


def set_logger(context):
    logger = logging.getLogger(context)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s: ' + '%(levelname)-.1s:' + context + ':[%(filename)s:%(funcName)s:%(lineno)3d]:%(message)s', datefmt=
        '%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger
