import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# file handler logs to a log file
# file is created in the same dir where the main module is run
file_handler = logging.FileHandler('decomp.log')
file_handler.setLevel(logging.DEBUG)

# console handler logs to the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create formatter and set it on the handlers
file_formatter = logging.Formatter(
    fmt='%(asctime)s | %(name)s.%(levelname)s: %(message)s',
    datefmt="%Y/%m/%d %H:%M:%S",
)
console_formatter = logging.Formatter(fmt='%(message)s')

file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

# add the handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

