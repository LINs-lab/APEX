import os
import sys
import logging

import ast
import yaml
import argparse

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def create_logger(name, logging_dir="", use_color=True):
  logger = logging.getLogger(name)

  if not logger.handlers:
    logger.setLevel(logging.INFO)

    if logging_dir != "":
      log_file = os.path.join(logging_dir, "log.txt")
      # Create file handler
      file_handler = logging.FileHandler(log_file)
      file_handler.setLevel(logging.INFO)
      file_formatter = logging.Formatter(
          fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
      )
      file_handler.setFormatter(file_formatter)
      logger.addHandler(file_handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    if use_color:
        console_formatter = ColoredFormatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
        )
    else:
        console_formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
        )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Prevent log messages from propagating to the root logger
    logger.propagate = False

  return logger


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds ANSI colors to log levels.
    """

    COLOR_MAP = {
        "DEBUG": "\033[37m",  # White
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Purple
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def parse_list(option_str):
    try:
        option_str = option_str.strip()
        if option_str.startswith("[") and option_str.endswith("]"):
            elements = option_str[1:-1].split(",")
            processed_elements = []
            for element in elements:
                element = element.strip()
                if element.replace(".", "", 1).isdigit():
                    processed_elements.append(element)
                else:
                    processed_elements.append(f'"{element}"')
            option_str = "[" + ", ".join(processed_elements) + "]"

        result = ast.literal_eval(option_str)
        if not isinstance(result, list):
            raise ValueError
        return result
    except:
        raise argparse.ArgumentTypeError("Invalid list format")
