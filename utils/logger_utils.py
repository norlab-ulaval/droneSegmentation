from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(logger_name: str, logfile_path: str | Path) -> None:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Logfile handler
    f_handler = logging.FileHandler(logfile_path, mode="a+")
    f_handler.setLevel(logging.DEBUG)
    f_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] - %(filename)s: %(message)s"
    )
    f_handler.setFormatter(f_format)

    # Console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_format = logging.Formatter("[%(levelname)s]: %(message)s")
    c_handler.setFormatter(c_format)

    logger.addHandler(f_handler)
    logger.addHandler(c_handler)
