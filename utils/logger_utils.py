from __future__ import annotations

import logging
from datetime import datetime as dt
from pathlib import Path


def setup_logging(logger_name: str, logfile_path: str | Path) -> None:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Logfile handler
    now = dt.now()
    log_stem = now.strftime(rf"{logfile_path.stem}-%Y%m%d-%H:%M:%S")
    log_path = logfile_path.with_stem(log_stem)
    f_handler = logging.FileHandler(log_path, mode="a+")
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


if __name__ == "__main__":
    log_file_path = Path(
        "lowAltitude_classification/Augmentation_iNat_classifier/log_aug24.txt"
    )

    setup_logging("aug24", log_file_path)
