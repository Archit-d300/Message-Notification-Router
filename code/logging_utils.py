"""
logging_utils.py
------------------
One function, `setup_logging()`, called once from main.py. Every other
module just does `logging.getLogger("orchestrate.<name>")` and inherits
this configuration -- no module configures logging for itself, so output
is consistent and doesn't duplicate handlers when re-imported (e.g. under
pytest).
"""

from __future__ import annotations

import logging
import sys

import config


def setup_logging(verbose: bool = False) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger("orchestrate")
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(config.LOG_DIR / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
