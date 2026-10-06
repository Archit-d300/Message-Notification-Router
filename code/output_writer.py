"""
output_writer.py
------------------
Responsibility: serialize a list of Prediction objects to output.csv with
the exact column order/names required by problem_statement.md, and validate
that every message_id from messages.csv is present exactly once before
writing (fail loud rather than silently submit a partial file).

Called by: main.py (last step).
Calls: nothing except the standard library `csv` module.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import config
from models import Message, Prediction

logger = logging.getLogger("orchestrate.output_writer")


def validate(predictions: list[Prediction], messages: list[Message]) -> None:
    expected_ids = {m.message_id for m in messages}
    got_ids = [p.message_id for p in predictions]

    missing = expected_ids - set(got_ids)
    duplicates = {mid for mid in got_ids if got_ids.count(mid) > 1}

    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} message_id(s): {sorted(missing)[:5]}...")
    if duplicates:
        raise ValueError(f"Duplicate predictions for message_id(s): {sorted(duplicates)}")

    for p in predictions:
        if p.action not in config.ACTIONS:
            raise ValueError(f"{p.message_id}: invalid action {p.action!r}")
        if p.message_type not in config.MESSAGE_TYPES:
            raise ValueError(f"{p.message_id}: invalid message_type {p.message_type!r}")
        if not (0.0 <= p.confidence <= 1.0):
            raise ValueError(f"{p.message_id}: confidence {p.confidence} out of range")


def write(predictions: list[Prediction], path: Path = config.OUTPUT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(config.OUTPUT_COLUMNS)
        for p in predictions:
            writer.writerow(
                [p.message_id, p.action, p.message_type, p.reason, f"{p.confidence:.2f}", p.evidence_message_ids]
            )
    logger.info("Wrote %d predictions to %s", len(predictions), path)
