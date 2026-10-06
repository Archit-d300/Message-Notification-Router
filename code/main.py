#!/usr/bin/env python3
"""
main.py
--------
Entry point. Run with:

    cd code
    python main.py

or, from the repo root:

    python code/main.py

Execution flow (see README for the full diagram):

1. logging_utils.setup_logging()          -- configure logging once
2. data_loader.load_all()                 -- read every dataset/*.csv into a DataStore
3. for each message in messages.csv:
       media_pipeline.analyze(message)    -- OCR / voice-note feature extraction
       decision_engine.decide(...)        -- risk + personalization + content fusion
4. output_writer.validate() + .write()    -- one row per message_id, correct schema
5. print a short run summary

This file intentionally contains almost no logic of its own -- it is a thin
orchestrator so every real decision lives in a single-responsibility module
that can be tested and reasoned about independently.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import data_loader  # noqa: E402
import media_pipeline  # noqa: E402
import output_writer  # noqa: E402
from decision_engine import decide  # noqa: E402
from logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("orchestrate.main")


def run(output_path: Path = config.OUTPUT_CSV, verbose: bool = False) -> list:
    setup_logging(verbose=verbose)
    start = time.time()

    logger.info("Loading dataset from %s", config.DATASET_DIR)
    store = data_loader.load_all()

    predictions = []
    action_counts = {"notify": 0, "digest": 0, "mute": 0}

    for message in store.messages_to_route:
        try:
            media_analysis = media_pipeline.analyze(message, store)
            prediction = decide(message, media_analysis, store)
        except Exception:
            logger.exception("Failed to route message_id=%s; falling back to safe digest", message.message_id)
            from models import Prediction

            prediction = Prediction(
                message_id=message.message_id,
                action="digest",
                message_type="unknown",
                reason="Fell back to a conservative digest decision after an internal processing error.",
                confidence=0.5,
                evidence_message_ids="none",
            )
        predictions.append(prediction)
        action_counts[prediction.action] += 1

    output_writer.validate(predictions, store.messages_to_route)
    output_writer.write(predictions, output_path)

    elapsed = time.time() - start
    logger.info(
        "Done in %.2fs -- notify=%d digest=%d mute=%d (total=%d)",
        elapsed, action_counts["notify"], action_counts["digest"], action_counts["mute"], len(predictions),
    )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate — Message Notification Router")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_CSV, help="Path to write output.csv")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    run(output_path=args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
