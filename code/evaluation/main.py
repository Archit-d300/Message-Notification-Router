#!/usr/bin/env python3
"""
evaluation/main.py
--------------------
Participant-side sanity check. The organizer's real ground truth is hidden,
but dataset/sample_messages.csv gives 71 already-solved examples in the
exact output schema. This script re-runs the pipeline against those sample
rows (which are drawn from the same distribution as messages.csv, just with
labels attached) and reports how well the router's action/message_type
predictions line up, so you can sanity-check the system before submitting.

This is NOT the official grader -- it exists purely so a participant can
catch obviously broken behavior (e.g. everything routed to "mute") before
submission. It never touches dataset/messages.csv's hidden labels because
there are none in this repo; it only checks against sample_messages.csv.

Run with:
    cd code
    python evaluation/main.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

import config  # noqa: E402
from decision_engine import decide  # noqa: E402
from media_pipeline import analyze  # noqa: E402
from models import Message  # noqa: E402
from data_loader import _to_int, load_all  # noqa: E402


def _load_sample_rows() -> list[dict]:
    with config.SAMPLE_MESSAGES_CSV.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    store = load_all()
    rows = _load_sample_rows()
    if not rows:
        print(f"No rows found in {config.SAMPLE_MESSAGES_CSV}")
        return

    action_correct = 0
    type_correct = 0
    both_correct = 0
    confusion = Counter()

    for row in rows:
        message = Message(
            message_id=row["message_id"],
            user_id=row.get("user_id", ""),
            conversation_type=row.get("conversation_type", ""),
            group_id=row.get("group_id", ""),
            business_id=row.get("business_id", ""),
            sender_user_id=row.get("sender_user_id", ""),
            created_at=row.get("created_at", ""),
            message_text=row.get("message_text", ""),
            media_type=row.get("media_type", ""),
            media_id=row.get("media_id", ""),
            forwarded_count=_to_int(row.get("forwarded_count")),
        )
        media = analyze(message, store)
        pred = decide(message, media, store)

        gold_action = row["action"]
        gold_type = row["message_type"]

        action_ok = pred.action == gold_action
        type_ok = pred.message_type == gold_type
        action_correct += action_ok
        type_correct += type_ok
        both_correct += action_ok and type_ok
        confusion[(gold_action, pred.action)] += 1

        if not action_ok:
            print(
                f"[MISMATCH action] {message.message_id}: gold={gold_action} pred={pred.action} "
                f"({pred.reason})"
            )

    n = len(rows)
    print("\n--- Summary over dataset/sample_messages.csv ---")
    print(f"rows evaluated:        {n}")
    print(f"action accuracy:       {action_correct / n:.1%}")
    print(f"message_type accuracy: {type_correct / n:.1%}")
    print(f"both correct:          {both_correct / n:.1%}")
    print("\nconfusion (gold_action -> predicted_action):")
    for (gold, pred), count in sorted(confusion.items()):
        print(f"  {gold:8s} -> {pred:8s}: {count}")


if __name__ == "__main__":
    main()
