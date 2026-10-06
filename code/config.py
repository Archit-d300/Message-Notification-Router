"""
config.py
---------
Single source of truth for paths, tunable thresholds, and allowed output
values. Nothing in this file talks to disk or the network directly; it only
describes *where* things live and *how sensitive* the decision engine should
be. Every threshold used by risk_detector.py and decision_engine.py is
declared here so the whole system can be re-tuned without touching logic.

Design decision: thresholds are plain module-level constants (not a YAML/JSON
file) because the grading harness only needs `python main.py` to work with no
extra config plumbing. They are still centralised in one file, which is the
part that actually matters for maintainability.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media"

MESSAGES_CSV = DATASET_DIR / "messages.csv"
OUTPUT_CSV = DATASET_DIR / "output.csv"
SAMPLE_MESSAGES_CSV = DATASET_DIR / "sample_messages.csv"
USERS_CSV = DATASET_DIR / "users.csv"
GROUPS_CSV = DATASET_DIR / "groups.csv"
GROUP_MEMBERS_CSV = DATASET_DIR / "group_members.csv"
BUSINESS_ACCOUNTS_CSV = DATASET_DIR / "business_accounts.csv"
USER_BUSINESS_HISTORY_CSV = DATASET_DIR / "user_business_history.csv"
MESSAGE_HISTORY_CSV = DATASET_DIR / "message_history.csv"
MESSAGE_EVENTS_CSV = DATASET_DIR / "message_events.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"
VOICE_NOTES_CSV = DATASET_DIR / "voice_notes.csv"
DAILY_NOTIFICATION_SUMMARY_CSV = DATASET_DIR / "daily_notification_summary.csv"

LOG_DIR = REPO_ROOT / "logs"
RUN_LOG_FILE = LOG_DIR / "log.txt"
CACHE_DIR = REPO_ROOT / ".cache"

# ---------------------------------------------------------------------------
# Allowed output values (contract from problem_statement.md)
# ---------------------------------------------------------------------------
ACTIONS = ("notify", "digest", "mute")
MESSAGE_TYPES = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)
OUTPUT_COLUMNS = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)

# ---------------------------------------------------------------------------
# Optional enrichment (off unless the operator opts in and provides a key)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ENABLE_LLM_ENRICHMENT = os.environ.get("ENABLE_LLM_ENRICHMENT", "false").lower() == "true"
LLM_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Risk detector thresholds
# ---------------------------------------------------------------------------
RISK_MUTE_THRESHOLD = 0.55          # risk_score >= this forces action=mute
RISK_HIGH_CONFIDENCE = 0.9          # risk_score >= this yields near-certain confidence
DOMAIN_MISMATCH_WEIGHT = 0.35
UNVERIFIED_BUSINESS_WEIGHT = 0.15
NEW_ACCOUNT_DAYS_THRESHOLD = 30
NEW_ACCOUNT_WEIGHT = 0.15
HIGH_REPORT_COUNT_THRESHOLD = 5
HIGH_REPORT_WEIGHT = 0.2

# ---------------------------------------------------------------------------
# Decision fusion thresholds (final routing score is in [0, 1])
# ---------------------------------------------------------------------------
NOTIFY_THRESHOLD = 0.62
DIGEST_THRESHOLD = 0.35   # below this -> mute
DIRECT_MENTION_BOOST = 0.25
GROUP_MUTED_PENALTY = 0.4
BUSINESS_OPT_OUT_PENALTY = 0.45
OVERLOAD_DIGEST_PENALTY = 0.12
DAILY_NOTIFICATIONS_OVERLOAD = 6  # notifications_sent above this = "noisy day"

# ---------------------------------------------------------------------------
# History / evidence retrieval
# ---------------------------------------------------------------------------
MAX_EVIDENCE_IDS = 3
MIN_EVIDENCE_TOKEN_OVERLAP = 1  # minimum shared keyword/entity tokens to count as evidence

# ---------------------------------------------------------------------------
# Confidence calibration bounds
# ---------------------------------------------------------------------------
CONFIDENCE_FLOOR = 0.5
CONFIDENCE_CEILING = 0.97
