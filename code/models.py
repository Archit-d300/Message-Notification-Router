"""
models.py
---------
Plain dataclasses for every entity in the dataset, plus the two output
records the pipeline produces (Signals and Prediction).

Why dataclasses instead of raw dicts: every downstream module (risk_detector,
personalization, decision_engine) reads dozens of named fields. Dicts would
mean silent KeyErrors and no editor/type support. Dataclasses give us
autocomplete, `mypy`-friendly type hints, and a single place (`data_loader.py`)
that is responsible for turning messy CSV strings into correctly-typed
values (ints, floats, bools, datetimes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class User:
    user_id: str
    do_not_disturb_window: str
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int


@dataclass
class Group:
    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    created_at: str
    messages_30d: int


@dataclass
class GroupMembership:
    group_id: str
    user_id: str
    role: str
    joined_at: str
    messages_sent_30d: int
    messages_read_30d: int
    replies_sent_30d: int
    notifications_dismissed_30d: int
    group_muted_by_user: bool


@dataclass
class BusinessAccount:
    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool
    official_domain: str
    domain_used_by_sender: str
    account_age_days: int
    messages_sent_30d: int
    user_reports_30d: int
    domain_used_by_sender_age_days: int


@dataclass
class UserBusinessHistory:
    user_id: str
    business_id: str
    why_user_knows_account: str
    last_activity_at: str
    allows_promotions: bool
    promotions_opted_out_at: str
    activity_count_180d: int
    messages_opened_30d: int
    messages_dismissed_30d: int
    messages_replied_30d: int
    last_reply_at: str


@dataclass
class MessageEvent:
    user_id: str
    message_id: str
    message_opened: bool
    message_replied: bool
    reaction_time_minutes: Optional[float]
    notification_dismissed: bool
    muted_after_message: bool
    message_reported: bool


@dataclass
class DailyNotificationSummary:
    user_id: str
    date: str
    notifications_sent: int
    notifications_dismissed: int


@dataclass
class MediaAsset:
    media_id: str
    file_path: str


@dataclass
class Message:
    """A row from messages.csv (to route) or message_history.csv (context)."""

    message_id: str
    user_id: str
    conversation_type: str  # personal | group | business
    group_id: str
    business_id: str
    sender_user_id: str
    created_at: str
    message_text: str
    media_type: str  # "" | image | voice
    media_id: str
    forwarded_count: int

    @property
    def created_at_dt(self) -> Optional[datetime]:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(self.created_at, fmt)
            except ValueError:
                continue
        return None

    @property
    def has_media(self) -> bool:
        return bool(self.media_type)


@dataclass
class MediaAnalysis:
    """Unified output of ocr_processor.py / audio_processor.py."""

    source: str  # "image" | "voice" | "none"
    extracted_text: str = ""          # OCR text or (if ever available) transcript
    summary: str = ""
    category_hint: str = ""           # e.g. "poster", "payment_screenshot"
    urgency_hint: float = 0.0         # 0..1
    scam_indicator_score: float = 0.0  # 0..1
    entities: list[str] = field(default_factory=list)
    deadline_mentions: list[str] = field(default_factory=list)
    notes: str = ""                   # honest disclosure of what could not be analyzed


@dataclass
class RiskAssessment:
    risk_score: float               # 0..1, higher = more dangerous
    risk_type: Optional[str]        # scam | spam | None
    indicators: list[str] = field(default_factory=list)


@dataclass
class RelationshipSignals:
    """Personalization features computed for one (message, user) pair."""

    relationship_strength: float = 0.0     # 0..1 sender/group/business closeness
    engagement_rate: float = 0.0           # historical open/reply rate with this source
    is_muted_source: bool = False          # group muted or business opted out
    is_direct_mention: bool = False
    is_overloaded_day: bool = False
    trust_level: float = 0.0               # 0..1 verified/known-good boost
    notes: list[str] = field(default_factory=list)


@dataclass
class EvidenceMatch:
    message_id: str
    score: float
    reason: str


@dataclass
class Prediction:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str  # already semicolon-joined, or "none"
