"""
data_loader.py
--------------
Responsibility: read every CSV under dataset/ exactly once, coerce string
fields into the correct types, and hand back an in-memory `DataStore` that
every other module queries by id. This is the only file that knows CSV
column names -- if a column is renamed upstream, this is the only file that
needs to change.

Called by: main.py (once, at startup).
Calls: nothing except the standard library `csv` module.

Design decision: a single frozen `DataStore` dataclass of dicts (O(1) lookup
by id) rather than pandas. The dataset is small (a few hundred rows per
file) so pandas would add a dependency and startup cost for no benefit; the
stdlib `csv` module keeps the solution runnable with zero third-party
packages for the core routing path (OCR is the only optional dependency).
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config
from models import (
    BusinessAccount,
    DailyNotificationSummary,
    Group,
    GroupMembership,
    MediaAsset,
    Message,
    MessageEvent,
    User,
    UserBusinessHistory,
)

logger = logging.getLogger("orchestrate.data_loader")


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        logger.warning("Dataset file missing, treating as empty: %s", path)
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default


def _to_float(value: str, default: Optional[float] = None) -> Optional[float]:
    try:
        v = str(value).strip()
        return float(v) if v != "" else default
    except (ValueError, TypeError):
        return default


def _to_bool(value: str) -> bool:
    return str(value).strip() in {"1", "true", "True", "yes"}


@dataclass
class DataStore:
    users: dict[str, User] = field(default_factory=dict)
    groups: dict[str, Group] = field(default_factory=dict)
    # (group_id, user_id) -> membership
    group_members: dict[tuple[str, str], GroupMembership] = field(default_factory=dict)
    businesses: dict[str, BusinessAccount] = field(default_factory=dict)
    # (user_id, business_id) -> history
    user_business_history: dict[tuple[str, str], UserBusinessHistory] = field(default_factory=dict)
    message_history: dict[str, Message] = field(default_factory=dict)
    # user_id -> [message_ids], newest-agnostic; used for fast personalized retrieval
    message_history_by_user: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    message_history_by_sender: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    message_history_by_business: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    message_history_by_group: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    # (user_id, message_id) -> event
    message_events: dict[tuple[str, str], MessageEvent] = field(default_factory=dict)
    images: dict[str, MediaAsset] = field(default_factory=dict)
    voice_notes: dict[str, MediaAsset] = field(default_factory=dict)
    daily_summary: dict[tuple[str, str], DailyNotificationSummary] = field(default_factory=dict)
    daily_summary_by_user: dict[str, list[DailyNotificationSummary]] = field(
        default_factory=lambda: defaultdict(list)
    )
    messages_to_route: list[Message] = field(default_factory=list)


def load_all() -> DataStore:
    """Read every dataset file and return a fully indexed DataStore."""
    store = DataStore()

    for row in _read_rows(config.USERS_CSV):
        u = User(
            user_id=row["user_id"],
            do_not_disturb_window=row.get("do_not_disturb_window", ""),
            messages_opened_30d=_to_int(row.get("messages_opened_30d")),
            messages_replied_30d=_to_int(row.get("messages_replied_30d")),
            notifications_dismissed_30d=_to_int(row.get("notifications_dismissed_30d")),
            messages_reported_30d=_to_int(row.get("messages_reported_30d")),
        )
        store.users[u.user_id] = u

    for row in _read_rows(config.GROUPS_CSV):
        g = Group(
            group_id=row["group_id"],
            group_name=row.get("group_name", ""),
            group_type=row.get("group_type", ""),
            member_count=_to_int(row.get("member_count")),
            admin_count=_to_int(row.get("admin_count")),
            created_at=row.get("created_at", ""),
            messages_30d=_to_int(row.get("messages_30d")),
        )
        store.groups[g.group_id] = g

    for row in _read_rows(config.GROUP_MEMBERS_CSV):
        gm = GroupMembership(
            group_id=row["group_id"],
            user_id=row["user_id"],
            role=row.get("role", ""),
            joined_at=row.get("joined_at", ""),
            messages_sent_30d=_to_int(row.get("messages_sent_30d")),
            messages_read_30d=_to_int(row.get("messages_read_30d")),
            replies_sent_30d=_to_int(row.get("replies_sent_30d")),
            notifications_dismissed_30d=_to_int(row.get("notifications_dismissed_30d")),
            group_muted_by_user=_to_bool(row.get("group_muted_by_user", "0")),
        )
        store.group_members[(gm.group_id, gm.user_id)] = gm

    for row in _read_rows(config.BUSINESS_ACCOUNTS_CSV):
        b = BusinessAccount(
            business_id=row["business_id"],
            display_name=row.get("display_name", ""),
            brand_name=row.get("brand_name", ""),
            category=row.get("category", ""),
            verified=_to_bool(row.get("verified", "0")),
            official_domain=row.get("official_domain", ""),
            domain_used_by_sender=row.get("domain_used_by_sender", ""),
            account_age_days=_to_int(row.get("account_age_days"), default=-1),
            messages_sent_30d=_to_int(row.get("messages_sent_30d")),
            user_reports_30d=_to_int(row.get("user_reports_30d")),
            domain_used_by_sender_age_days=_to_int(row.get("domain_used_by_sender_age_days"), default=-1),
        )
        store.businesses[b.business_id] = b

    for row in _read_rows(config.USER_BUSINESS_HISTORY_CSV):
        h = UserBusinessHistory(
            user_id=row["user_id"],
            business_id=row["business_id"],
            why_user_knows_account=row.get("why_user_knows_account", ""),
            last_activity_at=row.get("last_activity_at", ""),
            allows_promotions=_to_bool(row.get("allows_promotions", "0")),
            promotions_opted_out_at=row.get("promotions_opted_out_at", ""),
            activity_count_180d=_to_int(row.get("activity_count_180d")),
            messages_opened_30d=_to_int(row.get("messages_opened_30d")),
            messages_dismissed_30d=_to_int(row.get("messages_dismissed_30d")),
            messages_replied_30d=_to_int(row.get("messages_replied_30d")),
            last_reply_at=row.get("last_reply_at", ""),
        )
        store.user_business_history[(h.user_id, h.business_id)] = h

    for row in _read_rows(config.MESSAGE_HISTORY_CSV):
        m = Message(
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
        store.message_history[m.message_id] = m
        if m.user_id:
            store.message_history_by_user[m.user_id].append(m.message_id)
        if m.sender_user_id:
            store.message_history_by_sender[m.sender_user_id].append(m.message_id)
        if m.business_id:
            store.message_history_by_business[m.business_id].append(m.message_id)
        if m.group_id:
            store.message_history_by_group[m.group_id].append(m.message_id)

    for row in _read_rows(config.MESSAGE_EVENTS_CSV):
        ev = MessageEvent(
            user_id=row["user_id"],
            message_id=row["message_id"],
            message_opened=_to_bool(row.get("message_opened", "0")),
            message_replied=_to_bool(row.get("message_replied", "0")),
            reaction_time_minutes=_to_float(row.get("reaction_time_minutes")),
            notification_dismissed=_to_bool(row.get("notification_dismissed", "0")),
            muted_after_message=_to_bool(row.get("muted_after_message", "0")),
            message_reported=_to_bool(row.get("message_reported", "0")),
        )
        store.message_events[(ev.user_id, ev.message_id)] = ev

    for row in _read_rows(config.IMAGES_CSV):
        store.images[row["image_id"]] = MediaAsset(row["image_id"], row.get("file_path", ""))

    for row in _read_rows(config.VOICE_NOTES_CSV):
        store.voice_notes[row["voice_note_id"]] = MediaAsset(row["voice_note_id"], row.get("file_path", ""))

    for row in _read_rows(config.DAILY_NOTIFICATION_SUMMARY_CSV):
        d = DailyNotificationSummary(
            user_id=row["user_id"],
            date=row.get("date", ""),
            notifications_sent=_to_int(row.get("notifications_sent")),
            notifications_dismissed=_to_int(row.get("notifications_dismissed")),
        )
        store.daily_summary[(d.user_id, d.date)] = d
        store.daily_summary_by_user[d.user_id].append(d)

    for row in _read_rows(config.MESSAGES_CSV):
        m = Message(
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
        store.messages_to_route.append(m)

    logger.info(
        "Loaded dataset: %d users, %d groups, %d businesses, %d history messages, "
        "%d events, %d to route",
        len(store.users),
        len(store.groups),
        len(store.businesses),
        len(store.message_history),
        len(store.message_events),
        len(store.messages_to_route),
    )
    return store
