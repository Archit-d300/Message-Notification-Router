"""
personalization.py
--------------------
Responsibility: turn raw relationship data (group role/activity, business
opt-in history, past open/reply behavior, current notification load) into a
single RelationshipSignals object that decision_engine.py can fuse with
content urgency and risk.

Called by: decision_engine.py.
Calls: history_engine.engagement_rate(), text_signals for mention parsing.

Input: Message, DataStore, TextSignals (for @mentions).
Output: RelationshipSignals.

Design decisions:
- Every sub-score is normalized to [0, 1] so decision_engine.py can combine
  them with fixed weights instead of re-deriving scale for each source type.
- A group being muted by the user, or a business being explicitly opted
  out of promotions, is treated as a strong-but-not-absolute penalty (not
  an automatic mute) because the dataset explicitly calls out that "a muted
  family group can still contain an urgent direct mention" -- the penalty
  is applied as a score offset in decision_engine.py, then a direct-mention
  boost can still pull the message back up to notify.
"""

from __future__ import annotations

import config
from data_loader import DataStore
from history_engine import engagement_rate
from models import Message, RelationshipSignals
from text_signals import TextSignals


def _date_from_created_at(created_at: str) -> str:
    return created_at.split(" ")[0] if created_at else ""


def _is_overloaded(message: Message, store: DataStore) -> bool:
    date = _date_from_created_at(message.created_at)
    summary = store.daily_summary.get((message.user_id, date))
    if summary is None:
        return False
    return summary.notifications_sent >= config.DAILY_NOTIFICATIONS_OVERLOAD


def _direct_mention(message: Message, sig: TextSignals) -> bool:
    if not sig.mentions:
        return False
    user_token = message.user_id.strip().lower()
    return any(m.strip().lower() == user_token for m in sig.mentions)


def build(message: Message, store: DataStore, sig: TextSignals) -> RelationshipSignals:
    notes: list[str] = []
    strength = 0.4  # neutral prior
    engagement = 0.5
    trust = 0.3
    is_muted = False

    eng = engagement_rate(message, store)
    engagement = 0.6 * eng["reply_rate"] + 0.4 * eng["open_rate"]
    if eng["report_rate"] > 0 or eng["mute_rate"] > 0.5:
        engagement = min(engagement, 0.2)
        notes.append("user has reported or muted this source before")

    if message.conversation_type == "personal":
        strength = 0.5 + 0.3 * eng["reply_rate"]
        trust = 0.5 if eng["n"] > 0 else 0.35
        if eng["n"] == 0:
            notes.append("no prior personal history with this sender")

    elif message.conversation_type == "group":
        membership = store.group_members.get((message.group_id, message.user_id))
        group = store.groups.get(message.group_id)
        if membership is not None:
            is_muted = membership.group_muted_by_user
            read_signal = 1.0 if membership.messages_read_30d > 0 else 0.3
            reply_signal = min(1.0, membership.replies_sent_30d / 5.0)
            role_bonus = 0.15 if membership.role == "admin" else 0.0
            strength = 0.3 + 0.35 * read_signal + 0.25 * reply_signal + role_bonus
            if is_muted:
                notes.append(f"user has muted the '{group.group_name if group else message.group_id}' group")
        if group is not None:
            trust = 0.6 if group.group_type in {"family", "coworker", "school_group"} else 0.4

    elif message.conversation_type == "business":
        biz_hist = store.user_business_history.get((message.user_id, message.business_id))
        business = store.businesses.get(message.business_id)
        if biz_hist is not None:
            activity_signal = min(1.0, biz_hist.activity_count_180d / 8.0)
            reply_signal = min(1.0, biz_hist.messages_replied_30d / 3.0)
            strength = 0.25 + 0.45 * activity_signal + 0.3 * reply_signal
            if not biz_hist.allows_promotions and biz_hist.promotions_opted_out_at:
                is_muted = True
                notes.append("user has opted out of promotions from this business")
            if biz_hist.messages_dismissed_30d > biz_hist.messages_opened_30d:
                notes.append("user dismisses more messages than they open from this business")
        else:
            strength = 0.2
            notes.append("no prior relationship with this business")
        if business is not None:
            trust = 0.7 if business.verified else 0.25
            if business.user_reports_30d >= config.HIGH_REPORT_COUNT_THRESHOLD:
                trust = min(trust, 0.15)

    return RelationshipSignals(
        relationship_strength=max(0.0, min(1.0, strength)),
        engagement_rate=max(0.0, min(1.0, engagement)),
        is_muted_source=is_muted,
        is_direct_mention=_direct_mention(message, sig),
        is_overloaded_day=_is_overloaded(message, store),
        trust_level=max(0.0, min(1.0, trust)),
        notes=notes,
    )
