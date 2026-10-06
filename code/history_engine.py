"""
history_engine.py
--------------------
Responsibility: everything that requires looking *backwards* at
message_history.csv / message_events.csv.

Two jobs:
1. `find_evidence()` -- retrieval. Finds prior messages (scoped to the same
   user, and preferring the same sender/business/group) that share keyword
   overlap with the incoming message, so decision_engine.py can cite
   `evidence_message_ids` the way sample_messages.csv does.
2. `engagement_rate()` -- behavioral feature. Aggregates how this user has
   historically reacted (opened/replied/dismissed/reported/muted-after) to
   messages from the same sender, business, or group, which is the core
   personalization signal ("a sale poster may be useful for one user and
   noise for another").

Called by: personalization.py, decision_engine.py.
Calls: text_signals.keyword_tokens() for overlap scoring.

Design decision: evidence retrieval is scoped to `message.user_id` first
(the same person's own history is the most defensible evidence for how
*they* will react) and only broadened to sender/group/business scope when
same-user history has no keyword overlap at all, so we never cite another
user's private conversation as evidence for this user's message.
"""

from __future__ import annotations

import config, text_signals
from data_loader import DataStore
from models import EvidenceMatch, Message


def _candidate_ids(message: Message, store: DataStore) -> list[str]:
    """
    Only this user's own history is ever eligible as evidence. Pulling from
    message_history_by_sender/business/group directly would return OTHER
    users' messages with the same sender/business/group, which is a privacy
    bug (citing someone else's private conversation as "evidence" for this
    user's routing decision). Sender/business/group are used only as a
    *relevance* signal (via same_source_bonus below), never as a source of
    additional candidate ids.
    """
    return list(store.message_history_by_user.get(message.user_id, []))


def find_evidence(message: Message, store: DataStore, extra_text: str = "") -> list[EvidenceMatch]:
    target_tokens = text_signals.keyword_tokens(message.message_text + " " + extra_text)
    if not target_tokens:
        return []

    matches: list[EvidenceMatch] = []
    for mid in _candidate_ids(message, store):
        hist = store.message_history.get(mid)
        if hist is None:
            continue
        hist_tokens = text_signals.keyword_tokens(hist.message_text)
        overlap = target_tokens & hist_tokens
        if len(overlap) < config.MIN_EVIDENCE_TOKEN_OVERLAP:
            continue
        same_user_bonus = 0.3 if hist.user_id == message.user_id else 0.0
        same_source_bonus = 0.2 if (
            (message.sender_user_id and hist.sender_user_id == message.sender_user_id)
            or (message.business_id and hist.business_id == message.business_id)
            or (message.group_id and hist.group_id == message.group_id)
        ) else 0.0
        score = len(overlap) + same_user_bonus + same_source_bonus
        matches.append(
            EvidenceMatch(
                message_id=mid,
                score=score,
                reason=f"shares terms: {', '.join(sorted(overlap))[:60]}",
            )
        )

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[: config.MAX_EVIDENCE_IDS]


def engagement_rate(message: Message, store: DataStore) -> dict[str, float]:
    """
    Returns open_rate, reply_rate, dismiss_rate, report_rate, mute_rate over
    this user's historical messages from the same sender/business/group.
    All rates default to 0.5 (neutral) when there is no history to avoid
    biasing new relationships toward either extreme.
    """
    candidate_ids = set(_candidate_ids(message, store))
    # Only count history that actually shares the same conversational source
    # as the incoming message, and belongs to this user, for a clean estimate.
    relevant_ids = []
    for mid in candidate_ids:
        hist = store.message_history.get(mid)
        if hist is None or hist.user_id != message.user_id:
            continue
        same_source = (
            (message.sender_user_id and hist.sender_user_id == message.sender_user_id)
            or (message.business_id and hist.business_id == message.business_id)
            or (message.group_id and hist.group_id == message.group_id)
        )
        if same_source:
            relevant_ids.append(mid)

    if not relevant_ids:
        return {"open_rate": 0.5, "reply_rate": 0.5, "dismiss_rate": 0.5, "report_rate": 0.0, "mute_rate": 0.0, "n": 0}

    opened = replied = dismissed = reported = muted = 0
    n = 0
    for mid in relevant_ids:
        ev = store.message_events.get((message.user_id, mid))
        if ev is None:
            continue
        n += 1
        opened += int(ev.message_opened)
        replied += int(ev.message_replied)
        dismissed += int(ev.notification_dismissed)
        reported += int(ev.message_reported)
        muted += int(ev.muted_after_message)

    if n == 0:
        return {"open_rate": 0.5, "reply_rate": 0.5, "dismiss_rate": 0.5, "report_rate": 0.0, "mute_rate": 0.0, "n": 0}

    return {
        "open_rate": opened / n,
        "reply_rate": replied / n,
        "dismiss_rate": dismissed / n,
        "report_rate": reported / n,
        "mute_rate": muted / n,
        "n": n,
    }
