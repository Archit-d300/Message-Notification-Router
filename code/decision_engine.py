"""
decision_engine.py
--------------------
Responsibility: the single place where risk_detector, personalization,
text_signals, and media analysis are fused into one Prediction. This is the
"decision fusion" module referenced throughout the architecture: every other
module produces a *signal*, this module produces the *decision*.

Called by: main.py (once per message).
Calls: risk_detector.assess(), personalization.build(), history_engine.find_evidence(),
text_signals.extract(), media_pipeline (analysis already computed upstream and passed in).

Core rule ("safety always overrides personalization"): risk_detector's
verdict is checked FIRST. If risk_score crosses RISK_MUTE_THRESHOLD, the
message is muted as scam/spam no matter how strong the relationship or how
direct a mention is. Only once risk clears the bar does personalization
(relationship strength, engagement, mute/opt-out state, notification load,
direct mentions) determine notify vs digest vs mute.
"""

from __future__ import annotations

import config, text_signals
from data_loader import DataStore
from history_engine import find_evidence
from models import MediaAnalysis, Message, Prediction
from personalization import build as build_relationship_signals
from risk_detector import assess as assess_risk


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _classify_message_type(
    message: Message,
    sig: text_signals.TextSignals,
    media: MediaAnalysis,
    risk_type: str | None,
) -> str:
    if risk_type == "scam":
        return "scam"
    if risk_type == "spam":
        return "spam"

    has_deadline = bool(sig.deadline_mentions) or media.urgency_hint >= 0.7
    top = sig.top_category

    if media.category_hint == "payment_screenshot" and top in {"payment", ""}:
        return "payment"

    # Explicit keyword categories are more specific than a bare deadline
    # mention, so they win first. "urgent" only wins here when the message
    # actually used urgent language (not just "today"/"by 5pm").
    if top in {"event", "payment", "promotion", "greeting", "forward"}:
        return top
    if top == "urgent":
        return "urgent"
    if message.forwarded_count >= 8:
        return "forward"
    if has_deadline and message.conversation_type != "business":
        return "urgent"

    if sig.mentions and message.conversation_type in {"group", "personal"}:
        return "personal"
    if message.conversation_type == "business":
        return "business_update"
    if message.media_type == "voice" and not media.extracted_text:
        # Honest fallback: without a transcript we cannot tell a quick
        # acknowledgement from an urgent request, so default to the more
        # common, lower-stakes bucket rather than guessing "event".
        return "personal"
    if message.conversation_type in {"personal", "group"}:
        return "personal" if (message.message_text.strip() or media.source != "none") else "unknown"
    return "unknown"


def _confidence_from_margin(score: float, low: float, high: float) -> float:
    span = max(high - low, 1e-6)
    if score >= high:
        margin = min(1.0, (score - high) / span)
    elif score <= low:
        margin = min(1.0, (low - score) / span)
    else:
        # inside the digest band: confidence highest at the band center
        center = (low + high) / 2
        margin = 1.0 - (abs(score - center) / (span / 2))
    return _clamp(config.CONFIDENCE_FLOOR + margin * (config.CONFIDENCE_CEILING - config.CONFIDENCE_FLOOR),
                  config.CONFIDENCE_FLOOR, config.CONFIDENCE_CEILING)


def _build_reason(
    action: str,
    message_type: str,
    message: Message,
    rel,
    risk_indicators: list[str],
    has_deadline: bool,
) -> str:
    if action == "mute" and message_type in {"scam", "spam"}:
        top_indicator = risk_indicators[0] if risk_indicators else "suspicious content patterns"
        return f"Flagged as {message_type}: {top_indicator}; muted regardless of usual engagement."

    if action == "notify" and rel.is_direct_mention and rel.is_muted_source:
        return "Directly mentions this user inside an otherwise muted source, so it is surfaced despite the mute."

    if action == "mute" and rel.is_muted_source:
        source = "group" if message.conversation_type == "group" else "business"
        return f"User has muted or opted out of this {source} and the message has no urgent personal relevance."

    if action == "mute":
        return "Low historical engagement with this sender and no urgent or personalized content."

    if action == "notify" and has_deadline:
        return f"Time-sensitive {message_type} content the user is likely to need to act on now."

    if action == "notify" and rel.relationship_strength >= 0.6:
        return f"Strong relationship history with this sender/source supports interrupting the user now."

    if action == "notify":
        return f"Content and sender context indicate this {message_type} message is important enough to surface now."

    if action == "digest" and message_type == "promotion":
        return "Promotional content the user has some relationship with, but not urgent enough to interrupt now."

    if action == "digest":
        return f"Useful {message_type} content, but not time-critical enough to interrupt the user right now."

    return "Routine content evaluated against user history and relationship strength."


def decide(message: Message, media: MediaAnalysis, store: DataStore) -> Prediction:
    combined_text = " ".join(filter(None, [message.message_text, media.extracted_text]))
    sig = text_signals.extract(combined_text)

    business = store.businesses.get(message.business_id) if message.business_id else None
    risk = assess_risk(message, media, business)
    rel = build_relationship_signals(message, store, sig)

    message_type = _classify_message_type(message, sig, media, risk.risk_type)
    has_deadline = bool(sig.deadline_mentions) or media.urgency_hint >= 0.7

    if risk.risk_type is not None:
        action = "mute"
        confidence = _clamp(config.CONFIDENCE_FLOOR + risk.risk_score * (
            config.CONFIDENCE_CEILING - config.CONFIDENCE_FLOOR), config.CONFIDENCE_FLOOR, config.CONFIDENCE_CEILING)
    else:
        base = 0.5 * rel.relationship_strength + 0.3 * rel.engagement_rate + 0.2 * rel.trust_level

        if message_type in {"urgent", "event", "payment"} and has_deadline:
            base += 0.22
        if rel.is_direct_mention:
            base += config.DIRECT_MENTION_BOOST
        if message_type == "promotion":
            base -= 0.15 if rel.engagement_rate < 0.5 else 0.0
        if message_type in {"greeting", "forward"}:
            base -= 0.2
        if rel.is_muted_source and not rel.is_direct_mention:
            penalty = config.GROUP_MUTED_PENALTY if message.conversation_type == "group" else config.BUSINESS_OPT_OUT_PENALTY
            base -= penalty
        if rel.is_overloaded_day and message_type not in {"urgent", "payment"}:
            base -= config.OVERLOAD_DIGEST_PENALTY

        if sig.explicit_non_urgent and message_type not in {"greeting", "forward"}:
            base -= 0.2
        if message.conversation_type == "business" and sig.actionable_business_hits:
            base += 0.15

        score = _clamp(base, 0.0, 1.0)

        requires_response = ("?" in message.message_text or any(
            phrase in message.message_text.lower()
            for phrase in ("can you", "could you", "please call", "please reply", "need you to", "let me know")
        )) and not sig.explicit_non_urgent
        content_justifies_notify = (
            has_deadline
            or rel.is_direct_mention
            or message_type in {"urgent", "payment"}
            or (message_type in {"business_update", "event"} and rel.relationship_strength >= 0.55)
            or (message.conversation_type == "business" and sig.actionable_business_hits)
            or (
                message.conversation_type == "personal"
                and rel.engagement_rate >= 0.65
                and requires_response
            )
        )

        if score >= config.NOTIFY_THRESHOLD and content_justifies_notify:
            action = "notify"
        elif score >= config.DIGEST_THRESHOLD:
            action = "digest"
        else:
            action = "mute"

        confidence = _confidence_from_margin(score, config.DIGEST_THRESHOLD, config.NOTIFY_THRESHOLD)

    evidence = find_evidence(message, store, extra_text=media.extracted_text)
    evidence_ids = ";".join(m.message_id for m in evidence) if evidence else "none"

    reason = _build_reason(action, message_type, message, rel, risk.indicators, has_deadline)

    return Prediction(
        message_id=message.message_id,
        action=action,
        message_type=message_type,
        reason=reason,
        confidence=round(confidence, 2),
        evidence_message_ids=evidence_ids,
    )
