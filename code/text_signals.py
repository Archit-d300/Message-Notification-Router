"""
text_signals.py
----------------
Pure text/regex feature extraction shared by risk_detector.py and
decision_engine.py's message_type classifier. No I/O, no dataset knowledge --
this module only ever sees a string and returns structured signals, which
keeps it trivially unit-testable and reusable for OCR text and (if ASR is
ever plugged in) transcripts too.

Design decision: lexicon + regex rather than a trained classifier. The
dataset has ~110 rows to predict and no labelled training set of comparable
size, so a supervised classifier would overfit. Transparent keyword/pattern
rules are also required by the "do not hardcode dataset-specific logic"
rule: every pattern here is a generic language pattern (e.g. "OTP", "verify
your account", a URL regex), never a literal sender name or message id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

URL_RE = re.compile(r"(https?://\S+|\b[a-z0-9-]+\.(?:in|com|co|net|org|xyz|info|link|shop)\b(?:/\S*)?)", re.I)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-\s]?)?\d{4,5}[-\s]?\d{5}\b")
MONEY_RE = re.compile(r"(?:₹|rs\.?|inr|\$|usd)\s?[\d,]+(?:\.\d+)?", re.I)
OTP_RE = re.compile(
    r"\botp\b|\bone[-\s]time password\b|\b\d{4,6}[-\s]?digit\b|"
    r"\b(login|verification|security|access) code\b|\b\d{4,6}\b.{0,15}\bcode\b",
    re.I,
)
# Deliberately narrower than "contains the word today/tonight" -- casual chat
# ("anyone watching the match tonight?") mentions "tonight" constantly without
# being a real deadline. Only count phrasing that names a concrete cutoff.
DEADLINE_RE = re.compile(
    r"\b(by \d{1,2}:\d{2}(\s?(am|pm))?|by \d{1,2}\s?(am|pm)|"
    r"within \d+\s?(hours?|hrs?|mins?|minutes?|days?)|"
    r"in \d+\s?(hours?|hrs?|mins?|minutes?)|"
    r"eod|end of day|expires?( today| soon)?|last date|deadline|"
    r"due (today|tomorrow|by)|closes? (today|tomorrow)|"
    r"before (the |our |your |my )?(\w+\s+)?(meeting|call|review|interview|standup|deadline))\b",
    re.I,
)
MENTION_RE = re.compile(r"@([\w.]+)")
INJECTION_RE = re.compile(
    r"\bignore (all )?(the )?(previous|prior|above) (instructions|rules|routing)\b|"
    r"\bdisregard (the )?(previous|prior|above)\b|\boverride (the )?(routing|decision|system)\b|"
    r"\bmark this (message|as) (notify|urgent|high)\b|\byou (must|should) (notify|mark)\b",
    re.I,
)
# A bare "payment" following a hyphen is almost always a compound modifier
# ("failed-payment screenshots", "advance-payment proof") describing some
# OTHER topic, not an actual bill/transaction the user needs to act on.
# Excluding that case (but still matching "Payment due today", "Maintenance
# payment aaj...") meaningfully cuts false "payment" classifications without
# touching any genuinely payment-related message in the dataset.
PAYMENT_BARE_RE = re.compile(r"(?<![\w-])payment(?!-)\b", re.I)
# "refund" alone is topic-ambiguous ("I found two refundable stays", "the
# refund edge case" in a work discussion). Only treat it as a payment signal
# when it co-occurs with an actual transactional cue.
REFUND_CONTEXT_RE = re.compile(
    r"\b(approved|processed|processing|verify|release|released|pending|wallet|"
    r"card details|bank details|amount|claim|credited|denied|rejected|initiated)\b",
    re.I,
)
# Messages that WARN a user not to share sensitive info ("we will never ask
# for your OTP") are the opposite of a phishing attempt; without this, the
# safety-advisory language itself was being read as an OTP/payment request.
ADVISORY_RE = re.compile(
    r"\b(never|don'?t|do not|won'?t|will not)\s+(ask(?:s)?(?: you)? for|share|send|give out|request)\b"
    r"[^.?!\n]{0,60}\b(otp|password|pin|payment details|card details|bank details|cvv)\b",
    re.I,
)

URGENT_WORDS = (
    "urgent", "immediately", "asap", "emergency", "right now", "act now",
    "act fast", "important", "action required", "final notice", "last chance",
    "heads-up", "heads up", "need you", "please respond", "escalation",
    "escalating",
)
# "payment" itself is matched separately via PAYMENT_BARE_RE (word-boundary +
# hyphen-aware) rather than as a plain substring here -- see that regex's
# docstring for why. "refund" is matched separately too, gated by
# REFUND_CONTEXT_RE, since it is topic-ambiguous on its own.
PAYMENT_WORDS = (
    "invoice", "bill due", "autopay", "pay now", "reattempt fee",
    "transaction", "debited", "credited", "account balance",
)
PROMOTION_WORDS = (
    "sale", "discount", "offer", "% off", "coupon", "deal", "flash sale",
    "limited time", "buy now", "shop now", "new arrivals", "clearance",
    "selling", "for sale", "available for", "still available", "still up for grabs",
    "unsubscribe", "marketing message", "marketing messages",
)
GREETING_WORDS = (
    "good morning", "good night", "happy diwali", "happy new year",
    "blessed day", "gm ", "gn ", "wishing you",
)
FORWARD_CHAIN_WORDS = (
    "forward this", "send this to", "send to 10", "share with your contacts",
    "if you don't forward", "true story", "please forward",
)
EVENT_WORDS = (
    "event", "meeting", "reminder", "schedule", "rsvp", "invite",
    "registration", "appointment", "booking confirmed", "class", "bus",
    "form is open", "sign up", "sign-up",
)
ACTIONABLE_BUSINESS_WORDS = (
    "order", "delivery", "tracking", "dispatched", "prescription", "appointment",
    "claim", "pickup", "shipment", "arriving", "out for delivery",
)
SCAM_PHRASES = (
    "verify your account", "confirm your identity", "suspend", "blocked",
    "reactivate", "click the link", "claim your prize", "you have won",
    "lottery", "tax refund", "kyc update", "update your kyc",
    "share your otp", "enter your otp", "enter the code", "reset your password",
    "guaranteed returns", "double your money", "investment opportunity",
    "limited slots", "crypto giveaway", "prize money", "gift card",
    "reattempt fee", "delivery failed", "customs fee", "pay small fee",
    "ignore all previous", "ignore previous instructions", "disregard the previous",
    "override the routing", "mark this message as notify",
)
IMPERSONATION_WORDS = (
    "bank support", "customer care", "official support", "help desk",
    "we are calling from", "on behalf of", "government department",
)
NOT_URGENT_PHRASES = (
    "nothing urgent", "no rush", "not urgent", "no hurry", "whenever you can",
    "no need to reply immediately", "take your time", "no need to reply",
    "no need to respond", "whenever you get time", "whenever convenient",
)


@dataclass
class TextSignals:
    urls: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    money_mentions: list[str] = field(default_factory=list)
    has_otp_request: bool = False
    deadline_mentions: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    scam_phrase_hits: list[str] = field(default_factory=list)
    impersonation_hits: list[str] = field(default_factory=list)
    injection_detected: bool = False
    explicit_non_urgent: bool = False
    actionable_business_hits: list[str] = field(default_factory=list)
    category_scores: dict[str, float] = field(default_factory=dict)

    @property
    def top_category(self) -> str:
        if not self.category_scores:
            return "unknown"
        return max(self.category_scores, key=lambda k: self.category_scores[k])


def _count_hits(text_lower: str, phrases: tuple[str, ...]) -> list[str]:
    return [p for p in phrases if p in text_lower]


def extract(text: str) -> TextSignals:
    text = text or ""
    lower = text.lower()

    is_advisory = bool(ADVISORY_RE.search(text))
    money_mentions = MONEY_RE.findall(text)

    sig = TextSignals(
        urls=URL_RE.findall(text),
        phones=PHONE_RE.findall(text),
        money_mentions=money_mentions,
        # A message that WARNS the user not to share an OTP/password (a
        # legitimate safety advisory) is the opposite of a message REQUESTING
        # one -- don't let the advisory language itself trip the OTP signal.
        has_otp_request=bool(OTP_RE.search(text)) and not is_advisory,
        deadline_mentions=[m[0] if isinstance(m, tuple) else m for m in DEADLINE_RE.findall(text)],
        mentions=MENTION_RE.findall(text),
        scam_phrase_hits=_count_hits(lower, SCAM_PHRASES),
        impersonation_hits=_count_hits(lower, IMPERSONATION_WORDS),
        injection_detected=bool(INJECTION_RE.search(text)),
        explicit_non_urgent=bool(_count_hits(lower, NOT_URGENT_PHRASES)),
        actionable_business_hits=_count_hits(lower, ACTIONABLE_BUSINESS_WORDS),
    )

    # "payment" is matched with a hyphen-aware regex (not a plain substring)
    # so compound modifiers like "failed-payment screenshots" don't count as
    # a payment signal. "refund" only counts when paired with an actual
    # transactional cue, since on its own it is topic-ambiguous ("I found
    # two refundable stays", "the refund edge case" in a work discussion).
    payment_hit_count = len(_count_hits(lower, PAYMENT_WORDS))
    if PAYMENT_BARE_RE.search(text) and not is_advisory:
        payment_hit_count += 1
    if "refund" in lower and (money_mentions or REFUND_CONTEXT_RE.search(lower)):
        payment_hit_count += 1
    payment_money_bonus = 0.5 if (money_mentions and payment_hit_count) else 0.0

    scores = {
        # Deliberately keyword-only here: a delivery ETA or event date already
        # sets deadline_mentions, which decision_engine uses to boost the
        # *action* (notify) without forcing the *category* to "urgent".
        # Zeroed out when the message explicitly says it's NOT urgent (e.g.
        # "Nothing urgent") -- otherwise the literal word "urgent" inside
        # that negated phrase was scoring the message as urgent.
        "urgent": 0.0 if sig.explicit_non_urgent else len(_count_hits(lower, URGENT_WORDS)) * 1.0,
        "payment": payment_hit_count * 1.0 + payment_money_bonus,
        "promotion": len(_count_hits(lower, PROMOTION_WORDS)) * 1.0,
        "greeting": len(_count_hits(lower, GREETING_WORDS)) * 1.0,
        "forward": len(_count_hits(lower, FORWARD_CHAIN_WORDS)) * 1.0,
        "event": len(_count_hits(lower, EVENT_WORDS)) * 1.0,
    }
    sig.category_scores = {k: v for k, v in scores.items() if v > 0}
    return sig


def keyword_tokens(text: str) -> set[str]:
    """Lightweight token set used for historical-evidence overlap matching."""
    text = (text or "").lower()
    words = re.findall(r"[a-z0-9]{4,}", text)
    stop = {
        "this", "that", "with", "your", "have", "will", "from", "just",
        "please", "today", "know", "want", "there", "here", "were", "when",
    }
    return {w for w in words if w not in stop}
