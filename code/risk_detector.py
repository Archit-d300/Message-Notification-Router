"""
risk_detector.py
------------------
Responsibility: compute a single risk_score in [0, 1] and a risk_type
("scam" | "spam" | None) for a message, combining message text, OCR/voice
media analysis, and sender/business trust metadata. This is the ONLY module
whose output is allowed to override personalization -- decision_engine.py
enforces that a high risk_score forces action=mute regardless of how
engaged the user normally is with the sender.

Called by: decision_engine.py.
Calls: text_signals.extract().

Input: Message, MediaAnalysis, optional BusinessAccount.
Output: RiskAssessment.

Design decisions:
- Business-side signals (unverified account, brand domain mismatch, very
  new sending domain, high recent report count) are weighted heavily
  because they are structural, dataset-provided facts, not fragile keyword
  guesses.
- Content-side signals (OTP requests, "your account is suspended", payment
  links, urgency + money + link combined) catch phishing/social engineering
  even from senders with no business_accounts.csv row (personal/group scams,
  which the dataset explicitly includes, e.g. OTP-sharing requests from a
  personal contact).
- forwarded_count is a spam/virality signal, not a scam signal by itself --
  a highly forwarded chain message is classified as spam-leaning, not scam,
  unless it also carries phishing content.
"""

from __future__ import annotations

import config, text_signals
from models import BusinessAccount, MediaAnalysis, Message, RiskAssessment


def _business_risk(business: BusinessAccount | None) -> tuple[float, list[str]]:
    if business is None:
        return 0.0, []

    score = 0.0
    indicators: list[str] = []

    if not business.verified:
        score += config.UNVERIFIED_BUSINESS_WEIGHT
        indicators.append("sending business account is not verified")

    if (
        business.official_domain
        and business.domain_used_by_sender
        and business.official_domain.lower() != business.domain_used_by_sender.lower()
    ):
        score += config.DOMAIN_MISMATCH_WEIGHT
        indicators.append(
            f"sender domain '{business.domain_used_by_sender}' does not match "
            f"official domain '{business.official_domain}'"
        )

    if 0 <= business.domain_used_by_sender_age_days < config.NEW_ACCOUNT_DAYS_THRESHOLD:
        score += config.NEW_ACCOUNT_WEIGHT
        indicators.append("sending domain was registered very recently")

    if 0 <= business.account_age_days < config.NEW_ACCOUNT_DAYS_THRESHOLD:
        score += config.NEW_ACCOUNT_WEIGHT
        indicators.append("business account itself was created very recently")

    if business.user_reports_30d >= config.HIGH_REPORT_COUNT_THRESHOLD:
        score += config.HIGH_REPORT_WEIGHT
        indicators.append(f"{business.user_reports_30d} user reports against this sender in 30 days")

    return score, indicators


def assess(
    message: Message,
    media: MediaAnalysis,
    business: BusinessAccount | None,
) -> RiskAssessment:
    combined_text = " ".join(filter(None, [message.message_text, media.extracted_text]))
    sig = text_signals.extract(combined_text)

    score = 0.0
    indicators: list[str] = []

    if sig.injection_detected:
        score += 0.6
        indicators.append("message attempts to instruct the routing system directly (prompt-injection pattern)")

    if sig.has_otp_request:
        score += 0.45
        indicators.append("message asks the recipient to share an OTP/verification code")
        if business is None:
            score += 0.15
            indicators.append("request comes from a personal/group sender rather than a verified business")

    if sig.scam_phrase_hits:
        score += min(0.45, 0.15 * len(sig.scam_phrase_hits))
        indicators.append("contains classic scam phrasing (" + ", ".join(sig.scam_phrase_hits[:3]) + ")")

    if sig.impersonation_hits and (message.conversation_type == "personal" or business is None):
        score += 0.2
        indicators.append("claims to be official support/bank/government without a verified business account")

    if sig.urls and (sig.has_otp_request or sig.scam_phrase_hits):
        score += 0.15
        indicators.append("combines a link with urgency/credential-harvesting language")

    if sig.money_mentions and sig.urls and business is None:
        score += 0.15
        indicators.append("requests payment via an external link outside any known business account")

    biz_score, biz_indicators = _business_risk(business)
    score += biz_score
    indicators.extend(biz_indicators)

    score += media.scam_indicator_score * 0.5
    if media.notes and media.scam_indicator_score > 0:
        indicators.append(f"media analysis: {media.notes}")

    score = min(1.0, score)

    risk_type = None
    if score >= config.RISK_MUTE_THRESHOLD:
        # Distinguish scam (targeted credential/payment harvesting) from spam
        # (mass forwarded / repetitive low-value) using forwarded_count.
        if sig.has_otp_request or sig.scam_phrase_hits or sig.impersonation_hits or biz_indicators:
            risk_type = "scam"
        elif message.forwarded_count >= 5:
            risk_type = "spam"
        else:
            risk_type = "scam"

    return RiskAssessment(risk_score=score, risk_type=risk_type, indicators=indicators)
