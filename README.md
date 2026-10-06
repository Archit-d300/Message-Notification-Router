# Message Notification Router 🔔

A personalized notification router for WhatsApp-style message streams. For every incoming message (text, image, or voice note) it decides whether the user should be interrupted now, shown the message later, or never bothered with it.

Built for the HackerRank Orchestrate (August 2026) challenge.

| Action   | Meaning                                                     |
|----------|-------------------------------------------------------------|
| `notify` | Important enough to interrupt the user now                  |
| `digest` | Useful, but can be shown later in a batch                   |
| `mute`   | Low-value, repetitive, unwanted, suspicious, or unsafe      |

Each prediction also carries a `message_type`, a human-readable `reason`, a calibrated `confidence`, and the `evidence_message_ids` of past messages that support the decision.

---

## How it works

Decisions are **personalized**: the same message can be routed differently for different users based on their relationship with the sender, group or business, their past reactions, quiet hours, and notification load.

```text
messages.csv ──► data_loader ──► for each message:
                                   │
                  ┌────────────────┼─────────────────┐
                  ▼                ▼                 ▼
            media_pipeline    text_signals      history_engine
          (OCR / voice info)  (urgency, payment, (evidence + past
                               promo, scam, OTP,  engagement stats)
                               prompt injection)
                  └────────────────┼─────────────────┘
                                   ▼
                             risk_detector   ── high risk? ──► mute (scam / spam)
                                   │ no
                                   ▼
                            personalization
                       (relationship, engagement,
                        mute state, opt-outs, load)
                                   ▼
                            decision_engine ──► output.csv
```

**Decision order**

1. **Safety first.** `risk_detector` scores scam/spam risk (OTP requests, domain mismatch, unverified or new business accounts, heavy reports, suspicious phrasing). Above the threshold, the message is muted regardless of engagement or mentions.
2. **Personalized scoring.** Relationship strength, engagement rate, trust, urgency and deadlines, direct mentions, muted groups, business opt-outs, and daily notification load are fused into a routing score.
3. **Relevance gate.** A strong relationship alone does not justify `notify`; the message also needs a deadline, mention, actionable content, or an explicit ask.
4. **Classification.** `message_type` comes from a transparent keyword/regex rule set.
5. **Evidence.** Up to 3 related past messages (same user first) are cited.

All thresholds live in `code/config.py`, so behavior can be re-tuned without touching logic.

---

## Project structure

```text
.
├── code/
│   ├── main.py              # entry point: routes every message, writes output.csv
│   ├── config.py            # paths, thresholds, allowed values
│   ├── data_loader.py       # reads all dataset CSVs
│   ├── text_signals.py      # lexicon/regex signals
│   ├── risk_detector.py     # scam/spam scoring (overrides personalization)
│   ├── personalization.py   # per-user relationship and engagement logic
│   ├── history_engine.py    # evidence retrieval and engagement stats
│   ├── decision_engine.py   # fuses signals into the final prediction
│   ├── media_pipeline.py    # dispatches image / voice analysis
│   ├── ocr_processor.py     # Tesseract OCR for images
│   ├── audio_processor.py   # voice note handling (see limitations)
│   ├── llm_client.py        # optional LLM enrichment (off by default)
│   ├── output_writer.py     # validates and writes the output CSV
│   ├── models.py, logging_utils.py
│   ├── evaluation/main.py   # self-evaluation workflow
│   ├── requirements.txt
│   └── .env.example
├── dataset/                 # provided data (not included in code zip)
└── logs/run.log             # created on each run
```

---

## Setup

Requires Python 3.9+.

```bash
cd code
pip install -r requirements.txt
```

The core pipeline uses only the Python standard library. The packages are optional:

- `pytesseract` + `Pillow`: read text from image messages (needs the Tesseract binary)
- `anthropic`: only if you enable LLM enrichment

Install the system tools (optional but recommended):

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr ffmpeg

# macOS
brew install tesseract ffmpeg
```

On Windows, install the [Tesseract binary](https://github.com/tesseract-ocr/tesseract) and FFmpeg and add them to your `PATH`. Without Tesseract, image messages fall back to a no-OCR heuristic instead of crashing.

---

## Run

```bash
cd code
python main.py
```

This reads everything in `dataset/`, routes all messages in `dataset/messages.csv`, and writes `dataset/output.csv`.

```bash
python main.py --output path/to/output.csv   # custom output path
python main.py --verbose                     # debug logging
```

### Output schema

`message_id, action, message_type, reason, confidence, evidence_message_ids`

- `action`: `notify` | `digest` | `mute`
- `message_type`: `personal`, `urgent`, `event`, `payment`, `business_update`, `promotion`, `greeting`, `forward`, `spam`, `scam`, `unknown`
- `confidence`: 0 to 1
- `evidence_message_ids`: semicolon-separated history IDs, or `none`

---

## Evaluation

```bash
cd code
python evaluation/main.py
```

Re-runs the pipeline on the labeled rows in `dataset/sample_messages.csv` (30 messages) and reports action accuracy, message-type accuracy, and a confusion breakdown, listing every mismatch with its reason.

Latest run:

| Metric                    | Result |
|---------------------------|--------|
| Action accuracy           | 96.7%  |
| Message-type accuracy     | 83.3%  |
| Both correct              | 83.3%  |

Caveat: the labeled set is small and was used while developing the rules, so these numbers are optimistic and are a sanity check, not an estimate of hidden-test performance. The official scoring uses hidden labels.

Predictions on the 110 messages in `messages.csv` split into 26 `notify`, 40 `digest`, and 44 `mute`.

---

## Optional: LLM enrichment (off by default)

The pipeline is deterministic and needs no API key. To let Claude rewrite templated reasons into more natural prose:

```bash
cp .env.example .env
# set ENABLE_LLM_ENRICHMENT=true and ANTHROPIC_API_KEY=...
```

`llm_client.py` is the only module that touches the network, and only when both are set. Keep keys in `.env` and never commit it.

---

## Known limitations

- **Voice notes are not transcribed.** No speech-to-text service is bundled, and the code never fabricates a transcript. Voice messages are routed using duration and sender/conversation context only. `audio_processor.set_asr_provider()` is a ready hook for plugging in Whisper or another ASR without changing any other file.
- **Rule-based classification.** `message_type` and risk detection use transparent keyword/regex rules rather than a trained model, to avoid overfitting the small labeled sample and to avoid hardcoding labels.
- **Image understanding is OCR-only.** Visual layout and non-text cues are not analyzed.
- **Confidence is heuristic**, not statistically calibrated.

## Future improvements

- Local or hosted ASR for voice notes
- Vision-language model for posters and screenshots
- LLM second-opinion pass on low-confidence cases
- Calibrate confidence against labeled outcomes
- Learn weights from `message_events.csv` reactions

---

## Rules followed

- Reads the provided CSV and media files
- Writes `output.csv` in the exact required schema
- Includes an evaluation workflow
- No hardcoded test labels or message-specific answers
