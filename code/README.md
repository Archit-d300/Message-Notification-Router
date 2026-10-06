# HackerRank Orchestrate — Message Notification Router

Personalized WhatsApp-style message router. Classifies every message in
`dataset/messages.csv` into `notify` / `digest` / `mute`, with a
`message_type`, human-readable `reason`, `confidence`, and cited
`evidence_message_ids`, written to `dataset/output.csv`.

## Setup

```bash
cd code
pip install -r requirements.txt   # optional: only needed for image OCR
```

The core pipeline (loading data, risk detection, personalization, decision
fusion, output writing) uses **only the Python standard library** and runs
with zero installed packages. `pytesseract` + `Pillow` are optional and only
used to read text out of image messages (falls back gracefully if absent).
`anthropic` is optional and only used if you explicitly opt in to LLM
enrichment (see below) — never required.

Tesseract OCR binary (not a Python package) is needed for image text
extraction: `apt-get install tesseract-ocr` on Debian/Ubuntu, `brew install
tesseract` on macOS, or install the Windows binary from
https://github.com/tesseract-ocr/tesseract. `ffprobe` (part of `ffmpeg`) is
used to read voice-note duration; also optional.

## Run

```bash
cd code
python main.py
```

This reads every file in `dataset/`, analyzes text/image/voice messages,
and writes `dataset/output.csv`. A run log is written to `logs/run.log`.

Options:

```bash
python main.py --output /path/to/output.csv   # custom output path
python main.py --verbose                      # debug-level logging
```

## Self-evaluation (sanity check before submitting)

```bash
python evaluation/main.py
```

Re-runs the pipeline against `dataset/sample_messages.csv` (71 rows with
labels already attached) and reports action/message_type accuracy and a
confusion breakdown. This is **not** the official grader — it exists only
to catch obviously broken behavior before submission.

## Optional: LLM enrichment (off by default)

The pipeline is fully deterministic and requires no API key. If you want
Claude to rewrite templated `reason` strings into more natural prose, or to
add a richer image scene description, set:

```bash
cp .env.example .env
# edit .env:
ENABLE_LLM_ENRICHMENT=true
ANTHROPIC_API_KEY=sk-ant-...
```

`llm_client.py` is the only file that touches the network, and only when
both variables above are set.

## Architecture

```
main.py
  └─ data_loader.py        (reads every dataset/*.csv into a typed DataStore)
  └─ for each message:
       media_pipeline.py   (dispatch image/voice)
         ├─ ocr_processor.py    (Tesseract OCR -> text + scam/urgency hints)
         └─ audio_processor.py (duration heuristic; honest "no transcript"
                                 disclosure; pluggable real-ASR hook)
       decision_engine.py  (fusion — the only module that produces the
                             final Prediction)
         ├─ text_signals.py     (regex/lexicon: urgency, payment, promo,
         │                       scam phrasing, OTP requests, prompt-
         │                       injection detection, deadlines, mentions)
         ├─ risk_detector.py    (scam/spam scoring; OVERRIDES personalization)
         ├─ personalization.py  (relationship strength, engagement rate,
         │                       mute/opt-out state, notification load)
         └─ history_engine.py   (evidence retrieval + engagement-rate stats
                                  from message_history.csv / message_events.csv)
  └─ output_writer.py       (validate + write dataset/output.csv)
```

**Decision rule, in order:**

1. `risk_detector` runs first. If risk_score crosses a threshold, the
   message is muted as `scam`/`spam` regardless of relationship strength,
   engagement, or direct mentions — safety always overrides personalization.
2. Otherwise, `decision_engine` fuses relationship strength, engagement
   rate, trust level, urgency/deadline signals, direct mentions, mute/
   opt-out state, and notification load into a routing score, gated by a
   content-relevance check (a strong relationship alone is not sufficient
   to justify `notify` for a mundane update — it also needs a deadline,
   mention, actionable business content, or an explicit ask).
3. `message_type` is classified from keyword/regex categories (checked
   before a generic deadline-driven "urgent" guess, so a business's normal
   delivery ETA doesn't get miscategorized).
4. `evidence_message_ids` cites up to 3 prior messages (scoped to the same
   user first) that share keyword overlap with the incoming message.

## Known limitations (disclosed, not hidden)

- **Voice notes are not transcribed.** No network-connected ASR service is
  available in the reference environment, and this codebase will not
  fabricate a transcript. `audio_processor.py` uses duration as an honest
  fallback signal and exposes `set_asr_provider()` so real transcription
  (Whisper API, local model, etc.) can be dropped in with no other file
  changing.
- **message_type classification is a transparent rule set**, not a trained
  classifier — chosen deliberately because the labelled sample is small
  (71 rows) and dataset-specific overfitting is explicitly disallowed by
  the challenge rules.
