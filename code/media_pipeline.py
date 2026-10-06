"""
media_pipeline.py
-------------------
Thin dispatcher: given a Message with media_type in {image, voice}, resolve
its file path via the DataStore and call the matching analyzer. Kept
separate from ocr_processor/audio_processor so decision_engine.py has one
call site regardless of media type, and so a new media type (e.g. "video")
can be added by adding one branch here.

Called by: main.py (per message).
Calls: ocr_processor.analyze_image, audio_processor.analyze_voice_note.
"""

from __future__ import annotations

import logging

import audio_processor, config, ocr_processor
from data_loader import DataStore
from models import MediaAnalysis, Message

logger = logging.getLogger("orchestrate.media_pipeline")

# In-process cache keyed by "media_type:media_id". The same media_id is
# referenced by more than one message in this dataset (e.g. a poster
# forwarded to 3 different conversations) -- OCR/ffprobe is real work we
# should not repeat. Process-lifetime only, no disk persistence needed for
# a single `python main.py` run.
_analysis_cache: dict[str, MediaAnalysis] = {}


def analyze(message: Message, store: DataStore) -> MediaAnalysis:
    if not message.has_media or not message.media_id:
        return MediaAnalysis(source="none")

    cache_key = f"{message.media_type}:{message.media_id}"
    cached = _analysis_cache.get(cache_key)
    if cached is not None:
        return cached

    if message.media_type == "image":
        asset = store.images.get(message.media_id)
        if asset is None:
            logger.warning("No images.csv entry for media_id=%s", message.media_id)
            return MediaAnalysis(source="image", notes="media_id not found in images.csv")
        result = ocr_processor.analyze_image(config.DATASET_DIR / asset.file_path)
    elif message.media_type == "voice":
        asset = store.voice_notes.get(message.media_id)
        if asset is None:
            logger.warning("No voice_notes.csv entry for media_id=%s", message.media_id)
            return MediaAnalysis(source="voice", notes="media_id not found in voice_notes.csv")
        result = audio_processor.analyze_voice_note(config.DATASET_DIR / asset.file_path)
    else:
        return MediaAnalysis(source="none", notes=f"Unrecognized media_type={message.media_type!r}")

    _analysis_cache[cache_key] = result
    return result
