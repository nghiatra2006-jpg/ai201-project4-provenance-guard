"""
Structured audit logger.

Stores every attribution decision and appeal as a JSON array in audit_log.json.
On startup, loads the existing log if the file exists.

Log entry schema:
{
  "content_id": "uuid",
  "creator_id": "string",
  "timestamp": "ISO8601",
  "text_preview": "first 120 chars of the submitted text",
  "attribution": "likely_ai | uncertain | likely_human",
  "confidence": 0.82,
  "llm_score": 0.85,
  "stylo_score": 0.78,
  "stylo_breakdown": { ... },
  "llm_reasoning": "...",
  "label": "full label text",
  "status": "classified | under_review",
  "appeal_reasoning": null | "string",
  "appeal_timestamp": null | "ISO8601"
}
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path(__file__).parent / "audit_log.json"

_lock = threading.Lock()


def _load() -> list[dict]:
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save(entries: list[dict]) -> None:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def append_classification(
    *,
    content_id: str,
    creator_id: str,
    text: str,
    attribution: str,
    confidence: float,
    llm_score: float,
    stylo_score: float,
    stylo_breakdown: dict,
    llm_reasoning: str,
    label: str,
) -> dict:
    """
    Write a new classification entry to the audit log.

    Returns the entry dict (useful for the HTTP response).
    """
    entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text_preview": text[:120].replace("\n", " "),
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": llm_score,
        "stylo_score": stylo_score,
        "stylo_breakdown": stylo_breakdown,
        "llm_reasoning": llm_reasoning,
        "label": label,
        "status": "classified",
        "appeal_reasoning": None,
        "appeal_timestamp": None,
    }

    with _lock:
        entries = _load()
        entries.append(entry)
        _save(entries)

    return entry


def update_appeal(*, content_id: str, creator_reasoning: str) -> dict | None:
    """
    Update an existing log entry to mark it as under review.

    Returns the updated entry, or None if content_id was not found.
    """
    with _lock:
        entries = _load()
        for entry in entries:
            if entry.get("content_id") == content_id:
                entry["status"] = "under_review"
                entry["appeal_reasoning"] = creator_reasoning
                entry["appeal_timestamp"] = datetime.now(timezone.utc).isoformat()
                _save(entries)
                return entry
    return None


def get_all_entries() -> list[dict]:
    """Return all audit log entries, newest first."""
    with _lock:
        entries = _load()
    return list(reversed(entries))


def get_entry(content_id: str) -> dict | None:
    """Return a single entry by content_id, or None."""
    with _lock:
        entries = _load()
    for entry in entries:
        if entry.get("content_id") == content_id:
            return entry
    return None
