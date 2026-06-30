"""
Provenance Guard — Flask API

Endpoints:
  POST /submit   — Accept text, run detection pipeline, return classification
  POST /appeal   — Contest a classification by content_id
  GET  /log      — Return the structured audit log (newest first)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

import audit_log as log_store
from scoring import combine_scores, generate_label, get_attribution
from signals import groq_classify, stylometric_classify

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],          # No global default; set per-route
    storage_uri="memory://",    # In-memory storage (per Flask-Limiter >= 3.x requirement)
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bad_request(message: str, code: int = 400):
    return jsonify({"error": message}), code


# ---------------------------------------------------------------------------
# POST /submit
# ---------------------------------------------------------------------------
@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute; 100 per day")
def submit():
    """
    Accept a text submission for AI-origin classification.

    Request body (JSON):
      {
        "text":       "...",   (required, min 10 chars)
        "creator_id": "..."    (required)
      }

    Response:
      {
        "content_id":    "uuid",
        "attribution":   "likely_ai | uncertain | likely_human",
        "confidence":    0.82,
        "llm_score":     0.85,
        "stylo_score":   0.78,
        "stylo_breakdown": { ... },
        "llm_reasoning": "...",
        "label":         "full label text",
        "status":        "classified"
      }
    """
    data = request.get_json(silent=True)
    if not data:
        return _bad_request("Request body must be JSON.")

    text = data.get("text", "").strip()
    creator_id = data.get("creator_id", "").strip()

    if not text:
        return _bad_request("'text' field is required.")
    if len(text) < 10:
        return _bad_request("'text' must be at least 10 characters.")
    if not creator_id:
        return _bad_request("'creator_id' field is required.")

    content_id = str(uuid.uuid4())

    # --- Signal 1: LLM classification ---
    try:
        llm_score, llm_reasoning = groq_classify(text)
    except RuntimeError as exc:
        return jsonify({"error": f"LLM classification failed: {exc}"}), 502

    # --- Signal 2: Stylometric heuristics ---
    stylo_score, stylo_breakdown = stylometric_classify(text)

    # --- Confidence scoring ---
    confidence = combine_scores(llm_score, stylo_score)
    attribution = get_attribution(confidence)
    label = generate_label(confidence, attribution)

    # --- Audit log ---
    entry = log_store.append_classification(
        content_id=content_id,
        creator_id=creator_id,
        text=text,
        attribution=attribution,
        confidence=confidence,
        llm_score=llm_score,
        stylo_score=stylo_score,
        stylo_breakdown=stylo_breakdown,
        llm_reasoning=llm_reasoning,
        label=label,
    )

    return jsonify({
        "content_id": content_id,
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": llm_score,
        "stylo_score": stylo_score,
        "stylo_breakdown": stylo_breakdown,
        "llm_reasoning": llm_reasoning,
        "label": label,
        "status": entry["status"],
    }), 200


# ---------------------------------------------------------------------------
# POST /appeal
# ---------------------------------------------------------------------------
@app.route("/appeal", methods=["POST"])
def appeal():
    """
    Contest a classification.

    Request body (JSON):
      {
        "content_id":        "uuid",    (required)
        "creator_reasoning": "string"   (required, min 10 chars)
      }

    Response:
      {
        "message":    "Appeal received...",
        "content_id": "uuid",
        "status":     "under_review"
      }
    """
    data = request.get_json(silent=True)
    if not data:
        return _bad_request("Request body must be JSON.")

    content_id = data.get("content_id", "").strip()
    reasoning = data.get("creator_reasoning", "").strip()

    if not content_id:
        return _bad_request("'content_id' field is required.")
    if not reasoning:
        return _bad_request("'creator_reasoning' field is required.")
    if len(reasoning) < 10:
        return _bad_request("'creator_reasoning' must be at least 10 characters.")

    updated = log_store.update_appeal(
        content_id=content_id,
        creator_reasoning=reasoning,
    )

    if updated is None:
        return _bad_request(f"No submission found with content_id '{content_id}'.", 404)

    return jsonify({
        "message": "Appeal received. Your content is now under review.",
        "content_id": content_id,
        "status": "under_review",
        "original_attribution": updated.get("attribution"),
        "original_confidence": updated.get("confidence"),
        "appeal_timestamp": updated.get("appeal_timestamp"),
    }), 200


# ---------------------------------------------------------------------------
# GET /log
# ---------------------------------------------------------------------------
@app.route("/log", methods=["GET"])
def get_log():
    """
    Return all audit log entries as JSON (newest first).

    Query params:
      limit (int, optional): cap the number of entries returned
    """
    try:
        limit_param = request.args.get("limit")
        limit = int(limit_param) if limit_param else None
    except ValueError:
        return _bad_request("'limit' must be an integer.")

    entries = log_store.get_all_entries()
    if limit:
        entries = entries[:limit]

    return jsonify({
        "count": len(entries),
        "entries": entries,
    }), 200


# ---------------------------------------------------------------------------
# Rate limit error handler
# ---------------------------------------------------------------------------
@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({
        "error": "Rate limit exceeded. You may submit at most 10 requests per minute and 100 per day.",
        "retry_after": str(e.description),
    }), 429


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)
