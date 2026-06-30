"""
Local integration test for Provenance Guard.

Mocks the Groq API call so the full pipeline can be tested without an API key.
Run with:  python test_local.py
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Set a dummy key so the import doesn't fail
os.environ.setdefault("GROQ_API_KEY", "test-key-dummy")


def make_app():
    """Import and return the Flask app (import here to respect env setup above)."""
    import app as flask_app
    flask_app.app.config["TESTING"] = True
    # Disable rate limiting in tests
    flask_app.limiter.enabled = False
    return flask_app.app


class TestProvenanceGuard(unittest.TestCase):

    def setUp(self):
        self.app = make_app()
        self.client = self.app.test_client()
        # Remove any stale audit log from previous runs
        log_path = Path(__file__).parent / "audit_log.json"
        if log_path.exists():
            log_path.unlink()

    def _mock_groq(self, score: float, reasoning: str = "test reasoning"):
        """Return a patcher that makes groq_classify return (score, reasoning)."""
        return patch("app.groq_classify", return_value=(score, reasoning))

    # ------------------------------------------------------------------
    # Smoke test: POST /submit with AI-like text
    # ------------------------------------------------------------------
    def test_submit_clearly_ai(self):
        with self._mock_groq(0.92):
            resp = self.client.post(
                "/submit",
                json={
                    "text": (
                        "Artificial intelligence represents a transformative paradigm shift "
                        "in modern society. It is important to note that while the benefits "
                        "of AI are numerous, it is equally essential to consider the ethical "
                        "implications. Furthermore, stakeholders across various sectors must "
                        "collaborate to ensure responsible deployment."
                    ),
                    "creator_id": "test-user-ai",
                },
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("content_id", data)
        self.assertIn("attribution", data)
        self.assertIn("confidence", data)
        self.assertIn("llm_score", data)
        self.assertIn("stylo_score", data)
        self.assertIn("label", data)
        self.assertEqual(data["status"], "classified")
        self.assertEqual(data["attribution"], "likely_ai")
        self.assertGreaterEqual(data["confidence"], 0.75)
        self.assertIn("AI-Generated", data["label"])
        print(f"\n[AI text] confidence={data['confidence']}, attribution={data['attribution']}")
        print(f"  label snippet: {data['label'][:60]}...")
        return data["content_id"]

    # ------------------------------------------------------------------
    # POST /submit with clearly human text
    # ------------------------------------------------------------------
    def test_submit_clearly_human(self):
        with self._mock_groq(0.08):
            resp = self.client.post(
                "/submit",
                json={
                    "text": (
                        "ok so i finally tried that new ramen place downtown and honestly? "
                        "underwhelming. the broth was fine but they put WAY too much sodium "
                        "in it and i was thirsty for like three hours after. my friend got "
                        "the spicy version and said it was better. probably wont go back "
                        "unless someone drags me there"
                    ),
                    "creator_id": "test-user-human",
                },
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["attribution"], "likely_human")
        self.assertLess(data["confidence"], 0.45)
        self.assertIn("Human-Written", data["label"])
        print(f"\n[Human text] confidence={data['confidence']}, attribution={data['attribution']}")
        print(f"  label snippet: {data['label'][:60]}...")

    # ------------------------------------------------------------------
    # POST /submit with borderline academic text
    # ------------------------------------------------------------------
    def test_submit_uncertain(self):
        # Academic formal text: LLM rates it 0.62 (uncertain), stylo is low (no AI phrases)
        # combined = 0.55*0.62 + 0.45*0.33 = 0.341 + 0.149 = 0.49 → uncertain
        with self._mock_groq(0.62):
            resp = self.client.post(
                "/submit",
                json={
                    "text": (
                        "The relationship between monetary policy and asset price inflation "
                        "has been extensively studied in the literature. Central banks face "
                        "a fundamental tension between their mandate for price stability and "
                        "the unintended consequences of prolonged low interest rates on "
                        "equity and real estate valuations."
                    ),
                    "creator_id": "test-user-formal",
                },
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["attribution"], "uncertain")
        self.assertIn("Uncertain", data["label"])
        print(f"\n[Formal text] confidence={data['confidence']}, attribution={data['attribution']}")
        print(f"  label snippet: {data['label'][:60]}...")

    # ------------------------------------------------------------------
    # POST /appeal
    # ------------------------------------------------------------------
    def test_appeal_workflow(self):
        # First submit a piece
        with self._mock_groq(0.90):
            submit_resp = self.client.post(
                "/submit",
                json={
                    "text": (
                        "Artificial intelligence represents a transformative paradigm shift "
                        "in modern society. It is important to note that while the benefits "
                        "of AI are numerous, it is equally essential to consider the ethical "
                        "implications. Furthermore, stakeholders across various sectors must "
                        "collaborate to ensure responsible deployment."
                    ),
                    "creator_id": "creator-appealing",
                },
            )
        content_id = submit_resp.get_json()["content_id"]

        # Submit appeal
        appeal_resp = self.client.post(
            "/appeal",
            json={
                "content_id": content_id,
                "creator_reasoning": (
                    "I wrote this myself. I am an academic and tend to write in formal language. "
                    "This is my own work and it has not been AI-generated."
                ),
            },
        )
        self.assertEqual(appeal_resp.status_code, 200)
        appeal_data = appeal_resp.get_json()
        self.assertEqual(appeal_data["status"], "under_review")
        self.assertEqual(appeal_data["content_id"], content_id)
        print(f"\n[Appeal] content_id={content_id}, status={appeal_data['status']}")

    # ------------------------------------------------------------------
    # GET /log
    # ------------------------------------------------------------------
    def test_get_log(self):
        # Submit a couple of entries first
        with self._mock_groq(0.88):
            self.client.post(
                "/submit",
                json={"text": "It is important to note that various stakeholders must collaborate. Furthermore, this represents a paradigm shift.", "creator_id": "log-test-1"},
            )
        with self._mock_groq(0.12):
            self.client.post(
                "/submit",
                json={"text": "honestly I'm so tired today. don't even want to think about it. gonna just chill and watch something.", "creator_id": "log-test-2"},
            )

        log_resp = self.client.get("/log")
        self.assertEqual(log_resp.status_code, 200)
        log_data = log_resp.get_json()
        self.assertIn("count", log_data)
        self.assertIn("entries", log_data)
        self.assertGreater(log_data["count"], 0)

        first = log_data["entries"][0]
        self.assertIn("content_id", first)
        self.assertIn("timestamp", first)
        self.assertIn("attribution", first)
        self.assertIn("confidence", first)
        self.assertIn("llm_score", first)
        self.assertIn("stylo_score", first)
        self.assertIn("status", first)
        print(f"\n[Log] {log_data['count']} entries. First: {json.dumps(first, indent=2)[:200]}...")

    # ------------------------------------------------------------------
    # Validation: missing fields
    # ------------------------------------------------------------------
    def test_submit_missing_text(self):
        resp = self.client.post("/submit", json={"creator_id": "x"})
        self.assertEqual(resp.status_code, 400)

    def test_submit_missing_creator_id(self):
        resp = self.client.post("/submit", json={"text": "some text here longer than 10 chars"})
        self.assertEqual(resp.status_code, 400)

    def test_appeal_not_found(self):
        resp = self.client.post(
            "/appeal",
            json={"content_id": "nonexistent-id", "creator_reasoning": "I wrote this myself okay"},
        )
        self.assertEqual(resp.status_code, 404)

    # ------------------------------------------------------------------
    # Label variant coverage
    # ------------------------------------------------------------------
    def test_all_three_label_variants_reachable(self):
        from scoring import generate_label

        ai_label = generate_label(0.85, "likely_ai")
        uncertain_label = generate_label(0.60, "uncertain")
        human_label = generate_label(0.20, "likely_human")

        self.assertIn("AI-Generated Content Detected", ai_label)
        self.assertIn("Origin Uncertain", uncertain_label)
        self.assertIn("Likely Human-Written", human_label)

        # Confidence percentages should be in the label
        self.assertIn("85%", ai_label)
        self.assertIn("60%", uncertain_label)
        self.assertIn("80%", human_label)  # human_pct = 1 - 0.20 = 80%

        print("\n[Labels] All three variants confirmed:")
        print("  AI label:", ai_label[:80])
        print("  Uncertain label:", uncertain_label[:80])
        print("  Human label:", human_label[:80])


if __name__ == "__main__":
    unittest.main(verbosity=2)
