# Provenance Guard

A backend classification service for creative writing platforms. Accepts submitted text, runs it through a two-signal AI-origin detection pipeline, returns a confidence score and a plain-language transparency label, and handles creator appeals for contested classifications.

---

## Architecture Overview

The path a submitted piece of text takes from input to transparency label:

```
POST /submit
     │
     ▼
[Input Validation]          — checks for required fields, minimum length
     │
     ├───────────────────────────────────────┐
     ▼                                       ▼
[Signal 1: Groq LLM]              [Signal 2: Stylometric Heuristics]
  llama-3.3-70b-versatile            sentence length variance
  → llm_score (0.0–1.0)             AI phrase density
    semantic + stylistic              informality / contraction rate
    holistic assessment             → stylo_score (0.0–1.0)
                                      structural statistical analysis
     │                                       │
     └──────────────┬────────────────────────┘
                    ▼
     [Confidence Scoring]
       confidence = 0.55 * llm_score + 0.45 * stylo_score
       ≥ 0.75 → likely_ai   (High-confidence AI label)
       0.45–0.74 → uncertain (Uncertain label)
       < 0.45 → likely_human (High-confidence human label)
                    │
                    ▼
     [Transparency Label Generator]
       maps score + tier → plain-language label text
                    │
                    ▼
     [Audit Logger]
       writes structured JSON entry to audit_log.json
                    │
                    ▼
     [HTTP Response]
       content_id, attribution, confidence,
       signal scores, label text, status
```

**Appeal flow:**

```
POST /appeal
     │
     ▼
[Lookup content_id in audit log]
     │
     ▼
[Update status → "under_review"]
  add appeal_reasoning + appeal_timestamp
     │
     ▼
[Return confirmation]
```

---

## Detection Signals

### Signal 1: LLM-Based Classification (Groq)

**What it measures:** Semantic and stylistic coherence as a holistic gestalt — whether text "sounds like" AI output based on vocabulary, sentence flow, structural patterns, and the presence of LLM-characteristic phrasing.

**Why AI text differs here:** Large language models produce text with smooth, predictable transitions; balanced paragraph structures; hedged-but-authoritative language ("it is important to note..."); and an absence of genuine personal quirks, tangents, or contradictions. Human writing is messier, more idiosyncratic, and carries a personal voice that no prompt can fully specify.

**What it misses:** A human imitating AI style (e.g., writing very formally for a professional context) will score high. AI text with intentional informality injected will score lower. The model is also biased toward flagging non-native English writing as AI-generated, since it is often more grammatically formal. Very short texts (<50 words) provide insufficient signal.

**Output:** A float 0.0–1.0. Score of 1.0 means "highly confident this is AI-generated."

**Implementation:** `signals/llm_signal.py` — sends text to `llama-3.3-70b-versatile` with a structured system prompt requesting a JSON response with `{"score": float, "reasoning": string}`. Temperature is set to 0.1 for consistency.

---

### Signal 2: Stylometric Heuristics (Pure Python)

**What it measures:** Three statistical properties that empirically differ between AI-generated and human-written text:

1. **Sentence Length Variance (SLV):** AI text has very uniform sentence lengths. Human writing varies far more — short sentences next to long ones, fragments, run-ons. High variance → more human-like → lower AI score. Calibrated with a sigmoid at variance=18.

2. **AI Phrase Density (APD):** AI text reliably overuses a recognizable vocabulary: "it is important to note", "furthermore", "transformative", "stakeholders", "paradigm shift", "delve", "holistic", "robust", "seamlessly". The signal counts matches per 100 words using a lexicon of ~40 phrases. High density → high AI score.

3. **Informality Rate (INF):** Human casual writing uses contractions (I'm, don't, won't, that's), fillers (like, honestly, kinda, yeah, ok), and informal vocabulary. AI text in professional registers avoids these almost entirely. High informality → human-like → lower AI score.

**Combination:** `stylo_score = 0.35 * SLV + 0.40 * APD + 0.25 * INF`  
APD carries the most weight because the AI phrase lexicon is the most discriminating signal at paragraph length.

**What it misses:** Formal human academic writing will score higher than expected — it has uniform sentence structure and avoids contractions. Short poems with deliberate repetition will look statistically AI-like. Non-English or code-heavy text breaks the tokenization assumptions.

**Output:** A float 0.0–1.0. Score of 1.0 means "text has statistical properties strongly resembling AI output."

**Implementation:** `signals/stylometric_signal.py`

---

## Confidence Scoring

### Combination Formula

```
confidence = 0.55 × llm_score + 0.45 × stylo_score
```

The LLM signal receives a slightly higher weight (0.55) because it captures semantic and contextual patterns that stylometrics cannot — e.g., GPT-characteristic phrasing patterns that don't appear in the phrase lexicon. Stylometrics (0.45) provides a structural cross-check that is independent of the LLM's judgments.

### Thresholds

| Score Range | Attribution | Label Variant |
|---|---|---|
| ≥ 0.75 | `likely_ai` | High-confidence AI |
| 0.45 – 0.74 | `uncertain` | Uncertain |
| < 0.45 | `likely_human` | High-confidence human |

**Design rationale:** The threshold for `likely_ai` is set at 0.75, not 0.5. A false positive — labeling a human writer's work as AI-generated — is worse than a false negative on a writing platform. This asymmetry is reflected in the design: the system requires stronger evidence before making an accusatory claim. The "uncertain" band (0.45–0.74) is deliberately wide, catching edge cases where confidence is insufficient to render a verdict.

**What 0.5 means:** Neither signal is reading clearly. The system genuinely does not know. It will return an "Uncertain" label and present an appeal path. A 0.5 is not a verdict.

### Validation: Example Submissions

| Input | LLM score | Stylo score | Combined | Attribution |
|---|---|---|---|---|
| "Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits of AI are numerous, it is equally essential to consider the ethical implications. Furthermore, stakeholders across various sectors must collaborate to ensure responsible deployment." | 0.92 | 0.73 | 0.83 | `likely_ai` |
| "ok so i finally tried that new ramen place downtown and honestly? underwhelming. the broth was fine but they put WAY too much sodium in it and i was thirsty for like three hours after. my friend got the spicy version and said it was better. probably wont go back unless someone drags me there" | 0.08 | 0.15 | 0.11 | `likely_human` |
| "The relationship between monetary policy and asset price inflation has been extensively studied in the literature. Central banks face a fundamental tension between their mandate for price stability and the unintended consequences of prolonged low interest rates on equity and real estate valuations." | 0.62 | 0.33 | 0.49 | `uncertain` |
| "I've been thinking a lot about remote work lately. There are genuine tradeoffs — flexibility and no commute on one side, isolation and blurred work-life boundaries on the other. Studies show productivity varies widely by individual and role type." | 0.65 | 0.22 | 0.46 | `uncertain` |

The spread between clearly AI (0.83) and clearly human (0.11) is 0.72 — the scoring produces meaningful variation, not a constant.

---

## Transparency Labels

Three label variants based on confidence tier. The exact text displayed to readers:

### High-Confidence AI (`confidence ≥ 0.75`)

```
⚠️ AI-Generated Content Detected

Our analysis suggests this content was likely generated by an AI tool, not written by a human author.

Confidence: High ([SCORE]%)

This label is shown to help readers understand the origin of this content. If you are the creator and believe this is incorrect, you may submit an appeal below.
```

### Uncertain (`0.45 ≤ confidence < 0.75`)

```
🔍 Origin Uncertain

Our system was unable to determine with confidence whether this content was written by a human or generated by AI.

Confidence in AI origin: Moderate ([SCORE]%)

We don't have enough signal to make a definitive determination. Readers should consider this context. Creators who believe this result is inaccurate may submit an appeal.
```

### High-Confidence Human (`confidence < 0.45`)

```
✅ Likely Human-Written

Our analysis suggests this content was written by a human author.

Confidence in human origin: High ([SCORE]%)

This content passed our AI-origin detection checks. This label does not constitute a guarantee of authenticity.
```

**Note:** `[SCORE]` is replaced with the actual percentage at runtime. For the human label, the displayed percentage is `(1 - confidence) × 100` — i.e., confidence in human origin, not AI origin.

---

## Appeals Workflow

Creators who believe their work was misclassified can submit an appeal via `POST /appeal`.

**What's required:**
- `content_id` — the UUID returned by `/submit`
- `creator_reasoning` — their explanation (min 10 characters, free text)

**What happens:**
1. The system looks up the existing audit log entry for the `content_id`
2. Status is updated from `"classified"` to `"under_review"`
3. `appeal_reasoning` and `appeal_timestamp` fields are added to the log entry
4. The response confirms the appeal was received

**What a human reviewer sees when they query `GET /log`:**
```json
{
  "content_id": "...",
  "attribution": "likely_ai",
  "confidence": 0.83,
  "llm_score": 0.92,
  "stylo_score": 0.73,
  "status": "under_review",
  "appeal_reasoning": "I wrote this myself. I am an academic and tend to write in formal language.",
  "appeal_timestamp": "2026-06-30T10:00:00+00:00"
}
```

Automated re-classification is intentionally not implemented. Automated re-classification based on a creator's text input creates a bypass vulnerability — anyone could claim human authorship and trigger a score change.

**Example appeal curl:**
```bash
curl -s -X POST http://localhost:5000/appeal \
  -H "Content-Type: application/json" \
  -d '{
    "content_id": "PASTE-CONTENT-ID-HERE",
    "creator_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical."
  }' | python -m json.tool
```

---

## Rate Limiting

**Limits:** `10 requests per minute; 100 requests per day` per IP address on `POST /submit`.

**Reasoning:**

- A real writer submitting their own creative work might submit a handful of pieces in a session — a poem, a short story excerpt, a blog post. 10 per minute is generous for any legitimate workflow.
- 100 per day allows a prolific creator to submit an entire portfolio but prevents bulk automated use.
- The key threat model is adversarial probing: if there's no rate limit, a bad actor can run thousands of texts through the classifier to reverse-engineer its thresholds and craft bypass text. 100/day makes this economically impractical.
- The appeal endpoint is not rate-limited at the same level — appeals require a valid `content_id` from a previous submission, which naturally constrains the volume.

**Verified rate-limit behavior** (13 rapid requests, limit = 10/minute):

```
Request  1: HTTP 200
Request  2: HTTP 200
Request  3: HTTP 200
Request  4: HTTP 200
Request  5: HTTP 200
Request  6: HTTP 200
Request  7: HTTP 200
Request  8: HTTP 200
Request  9: HTTP 200
Request 10: HTTP 200
Request 11: HTTP 429
Request 12: HTTP 429
Request 13: HTTP 429
```

**Flask-Limiter setup:**
```python
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute; 100 per day")
def submit():
    ...
```

---

## Audit Log

Every attribution decision and appeal is captured in `audit_log.json` as a structured JSON array.

**Log entry schema:**
```json
{
  "content_id": "uuid",
  "creator_id": "string",
  "timestamp": "ISO8601",
  "text_preview": "first 120 chars of submitted text",
  "attribution": "likely_ai | uncertain | likely_human",
  "confidence": 0.83,
  "llm_score": 0.92,
  "stylo_score": 0.73,
  "stylo_breakdown": {
    "sentence_length_variance_score": 0.29,
    "ai_phrase_density_score": 0.94,
    "informality_score": 1.0
  },
  "llm_reasoning": "one-sentence explanation from the LLM",
  "label": "full label text shown to readers",
  "status": "classified | under_review",
  "appeal_reasoning": null,
  "appeal_timestamp": null
}
```

**Sample `GET /log` output** (3 entries, newest first):

```json
{
  "count": 3,
  "entries": [
    {
      "content_id": "a1b2c3d4-...",
      "creator_id": "creator-appealing",
      "timestamp": "2026-06-30T06:41:12.000Z",
      "text_preview": "Artificial intelligence represents a transformative paradigm shift...",
      "attribution": "likely_ai",
      "confidence": 0.8331,
      "llm_score": 0.92,
      "stylo_score": 0.7268,
      "stylo_breakdown": {
        "sentence_length_variance_score": 0.2888,
        "ai_phrase_density_score": 0.9394,
        "informality_score": 1.0
      },
      "llm_reasoning": "The text uses multiple AI-typical connective phrases and is structurally uniform.",
      "label": "⚠️ AI-Generated Content Detected\n\nOur analysis suggests this content was likely generated...",
      "status": "under_review",
      "appeal_reasoning": "I wrote this myself. I am an academic and tend to write in formal language.",
      "appeal_timestamp": "2026-06-30T06:41:13.000Z"
    },
    {
      "content_id": "e5f6g7h8-...",
      "creator_id": "test-user-formal",
      "timestamp": "2026-06-30T06:41:11.000Z",
      "text_preview": "The relationship between monetary policy and asset price inflation...",
      "attribution": "uncertain",
      "confidence": 0.4896,
      "llm_score": 0.62,
      "stylo_score": 0.3303,
      "stylo_breakdown": {
        "sentence_length_variance_score": 0.2293,
        "ai_phrase_density_score": 0.0,
        "informality_score": 1.0
      },
      "llm_reasoning": "Text is formal and structured but lacks the most common AI marker phrases.",
      "label": "🔍 Origin Uncertain\n\nOur system was unable to determine with confidence...",
      "status": "classified",
      "appeal_reasoning": null,
      "appeal_timestamp": null
    },
    {
      "content_id": "i9j0k1l2-...",
      "creator_id": "test-user-human",
      "timestamp": "2026-06-30T06:41:10.000Z",
      "text_preview": "ok so i finally tried that new ramen place downtown and honestly?...",
      "attribution": "likely_human",
      "confidence": 0.1122,
      "llm_score": 0.08,
      "stylo_score": 0.1516,
      "stylo_breakdown": {
        "sentence_length_variance_score": 0.2416,
        "ai_phrase_density_score": 0.0,
        "informality_score": 0.2683
      },
      "llm_reasoning": "Text is casual, uses contractions and personal voice; unlikely to be AI-generated.",
      "label": "✅ Likely Human-Written\n\nOur analysis suggests this content was written by a human author...",
      "status": "classified",
      "appeal_reasoning": null,
      "appeal_timestamp": null
    }
  ]
}
```

---

## API Reference

### `POST /submit`

Accept a text submission for AI-origin classification.

**Request body:**
```json
{
  "text": "string (required, min 10 chars)",
  "creator_id": "string (required)"
}
```

**Response:**
```json
{
  "content_id": "uuid",
  "attribution": "likely_ai | uncertain | likely_human",
  "confidence": 0.83,
  "llm_score": 0.92,
  "stylo_score": 0.73,
  "stylo_breakdown": { ... },
  "llm_reasoning": "...",
  "label": "full label text",
  "status": "classified"
}
```

**Rate limit:** 10 per minute, 100 per day per IP. Returns HTTP 429 when exceeded.

---

### `POST /appeal`

Contest a classification.

**Request body:**
```json
{
  "content_id": "uuid (required)",
  "creator_reasoning": "string (required, min 10 chars)"
}
```

**Response:**
```json
{
  "message": "Appeal received. Your content is now under review.",
  "content_id": "uuid",
  "status": "under_review",
  "original_attribution": "likely_ai",
  "original_confidence": 0.83,
  "appeal_timestamp": "2026-06-30T10:00:00+00:00"
}
```

---

### `GET /log`

Return structured audit log entries, newest first.

**Query params:** `limit` (optional integer)

**Response:**
```json
{
  "count": 5,
  "entries": [ ... ]
}
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/ai201-project4-provenance-guard.git
cd ai201-project4-provenance-guard

python -m venv .venv
source .venv/bin/activate          # Mac/Linux

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GROQ_API_KEY

python app.py
```

**Run tests (no API key needed):**
```bash
GROQ_API_KEY=test-dummy python test_local.py
```

---

## Known Limitations

**1. Formal human academic writing generates false positives.**
The stylometric signal cannot distinguish between AI-generated formality and human academic formality. A professor writing in their normal professional register — careful syntax, no contractions, long structured sentences — will produce a `stylo_score` in the 0.35–0.55 range. If the LLM also reads the text as AI-like (which it may, since formal grammar correlates with AI training data), the combined confidence can reach the uncertain or even `likely_ai` threshold. This is a known tradeoff: the AI phrase lexicon (Signal 2's main discriminator) was built around the most recognizable LLM patterns. Academic writing lacks those specific phrases, but it shares their structural properties.

**2. The system is calibrated for English-language prose only.**
The AI phrase lexicon is English. Stylometric tokenization assumes English word boundaries. Texts in other languages, or texts mixing English with code, equations, or structured data, will produce unreliable scores. Signal 1 (LLM) may still work cross-lingually but with reduced reliability.

**3. Very short texts (< 3 sentences) produce neutral stylometric scores.**
Sentence length variance requires at least two sentences to compute. AI phrase density is meaningful only at sufficient length. For texts under ~50 words, the stylometric signal falls back to a neutral 0.5, and the combined score depends almost entirely on the LLM signal.

---

## Spec Reflection

**One way the spec helped:** Defining the three label variants before writing any code was the most useful constraint in the entire project. Having to write the exact text of "what does a 0.5 score say to a non-technical user?" forced a decision about what uncertainty means — and that decision then dictated the threshold placement (the wide 0.45–0.74 band), which then shaped the signal calibration. The spec wasn't just documentation; it was a design forcing function.

**One way implementation diverged:** The initial Signal 2 design used type-token ratio (TTR) and punctuation density as the main stylometric features. In implementation, both produced very similar values across AI and human text at paragraph length (all samples clustered in 0.45–0.55). The root cause was that TTR and punctuation density don't discriminate well at short text lengths — AI text at 100–200 words looks statistically similar to human text on raw vocabulary diversity metrics. The implementation switched to an AI-phrase lexicon approach (which the spec didn't specify) because it's far more discriminating: a text either contains "it is important to note that" or it doesn't. This divergence improved classification accuracy significantly but required expanding the planning doc to reflect the change.

---

## AI Usage

**Instance 1: Flask app skeleton and signal integration**

I directed the AI to generate the `app.py` Flask skeleton and the `groq_classify()` function in `signals/llm_signal.py`, providing the API contract from `planning.md` and the architecture diagram. The AI produced a reasonable Flask structure but used `@app.before_request` for input validation instead of inline validation inside each route handler. I overrode this because inline validation is clearer to read and easier to customize per-endpoint. I also revised the Groq prompt from a multiple-choice format ("Is this AI-generated? Yes/No/Uncertain") to a free-form JSON response format with a numeric score — the original prompt collapsed the output to a binary and lost the gradient.

**Instance 2: Stylometric signal calibration**

After the initial TTR/punctuation-density stylometric signal produced scores that clustered too tightly (all inputs in 0.45–0.55), I asked the AI to help diagnose why and suggest alternative features. It proposed replacing TTR with a lexicon-based approach keyed on specific AI-characteristic phrases, and replacing punctuation density with a contraction/informality rate. I vetted the lexicon it generated (40 phrases) and removed several entries that would have flagged legitimate academic writing (e.g., "significant" and "important" are common in human academic text and shouldn't be AI markers). The final lexicon contains only the most distinctively AI-generated patterns.

---

## Project Structure

```
ai201-project4-provenance-guard/
├── app.py                  # Flask app, all three endpoints
├── scoring.py              # combine_scores(), get_attribution(), generate_label()
├── audit_log.py            # Structured JSON audit logger
├── signals/
│   ├── __init__.py
│   ├── llm_signal.py       # Signal 1: Groq LLM classifier
│   └── stylometric_signal.py  # Signal 2: Stylometric heuristics
├── test_local.py           # Integration tests (mocked Groq, no API key needed)
├── planning.md             # Architecture spec, written before any code
├── requirements.txt
├── .env.example
└── .gitignore
```
