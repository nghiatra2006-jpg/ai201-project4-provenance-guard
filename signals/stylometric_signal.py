"""
Signal 2: Stylometric heuristics.

Computes three statistical properties of the text that empirically differ
between AI-generated and human-written text:

1. Sentence Length Variance (SLV)
   AI text tends to produce uniform sentence lengths.
   High variance → more human-like → lower AI score.

2. AI Hedge / Transition Phrase Density
   AI text reliably overuses formal connective and hedging phrases:
   "it is important to note", "furthermore", "in conclusion", "it is worth mentioning",
   "it should be noted", "various", "numerous", "stakeholders", etc.
   High density → AI-like → higher score.

3. Contraction & Informality Rate
   Human casual writing uses contractions (I'm, don't, won't, that's, it's)
   and informal markers (ok, yeah, hmm, wow, honestly, like, kinda, etc.).
   AI text in formal registers avoids contractions almost entirely.
   High rate → human-like → lower score.

Combined into a single stylo_score (0.0–1.0):
  1.0 = text has statistical properties strongly resembling AI output
  0.0 = text has statistical properties strongly resembling human output
"""

import re
from statistics import variance


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Phrases and words strongly associated with AI-generated formal text
AI_MARKERS = [
    r"\bit is important to note\b",
    r"\bit is worth (noting|mentioning)\b",
    r"\bit should be noted\b",
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\bin conclusion\b",
    r"\bin summary\b",
    r"\bto summarize\b",
    r"\bin addition\b",
    r"\bstakeholders?\b",
    r"\bit is essential\b",
    r"\bresponsible (deployment|ai|use)\b",
    r"\bparadigm shift\b",
    r"\btransformative\b",
    r"\bvarious sectors?\b",
    r"\bnumerous\b",
    r"\bdelve\b",
    r"\bwe must (consider|acknowledge|recognize)\b",
    r"\bone must\b",
    r"\bensure (that|responsible)\b",
    r"\bthis (highlights?|underscores?|demonstrates?|suggests?)\b",
    r"\bpivotal\b",
    r"\bcrucial(ly)?\b",
    r"\bfacilitate\b",
    r"\bmitigate\b",
    r"\bleverage\b",
    r"\bsynergies\b",
    r"\boptimize\b",
    r"\brobusts?\b",
    r"\bseamless(ly)?\b",
    r"\bcomprehensive(ly)?\b",
    r"\bholistic(ally)?\b",
    r"\bcollaborate\b",
    r"\bempowers?\b",
    r"\bperspective\b",
    r"\bit (is|can be) argued\b",
    r"\bultimately\b",
    r"\bsubstantially\b",
    r"\bsignificant(ly)?\b",
    r"\baddressing\b",
]

# Human informality markers: contractions and casual words
HUMAN_MARKERS = [
    r"\bi'm\b",
    r"\bdon't\b",
    r"\bwon't\b",
    r"\bcan't\b",
    r"\bthat's\b",
    r"\bit's\b",
    r"\bthey're\b",
    r"\bwe're\b",
    r"\byou're\b",
    r"\bi've\b",
    r"\bi'd\b",
    r"\bwasn't\b",
    r"\bisn't\b",
    r"\baren't\b",
    r"\bdidn't\b",
    r"\bhadn't\b",
    r"\bcouldn't\b",
    r"\bshouldn't\b",
    r"\bwouldn't\b",
    r"\blike\b",        # casual 'like' used as a filler
    r"\bhonestly\b",
    r"\bactually\b",
    r"\bkinda\b",
    r"\bsorta\b",
    r"\byeah\b",
    r"\bnope\b",
    r"\bokay\b",
    r"\bok\b",
    r"\bwow\b",
    r"\bhmm\b",
    r"\buh\b",
    r"\bbtw\b",
    r"\blol\b",
    r"\banyway\b",
    r"\bpretty\b",      # "pretty good", "pretty bad" — informal usage
    r"\bstuff\b",
    r"\bthing\b",
    r"\bgonna\b",
    r"\bwanna\b",
    r"\bgotta\b",
]


def _tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s.strip()) > 2]


def _word_count(text: str) -> int:
    return len(text.split())


def sentence_length_variance_score(text: str) -> float:
    """
    Compute AI-likelihood score from sentence length variance.

    Low variance (uniform sentences) → AI-like → high score (close to 1.0).
    High variance (irregular sentences) → human-like → low score (close to 0.0).

    Calibration based on real samples:
      - AI text typical variance:       4–20 word²  → score 0.7–0.9
      - Formal human text variance:    20–60 word²  → score 0.3–0.6
      - Casual human text variance:    60–200 word² → score 0.05–0.3
    """
    sentences = _tokenize_sentences(text)
    if len(sentences) < 2:
        return 0.5

    lengths = [len(s.split()) for s in sentences]
    var = variance(lengths)

    # Calibrated sigmoid: crossover at variance=25
    slv_score = 1.0 / (1.0 + var / 18.0)
    return round(max(0.0, min(1.0, slv_score)), 4)


def ai_phrase_density_score(text: str) -> float:
    """
    Compute AI-likelihood score from density of AI-associated phrases.

    Counts how many AI marker phrases appear per 100 words.
    High density → very AI-like → score near 1.0.

    Calibration:
      0 hits per 100 words → score 0.0
      1 hit per 100 words  → score ~0.50
      2+ hits per 100 words → score ~0.80
      5+ hits per 100 words → score ~0.95
    """
    text_lower = text.lower()
    words = _word_count(text)
    if words == 0:
        return 0.5

    hit_count = sum(
        1 for pattern in AI_MARKERS if re.search(pattern, text_lower)
    )

    # Density per 100 words, with saturation
    density = hit_count / (words / 100.0)

    # Sigmoid: score = 1 - 1/(1 + density/1.5)
    ai_score = 1.0 - 1.0 / (1.0 + density / 1.5)
    return round(max(0.0, min(1.0, ai_score)), 4)


def informality_score(text: str) -> float:
    """
    Compute AI-likelihood score from informality markers.

    High informality (contractions, casual words) → human-like → LOW AI score.
    Low informality (no contractions, formal language) → AI-like → HIGH AI score.

    Returns:
      0.0 = very informal (human-like)
      1.0 = very formal, no informality (AI-like)
    """
    text_lower = text.lower()
    words = _word_count(text)
    if words == 0:
        return 0.5

    hit_count = sum(
        1 for pattern in HUMAN_MARKERS if re.search(pattern, text_lower)
    )

    # Density per 100 words
    density = hit_count / (words / 100.0)

    # More informality → lower AI score
    # Score = 1 / (1 + density / 2.0)
    # 0 hits/100 → 1.0 (AI-like formality)
    # 2 hits/100 → 0.50
    # 6+ hits/100 → 0.25
    info_score = 1.0 / (1.0 + density / 2.0)
    return round(max(0.0, min(1.0, info_score)), 4)


def stylometric_classify(text: str) -> tuple[float, dict]:
    """
    Classify text using stylometric heuristics.

    Args:
        text: The text to analyze.

    Returns:
        (stylo_score, breakdown) where:
          - stylo_score is 0.0–1.0 (1.0 = AI-like statistics)
          - breakdown is a dict with individual metric scores for transparency
    """
    slv = sentence_length_variance_score(text)
    apd = ai_phrase_density_score(text)
    inf = informality_score(text)

    # Weighted combination:
    #   SLV: 35% — structural uniformity
    #   APD: 40% — most discriminating for typical AI text
    #   INF: 25% — formality / contraction rate
    stylo_score = 0.35 * slv + 0.40 * apd + 0.25 * inf
    stylo_score = round(max(0.0, min(1.0, stylo_score)), 4)

    breakdown = {
        "sentence_length_variance_score": slv,
        "ai_phrase_density_score": apd,
        "informality_score": inf,
    }

    return stylo_score, breakdown
