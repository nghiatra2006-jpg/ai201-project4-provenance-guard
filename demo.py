"""
Demo script — runs all 3 showcase tests against the live server.
Usage: python demo.py
"""

import json
import urllib.request

BASE = "http://localhost:5001"


def post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read())


def show(title, result):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)
    print(json.dumps(result, indent=2, ensure_ascii=False))


# ── Test 1: Clearly AI ──────────────────────────────────────────
r1 = post("/submit", {
    "text": (
        "Artificial intelligence represents a transformative paradigm shift "
        "in modern society. It is important to note that while the benefits "
        "of AI are numerous, it is equally essential to consider the ethical "
        "implications. Furthermore, stakeholders across various sectors must "
        "collaborate to ensure responsible deployment."
    ),
    "creator_id": "demo-user-ai",
})
show("TEST 1 — Clearly AI text", r1)
print(f"\n  ➜ attribution : {r1['attribution']}")
print(f"  ➜ confidence  : {r1['confidence']}")
print(f"  ➜ label       : {r1['label'][:60]}...")

# ── Test 2: Clearly Human ───────────────────────────────────────
r2 = post("/submit", {
    "text": (
        "ok so i finally tried that new ramen place downtown and honestly? "
        "underwhelming. the broth was fine but they put WAY too much sodium "
        "in it and i was thirsty for like three hours after. my friend got "
        "the spicy version and said it was better. probably wont go back "
        "unless someone drags me there"
    ),
    "creator_id": "demo-user-human",
})
show("TEST 2 — Clearly Human text", r2)
print(f"\n  ➜ attribution : {r2['attribution']}")
print(f"  ➜ confidence  : {r2['confidence']}")
print(f"  ➜ label       : {r2['label'][:60]}...")

# ── Test 3: Uncertain + Appeal ──────────────────────────────────
r3 = post("/submit", {
    "text": (
        "The relationship between monetary policy and asset price inflation "
        "has been extensively studied in the literature. Central banks face "
        "a fundamental tension between their mandate for price stability and "
        "the unintended consequences of prolonged low interest rates on "
        "equity and real estate valuations."
    ),
    "creator_id": "demo-user-academic",
})
show("TEST 3 — Uncertain (formal academic) text", r3)
print(f"\n  ➜ attribution : {r3['attribution']}")
print(f"  ➜ confidence  : {r3['confidence']}")
print(f"  ➜ label       : {r3['label'][:60]}...")

content_id = r3["content_id"]

# Appeal it
r4 = post("/appeal", {
    "content_id": content_id,
    "creator_reasoning": (
        "I wrote this myself. I am an economist and this is my normal "
        "writing style — formal and citation-heavy. I am not a native "
        "English speaker and tend to avoid contractions."
    ),
})
show("TEST 3b — Appeal submitted", r4)
print(f"\n  ➜ status : {r4['status']}")

# ── Final log ───────────────────────────────────────────────────
log = get("/log")
show(f"AUDIT LOG — {log['count']} entries", log)
