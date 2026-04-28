from __future__ import annotations

import re

try:
    from textblob import TextBlob
except Exception:  # pragma: no cover
    TextBlob = None

EAT_SENTENCE = "I would like something to eat."

COMMON_REWRITE_RULES = {
    "i want water": "I want some water, please.",
    "i want some water": "I want some water, please.",
    "i want eat": EAT_SENTENCE,
    "i want to eat": EAT_SENTENCE,
    "want eat": EAT_SENTENCE,
    "love you": "I love you.",
}

def clean_phrase_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def basic_spell_correct_phrase(raw_phrase: str) -> str:
    phrase = clean_phrase_spacing(raw_phrase)
    if not phrase:
        return ""
    lowered = phrase.lower()
    if lowered in COMMON_REWRITE_RULES:
        return COMMON_REWRITE_RULES[lowered]
    if TextBlob is None:
        return phrase
    corrected = str(TextBlob(phrase).correct())
    return corrected.strip() or phrase
