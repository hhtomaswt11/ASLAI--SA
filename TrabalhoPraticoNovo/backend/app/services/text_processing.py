import logging
import re
import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)


PHI3_MODEL = "phi3:latest"
PHI3_URL = "http://localhost:11434/api/generate"

COMMON_REWRITE_RULES = {
    "i want water": "I want some water, please.",
    "love you": "I love you.",
    "hello how are you": "Hello, how are you?",
    "good morning": "Good morning.",
    "good afternoon": "Good afternoon.",
    "good night": "Good night.",
    "where toilet": "Where is the toilet, please?",
    "thank you": "Thank you very much.",
    "help me": "Can you help me, please?",
    "i hungry": "I am hungry.",
    "i thirsty": "I am thirsty.",
    "nice meet you": "Nice to meet you.",
    "sorry": "I am sorry.",
    "what your name": "What is your name?",
    "i fine": "I am fine, thank you.",
    "please help": "Please, can you help me?",
}

def clean_phrase_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def basic_spell_correct_phrase(raw_phrase: str) -> str:
    """
    Corrects a phrase using local rules or an LLM (Ollama).
    ASL often uses 'gloss' (word-for-word) which needs reconstruction.
    """
    phrase = clean_phrase_spacing(raw_phrase)
    if not phrase:
        return ""
    
    settings = get_settings()
    lowered = phrase.lower()
    
    # 1. Quick check for hardcoded common rules
    if lowered in COMMON_REWRITE_RULES:
        return COMMON_REWRITE_RULES[lowered]
    
    # 2. If LLM is disabled, just return as is (or with basic spacing cleanup)
    if not settings.enable_llm_correction:
        return phrase

    # 3. Use Ollama for advanced reconstruction
    prompt = (
        f"You are a Sign Language to English translator. "
        f"The following is a sequence of words recognized from gestures. "
        f"Rewrite it into a natural, grammatically correct English sentence. "
        f"Only return the corrected sentence, nothing else.\n\n"
        f"Words: {phrase}\n"
        f"Sentence:"
    )
    
    try:
        response = requests.post(
            PHI3_URL,
            json={
                "model": PHI3_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 50,
                }
            },
            timeout=5
        )
        response.raise_for_status()
        corrected = response.json().get("response", "").strip()
        
        # Clean up any quotes the LLM might have added
        corrected = re.sub(r'^["\']|["\']$', '', corrected)
        
        return corrected or phrase
    except Exception as e:
        logger.warning(f"Ollama correction failed: {e}. Falling back to original phrase.")
        return phrase
