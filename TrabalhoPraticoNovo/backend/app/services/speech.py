from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pyttsx3

_executor = ThreadPoolExecutor(max_workers=1)


def _speak(text: str) -> None:
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def speak_async(text: str) -> None:
    _executor.submit(_speak, text)
