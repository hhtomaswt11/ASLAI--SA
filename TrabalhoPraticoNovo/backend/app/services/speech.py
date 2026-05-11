import asyncio
import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import edge_tts

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1)

# Voices: pt-PT-RaquelNeural, pt-PT-DuarteNeural, pt-BR-FranciscaNeural, pt-BR-AntonioNeural
VOICE = "pt-PT-DuarteNeural"

def _find_player() -> str | None:
    """Find a command-line player to play the MP3."""
    for player in ["mpv", "ffplay", "mplayer"]:
        if shutil.which(player):
            return player
    return None

async def _generate_and_play(text: str) -> None:
    try:
        output_file = "speech_output.mp3"
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(output_file)
        
        player = _find_player()
        if player:
            if player == "ffplay":
                cmd = [player, "-nodisp", "-autoexit", output_file]
            else:
                cmd = [player, output_file]
                
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            logger.warning("No MP3 player found (mpv, ffplay, mplayer). Audio generated but not played.")
            
    except Exception as e:
        logger.error(f"Failed to generate or play speech: {e}")

def _speak(text: str) -> None:
    """Wrapper to run async edge-tts in a thread."""
    try:
        asyncio.run(_generate_and_play(text))
    except Exception as e:
        logger.error(f"Error in _speak thread: {e}")

def speak_async(text: str) -> None:
    _executor.submit(_speak, text)
