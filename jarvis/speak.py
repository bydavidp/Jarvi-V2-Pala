"""Prueba de TTS: sintetiza y reproduce un texto.

Uso:
    python -m jarvis.speak "Hola David, soy Jarvis"
"""

import asyncio
import sys

from audio.tts import TtsEngine
from core.config import Config


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python -m jarvis.speak <texto>")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    config = Config()

    async def run() -> None:
        engine = TtsEngine(config.piper)
        await engine.speak(text)
        await asyncio.sleep(1)

    asyncio.run(run())


if __name__ == "__main__":
    main()
