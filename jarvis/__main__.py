"""Entry point del asistente JARVIS.

Uso:
    python -m jarvis
"""

import asyncio
import sys

from audio.capture import AudioCapture
from audio.stt import SttEngine
from audio.tts import TtsEngine
from brain.llm import OllamaClient
from brain.router import Router
from brain.classifier import IntentClassifier
from core.bus import EventBus
from core.config import Config
from core.logs import setup_logging
from core.orchestrator import Orchestrator
from skills.base import SkillRegistry
from skills import discover_skills
from security.policy import Policy


async def _main_async() -> None:
    setup_logging()
    print("Iniciando JARVIS...")

    config = Config()
    bus = EventBus()
    bus.bind_loop(asyncio.get_running_loop())

    policy = Policy(
        permissions_path="config/permissions.yaml",
        whitelist_apps=config.settings.whitelist.apps,
    )

    capture = AudioCapture(
        device=config.audio.input_device,
        sample_rate=config.audio.sample_rate,
        vad_silence_ms=config.audio.vad_silence_ms,
        vad_min_speech_ms=config.audio.vad_min_speech_ms,
        vad_threshold=config.audio.vad_threshold,
    )

    stt = SttEngine(config.whisper)

    ollama = OllamaClient(
        base_url=config.ollama.url,
        timeout=config.ollama.timeout,
        keep_alive=config.ollama.keep_alive,
    )

    router = Router(
        ollama=ollama,
        model=config.ollama.model,
        classifier=IntentClassifier(threshold=0.60),
    )

    registry = SkillRegistry(policy)
    discover_skills(registry)

    tts = TtsEngine(config=config.piper, bus=bus)

    orch = Orchestrator(
        capture=capture,
        stt=stt,
        router=router,
        llm=ollama,
        tts=tts,
        registry=registry,
        bus=bus,
    )

    print("JARVIS listo. Presiona ESPACIO para hablar, ESC para interrumpir.\n")
    await orch.start()


def main() -> None:
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        print("\nJARVIS detenido.")
        sys.exit(0)


if __name__ == "__main__":
    main()
