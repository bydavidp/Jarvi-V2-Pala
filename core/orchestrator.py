"""Máquina de estados del asistente. Coordina STT -> Router -> Skills/TTS."""

import asyncio
import logging
import time
from enum import Enum
from typing import Any

from core.bus import Event, EventBus
from core.state import AssistantState

logger = logging.getLogger("jarvis.orchestrator")


class OrchestratorState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class OrchestratorError(Exception):
    """Error interno del orchestrator."""


# ─── Frases de espera (cuando cae al LLM) ──────────────────────────────

_WAITING_PHRASES = [
    "Dame un momento, estoy pensando...",
    "Dejame ver eso, un segundo...",
    "Estoy consultando, ya te digo...",
    "Un momento, estoy procesando...",
]

_ERROR_PHRASES = {
    "no_audio": "No escuche nada, intenta de nuevo.",
    "no_transcription": "No entendi lo que dijiste. Podrias repetirlo?",
    "routing_timeout": "Me esta tomando mucho decidir. Intenta de nuevo.",
    "llm_timeout": "El modelo esta tardando mucho. Intenta con una pregunta mas corta.",
    "llm_error": "Hubo un error al consultar el modelo. Revisa que Ollama este corriendo.",
    "skill_error": "No pude ejecutar esa accion.",
    "deny": None,  # se rellena con el motivo del policy
}

_VALID_TRANSITIONS = {
    OrchestratorState.IDLE: {OrchestratorState.LISTENING, OrchestratorState.ERROR},
    OrchestratorState.LISTENING: {OrchestratorState.THINKING, OrchestratorState.IDLE, OrchestratorState.ERROR},
    OrchestratorState.THINKING: {OrchestratorState.SPEAKING, OrchestratorState.IDLE, OrchestratorState.ERROR},
    OrchestratorState.SPEAKING: {OrchestratorState.IDLE, OrchestratorState.LISTENING, OrchestratorState.ERROR},
    OrchestratorState.ERROR: {OrchestratorState.IDLE, OrchestratorState.SPEAKING},
}


class Orchestrator:
    def __init__(
        self,
        *,
        capture: Any,
        stt: Any,
        router: Any,
        llm: Any,
        tts: Any,
        registry: Any,
        bus: EventBus | None = None,
        listen_key: str = "space",
        stt_timeout: float = 10.0,
        routing_timeout: float = 15.0,
        llm_timeout: float = 30.0,
    ) -> None:
        self._capture = capture
        self._stt = stt
        self._router = router
        self._llm = llm
        self._tts = tts
        self._registry = registry
        self._bus = bus
        self._listen_key = listen_key
        self._stt_timeout = stt_timeout
        self._routing_timeout = routing_timeout
        self._llm_timeout = llm_timeout
        self._state = OrchestratorState.IDLE
        self._interrupted = False
        self._current_llm_task: asyncio.Task[Any] | None = None
        self._speaking_end = 0.0

    # ─── Public API ────────────────────────────────────────────────────

    @property
    def state(self) -> OrchestratorState:
        return self._state

    async def start(self) -> None:
        logger.info("Orchestrator iniciado. Presiona '%s' para hablar.", self._listen_key)
        self._emit_state()

        # Drenar stdin para evitar triggers espurios del terminal
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
        import time as _time
        _time.sleep(0.1)
        while msvcrt.kbhit():
            msvcrt.getch()

        import threading

        def key_listener() -> None:
            try:
                import msvcrt
                while True:
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b" ":
                            self.trigger()
                        elif key == b"\x1b":
                            self._interrupted = True
                            self._tts.stop()
                    else:
                        _time.sleep(0.05)
            except Exception:
                pass

        listener = threading.Thread(target=key_listener, daemon=True)
        listener.start()

        while True:
            try:
                if self._state == OrchestratorState.IDLE:
                    await asyncio.sleep(0.1)
                elif self._state == OrchestratorState.LISTENING:
                    await self._listen_phase()
                elif self._state == OrchestratorState.THINKING:
                    await self._think_phase()
                elif self._state == OrchestratorState.SPEAKING:
                    now = time.monotonic()
                    if now >= self._speaking_end:
                        self._transition(OrchestratorState.IDLE)
                    else:
                        await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error en loop principal")
                self._transition(OrchestratorState.ERROR)
                await asyncio.sleep(1)
                self._transition(OrchestratorState.IDLE)

    def trigger(self) -> None:
        if self._state == OrchestratorState.IDLE:
            logger.info("Trigger recibido, pasando a LISTENING")
            self._transition(OrchestratorState.LISTENING)

    # ─── Fases ─────────────────────────────────────────────────────────

    async def _listen_phase(self) -> None:
        self._interrupted = False
        logger.info("Escuchando...")

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._capture.capture_phrase, self._capture_timeout()),
                timeout=self._stt_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Timeout de captura STT")
            await self._speak_error("no_audio")
            return
        except Exception:
            logger.exception("Error en captura")
            await self._speak_error("no_audio")
            return

        if not result or not result.get("audio"):
            logger.info("Vacio: sin audio detectado")
            await self._speak_error("no_audio")
            return

        self._transition(OrchestratorState.THINKING)

        try:
            stt_result = await asyncio.wait_for(
                asyncio.to_thread(self._stt.transcribe, result["audio"]),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Timeout de transcripcion")
            await self._speak_error("no_transcription")
            self._transition(OrchestratorState.IDLE)
            return
        except Exception:
            logger.exception("Error en transcripcion")
            await self._speak_error("no_transcription")
            self._transition(OrchestratorState.IDLE)
            return

        text = stt_result.get("text", "").strip()
        if not text:
            logger.info("Transcripcion vacia")
            await self._speak_error("no_transcription")
            self._transition(OrchestratorState.IDLE)
            return

        logger.info("Transcripcion: %s", text)
        await self._process_text(text)

    async def _think_phase(self) -> None:
        pass  # processing happens in _process_text, state transition done there

    async def _process_text(self, text: str) -> None:
        try:
            decision = await asyncio.wait_for(
                self._router.route(text),
                timeout=self._routing_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Timeout de routing")
            await self._speak_error("routing_timeout")
            self._transition(OrchestratorState.IDLE)
            return
        except Exception:
            logger.exception("Error en routing")
            await self._speak_error("routing_timeout")
            self._transition(OrchestratorState.IDLE)
            return

        if decision.intent.value == "REJECT":
            await self._speak("Esa orden no esta autorizada.")
            self._transition(OrchestratorState.IDLE)
            return

        if decision.intent.value == "SKILL":
            await self._execute_skill(decision)
            return

        if decision.intent.value == "SEARCH":
            await self._speak("La busqueda web no esta disponible todavia.")
            self._transition(OrchestratorState.IDLE)
            return

        # CHAT
        need_wait = decision.resolved_by == "llm"
        if need_wait:
            await self._speak_waiting()

        try:
            from brain.prompts import CHAT_PROMPT, SYSTEM_PROMPT

            llm_task = asyncio.create_task(
                self._llm.generate(
                    model=self._llm_model_name(),
                    prompt=CHAT_PROMPT.format(user_text=text),
                    system=SYSTEM_PROMPT,
                    max_tokens=200,
                    num_ctx=4096,
                )
            )
            self._current_llm_task = llm_task

            response = await asyncio.wait_for(llm_task, timeout=self._llm_timeout)
            await self._speak(response.text)

        except asyncio.TimeoutError:
            logger.warning("Timeout del LLM, cancelando")
            if self._current_llm_task:
                self._current_llm_task.cancel()
                self._current_llm_task = None
            await self._speak_error("llm_timeout")
            self._transition(OrchestratorState.IDLE)
            return
        except Exception:
            logger.exception("Error en LLM")
            await self._speak_error("llm_error")
            self._transition(OrchestratorState.IDLE)
            return

        self._transition(OrchestratorState.IDLE)

    async def _execute_skill(self, decision: Any) -> None:
        try:
            result = await asyncio.wait_for(
                self._registry.execute(decision.skill_name, decision.operation, decision.params),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            await self._speak_error("skill_error")
            self._transition(OrchestratorState.IDLE)
            return
        except Exception:
            logger.exception("Error ejecutando skill")
            await self._speak_error("skill_error")
            self._transition(OrchestratorState.IDLE)
            return

        if result.success:
            data = result.data
            if "hora" in data:
                await self._speak(f"Son las {data['hora']}. {data['fecha']}.")
            elif result.data:
                msg = str(list(result.data.values())[0]) if len(result.data) == 1 else str(result.data)
                await self._speak(f"Listo. {msg}")
            else:
                await self._speak("Listo.")
        elif result.needs_confirmation:
            await self._speak("Esa accion requiere confirmacion por PIN. No disponible por voz todavia.")
        else:
            await self._speak(f"No pude: {result.error}")

        self._transition(OrchestratorState.IDLE)

    # ─── Voz ───────────────────────────────────────────────────────────

    async def _speak(self, text: str) -> None:
        self._tts.stop()
        self._transition(OrchestratorState.SPEAKING)
        await self._tts.speak(text)
        await asyncio.sleep(0.5)

    async def _speak_waiting(self) -> None:
        import random
        phrase = random.choice(_WAITING_PHRASES)
        await self._speak(phrase)

    async def _speak_error(self, key: str) -> None:
        phrase = _ERROR_PHRASES.get(key, _ERROR_PHRASES["no_audio"])
        # Garantizar que siempre hay ruta valida: ERROR -> SPEAKING -> IDLE
        self._transition(OrchestratorState.ERROR)
        await self._speak(phrase)
        self._transition(OrchestratorState.IDLE)

    # ─── Estado ────────────────────────────────────────────────────────

    def _transition(self, to: OrchestratorState) -> None:
        if self._state == to:
            return  # mismo estado, sin accion
        valid = _VALID_TRANSITIONS.get(self._state, set())
        if to not in valid and self._state != OrchestratorState.ERROR:
            logger.error(
                "Transicion invalida: %s -> %s. Permitidas: %s",
                self._state.value, to.value, [s.value for s in valid],
            )
            self._transition(OrchestratorState.ERROR)
            return

        old = self._state
        self._state = to

        if to == OrchestratorState.SPEAKING:
            self._speaking_end = time.monotonic() + 100.0
        elif old == OrchestratorState.SPEAKING:
            self._speaking_end = time.monotonic() + 0.2

        logger.info("Estado: %s -> %s", old.value, to.value)
        self._emit_state()

    def _emit_state(self) -> None:
        if self._bus is not None:
            try:
                self._bus.publish_threadsafe(Event(type="STATE_CHANGE", data={"state": self._state.value}))
            except Exception:
                pass

    def _capture_timeout(self) -> float:
        return self._stt_timeout

    def _llm_model_name(self) -> str:
        return "llama3.2:latest"

    def is_speaking(self) -> bool:
        return self._state == OrchestratorState.SPEAKING
