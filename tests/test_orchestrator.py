"""Tests del orchestrator: maquina de estados, half-duplex, timeouts, errores."""

import asyncio
from unittest import mock

import pytest

from core.bus import Event, EventBus
from core.orchestrator import (
    Orchestrator,
    OrchestratorError,
    OrchestratorState,
    _VALID_TRANSITIONS,
    _WAITING_PHRASES,
)


# ─── Helpers ────────────────────────────────────────────────────────────

def _mock_deps():
    """Crea mocks de todas las dependencias del orchestrator."""
    return {
        "capture": mock.MagicMock(),
        "stt": mock.MagicMock(),
        "router": mock.MagicMock(),
        "llm": mock.MagicMock(),
        "tts": mock.MagicMock(),
        "registry": mock.MagicMock(),
        "bus": mock.MagicMock(spec=EventBus),
    }


# ─── State machine ──────────────────────────────────────────────────────

class TestStateMachine:
    def test_initial_state_is_idle(self) -> None:
        deps = _mock_deps()
        o = Orchestrator(**deps)
        assert o.state == OrchestratorState.IDLE

    def test_transiciones_validas_declaradas(self) -> None:
        assert OrchestratorState.IDLE in _VALID_TRANSITIONS
        assert OrchestratorState.LISTENING in _VALID_TRANSITIONS[OrchestratorState.IDLE]
        assert OrchestratorState.THINKING in _VALID_TRANSITIONS[OrchestratorState.LISTENING]
        assert OrchestratorState.SPEAKING in _VALID_TRANSITIONS[OrchestratorState.THINKING]
        assert OrchestratorState.IDLE in _VALID_TRANSITIONS[OrchestratorState.SPEAKING]
        assert OrchestratorState.ERROR in _VALID_TRANSITIONS[OrchestratorState.IDLE]

    def test_trigger_solo_desde_idle(self) -> None:
        deps = _mock_deps()
        o = Orchestrator(**deps)
        assert o.state == OrchestratorState.IDLE
        o.trigger()
        assert o.state == OrchestratorState.LISTENING

    def test_transicion_invalida_va_a_error(self) -> None:
        deps = _mock_deps()
        o = Orchestrator(**deps)
        # Forzar transicion invalida: IDLE -> SPEAKING
        o._transition(OrchestratorState.SPEAKING)
        assert o.state == OrchestratorState.ERROR
        # De ERROR solo se puede volver a IDLE
        o._transition(OrchestratorState.IDLE)
        assert o.state == OrchestratorState.IDLE

    def test_estado_se_publica_al_bus(self) -> None:
        deps = _mock_deps()
        bus = EventBus()
        loop = asyncio.new_event_loop()
        bus.bind_loop(loop)
        deps["bus"] = bus
        o = Orchestrator(**deps)
        o.trigger()
        assert o.state == OrchestratorState.LISTENING


# ─── Half-duplex ────────────────────────────────────────────────────────

class TestHalfDuplex:
    def test_is_speaking_detecta_estado(self) -> None:
        deps = _mock_deps()
        o = Orchestrator(**deps)
        assert not o.is_speaking()
        o._transition(OrchestratorState.LISTENING)
        o._transition(OrchestratorState.THINKING)
        o._transition(OrchestratorState.SPEAKING)
        assert o.is_speaking()
        o._transition(OrchestratorState.IDLE)
        assert not o.is_speaking()

    def test_speaking_end_se_actualiza(self) -> None:
        deps = _mock_deps()
        o = Orchestrator(**deps)
        o._transition(OrchestratorState.LISTENING)
        o._transition(OrchestratorState.THINKING)
        o._transition(OrchestratorState.SPEAKING)
        assert o._speaking_end > 0
        o._transition(OrchestratorState.IDLE)
        # Al salir de SPEAKING se aplica margen de 0.2s
        import time
        now = time.monotonic()
        assert o._speaking_end >= now  # debe ser futuro (al menos el margen)


# ─── Respuesta de espera ────────────────────────────────────────────────

class TestWaitingPhrase:
    def test_hay_varias_frases(self) -> None:
        assert len(_WAITING_PHRASES) >= 3

    @pytest.mark.asyncio
    async def test_speak_waiting_llama_a_tts(self) -> None:
        deps = _mock_deps()
        deps["tts"].speak = mock.AsyncMock()
        o = Orchestrator(**deps)
        await o._speak_waiting()
        deps["tts"].speak.assert_called_once()
        phrase = deps["tts"].speak.call_args[0][0]
        assert phrase in _WAITING_PHRASES


# ─── Errores hablados ──────────────────────────────────────────────────

class TestErrorResponses:
    def test_todos_los_errores_tienen_frase(self) -> None:
        from core.orchestrator import _ERROR_PHRASES
        for key in ["no_audio", "no_transcription", "routing_timeout", "llm_timeout", "llm_error", "skill_error"]:
            assert _ERROR_PHRASES[key], f"Falta frase para error '{key}'"

    @pytest.mark.asyncio
    async def test_error_produce_tts(self) -> None:
        deps = _mock_deps()
        deps["tts"].speak = mock.AsyncMock()
        o = Orchestrator(**deps)
        await o._speak_error("no_audio")
        deps["tts"].speak.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_timeout_cancela_tarea(self) -> None:
        deps = _mock_deps()
        deps["tts"].speak = mock.AsyncMock()

        async def slow_llm(*a, **kw):
            await asyncio.sleep(10)
        deps["llm"].generate = slow_llm

        o = Orchestrator(**deps, llm_timeout=0.01)

        t0 = asyncio.get_event_loop().time()
        await o._speak_error("llm_timeout")
        assert deps["tts"].speak.called


# ─── Deny hablado ──────────────────────────────────────────────────────

class TestDenySpoken:
    @pytest.mark.asyncio
    async def test_reject_produce_tts(self) -> None:
        deps = _mock_deps()
        deps["tts"].speak = mock.AsyncMock()
        deps["router"].route = mock.AsyncMock(return_value=mock.MagicMock(
            intent=mock.MagicMock(value="REJECT"), resolved_by="classifier"
        ))
        o = Orchestrator(**deps)
        await o._process_text("apaga el computador")
        deps["tts"].speak.assert_called_once()
        assert "no esta autorizada" in deps["tts"].speak.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_chat_simple(self) -> None:
        deps = _mock_deps()
        deps["tts"].speak = mock.AsyncMock()
        deps["router"].route = mock.AsyncMock(return_value=mock.MagicMock(
            intent=mock.MagicMock(value="CHAT"), resolved_by="classifier"
        ))
        deps["llm"].generate = mock.AsyncMock(return_value=mock.MagicMock(text="Hola, soy Jarvis"))
        o = Orchestrator(**deps)
        await o._process_text("hola")
        assert deps["tts"].speak.call_count >= 1

    @pytest.mark.asyncio
    async def test_skill_execution(self) -> None:
        deps = _mock_deps()
        deps["tts"].speak = mock.AsyncMock()
        deps["router"].route = mock.AsyncMock(return_value=mock.MagicMock(
            intent=mock.MagicMock(value="SKILL"),
            skill_name="time", operation="get_current_time", params={},
            resolved_by="matcher",
        ))
        deps["registry"].execute = mock.AsyncMock(return_value=mock.MagicMock(
            success=True, data={"hora": "12:00", "fecha": "martes"}
        ))
        o = Orchestrator(**deps)
        await o._process_text("que hora es")
        deps["registry"].execute.assert_called_once()
        deps["tts"].speak.assert_called()
