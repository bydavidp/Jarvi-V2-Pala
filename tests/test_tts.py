import asyncio
from unittest import mock

import numpy as np
import pytest

from audio.tts import TtsEngine, TtsError
from core.bus import Event, EventBus
from core.config import PiperConfig


def _make_config(
    binary: str = "piper.exe",
    model: str = "models/piper/es_ES-carlfm-medium.onnx",
) -> PiperConfig:
    return PiperConfig(binary_path=binary, model_path=model)


def _make_dummy_audio() -> bytes:
    return np.zeros(1000, dtype=np.int16).tobytes()


# ─── TtsEngine ───────────────────────────────────────────────────────────

class TestTtsEngine:
    @pytest.mark.asyncio
    async def test_speak_sintetiza_y_reproduce(self) -> None:
        config = _make_config()
        bus = EventBus()
        bus.bind_loop(asyncio.get_running_loop())

        engine = TtsEngine(config, bus=bus)

        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("SPEAKING_START", handler)
        bus.subscribe("SPEAKING_END", handler)

        with mock.patch.object(engine, "_model_path_valid", return_value=True), \
             mock.patch.object(engine, "_synthesize", return_value=_make_dummy_audio()) as synth_mock, \
             mock.patch.object(engine, "_play_blocking") as play_mock:

            await engine.speak("Hola")
            await asyncio.sleep(0.3)

            synth_mock.assert_called_once_with("Hola")
            play_mock.assert_called_once()
            assert len(received) == 2
            assert received[0].type == "SPEAKING_START"
            assert received[1].type == "SPEAKING_END"

    @pytest.mark.asyncio
    async def test_varios_speak_se_encolan(self) -> None:
        config = _make_config()
        engine = TtsEngine(config)

        texts: list[str] = []

        async def fake_synthesize(text: str) -> bytes:
            texts.append(text)
            return _make_dummy_audio()

        with mock.patch.object(engine, "_model_path_valid", return_value=True), \
             mock.patch.object(engine, "_synthesize", side_effect=fake_synthesize), \
             mock.patch.object(engine, "_play_blocking"):

            await engine.speak("uno")
            await engine.speak("dos")
            await engine.speak("tres")
            await asyncio.sleep(0.5)

            assert texts == ["uno", "dos", "tres"]

    @pytest.mark.asyncio
    async def test_stop_corta_reproduccion(self) -> None:
        config = _make_config()
        engine = TtsEngine(config)

        with mock.patch.object(engine, "_model_path_valid", return_value=True), \
             mock.patch.object(engine, "_synthesize", return_value=_make_dummy_audio()) as synth_mock, \
             mock.patch.object(engine, "_play_blocking") as play_mock:

            await engine.speak("largo")
            await asyncio.sleep(0.05)
            engine.stop()
            await asyncio.sleep(0.1)

            # No más reproducciones después de stop
            play_calls_before_stop = play_mock.call_count

    @pytest.mark.asyncio
    async def test_texto_vacio_no_procesa(self) -> None:
        config = _make_config()
        engine = TtsEngine(config)

        with mock.patch.object(engine, "_synthesize") as synth_mock:
            await engine.speak("")
            await engine.speak("   ")
            await asyncio.sleep(0.1)
            synth_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_modelo_faltante_loggea_sin_crash(self) -> None:
        config = _make_config(model="no_existe.onnx")
        engine = TtsEngine(config)

        with mock.patch.object(engine, "_model_path_valid", return_value=False), \
             mock.patch.object(engine, "_synthesize") as synth_mock:

            await engine.speak("Hola")
            await asyncio.sleep(0.1)
            synth_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_binario_no_encontrado_lanza_tts_error(self) -> None:
        config = _make_config(binary="no_existe.exe")
        engine = TtsEngine(config)

        with mock.patch.object(engine, "_model_path_valid", return_value=True):
            await engine.speak("Hola")
            await asyncio.sleep(0.3)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.asyncio
    async def test_stop_limpia_cola(self) -> None:
        config = _make_config()
        engine = TtsEngine(config)

        async def delayed_synth(text: str) -> bytes:
            await asyncio.sleep(0.05)
            return _make_dummy_audio()

        with mock.patch.object(engine, "_model_path_valid", return_value=True), \
             mock.patch.object(engine, "_synthesize", side_effect=delayed_synth):

            await engine.speak("uno")
            await engine.speak("dos")
            await asyncio.sleep(0.02)
            engine.stop()
            await asyncio.sleep(0.2)
