import asyncio
import threading
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from core.bus import Event, EventBus, EventBusError
from core.config import Config, Settings, load_settings
from core.logs import get_logger, setup_logging
from core.state import AssistantState


class TestConfig:
    def test_load_settings_valido(self) -> None:
        settings = load_settings("config/settings.yaml")
        assert settings.idioma == "es"
        assert settings.ollama.url == "http://localhost:11434"
        assert settings.ollama.model == "llama3.2:3b"
        assert settings.whisper.language == "es"
        assert settings.whisper.compute_type == "int8"
        assert isinstance(settings.audio.vad_silence_ms, int)
        assert settings.audio.sample_rate == 16000
        assert "navegador" in settings.whitelist.apps
        assert settings.skills.time.enabled is True

    def test_config_objeto(self) -> None:
        config = Config()
        assert config.idioma == "es"
        assert config.ollama.timeout == 30

    def test_settings_falla_si_falta_piper(self, tmp_path: Path) -> None:
        incomplete = {
            "idioma": "es",
            "piper": {"model_path": "x"},
        }
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(yaml.dump(incomplete), encoding="utf-8")

        with pytest.raises(ValueError, match="Error de validación"):
            load_settings(str(yaml_path))

    def test_settings_falla_si_archivo_no_existe(self) -> None:
        with pytest.raises(FileNotFoundError, match="No se encontró"):
            load_settings("config/no_existe.yaml")

    def test_settings_falla_si_vacio(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="está vacío"):
            load_settings(str(yaml_path))


class TestState:
    def test_estados_definidos(self) -> None:
        assert AssistantState.IDLE.value == "idle"
        assert AssistantState.LISTENING.value == "listening"
        assert AssistantState.THINKING.value == "thinking"
        assert AssistantState.SPEAKING.value == "speaking"
        assert AssistantState.ERROR.value == "error"

    def test_estados_son_cinco(self) -> None:
        assert len(AssistantState) == 5

    def test_estado_por_defecto_es_idle(self) -> None:
        estado = AssistantState.IDLE
        assert estado == "idle"


class TestLogging:
    def test_setup_logging_devuelve_logger(self) -> None:
        logger = setup_logging(level=10, log_dir="logs", log_file="test.log")
        assert logger.name == "jarvis"
        assert logger.level == 10

    def test_get_logger_devuelve_instancia(self) -> None:
        logger = get_logger()
        assert logger is not None
        assert logger.name == "jarvis"


class TestEventBus:
    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test", handler)
        event = Event(type="test", data={"msg": "hola"})
        await bus.publish(event)

        assert len(received) == 1
        assert received[0].type == "test"
        assert received[0].data["msg"] == "hola"

    @pytest.mark.asyncio
    async def test_multiples_suscriptores(self) -> None:
        bus = EventBus()
        received: list[str] = []

        async def handler_a(event: Event) -> None:
            received.append("a")

        async def handler_b(event: Event) -> None:
            received.append("b")

        bus.subscribe("test", handler_a)
        bus.subscribe("test", handler_b)
        await bus.publish(Event(type="test"))

        assert len(received) == 2
        assert "a" in received
        assert "b" in received

    @pytest.mark.asyncio
    async def test_suscriptor_no_recibe_otro_tipo(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("tipo_a", handler)
        await bus.publish(Event(type="tipo_b"))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test", handler)
        bus.unsubscribe("test", handler)
        await bus.publish(Event(type="test"))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_publish_threadsafe_desde_hilo_real(self) -> None:
        bus = EventBus()
        bus.bind_loop(asyncio.get_running_loop())
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test", handler)

        def publish_from_thread() -> None:
            bus.publish_threadsafe(Event(type="test", data={"from": "thread"}))

        thread = threading.Thread(target=publish_from_thread)
        thread.start()
        thread.join()
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].data["from"] == "thread"

    def test_publish_threadsafe_sin_loop_enlazado(self) -> None:
        bus = EventBus()
        with pytest.raises(EventBusError, match="no tiene un event loop enlazado"):
            bus.publish_threadsafe(Event(type="test"))

    def test_publish_threadsafe_con_loop_cerrado(self) -> None:
        loop = asyncio.new_event_loop()
        bus = EventBus()
        bus.bind_loop(loop)
        loop.close()

        with pytest.raises(EventBusError):
            bus.publish_threadsafe(Event(type="test"))

    @pytest.mark.asyncio
    async def test_publish_threadsafe_suscriptor_lanza_excepcion(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus = EventBus()
        bus.bind_loop(asyncio.get_running_loop())

        async def faulty_handler(event: Event) -> None:
            raise ValueError("error simulado en suscriptor")

        bus.subscribe("test", faulty_handler)

        def publish_from_thread() -> None:
            bus.publish_threadsafe(Event(type="test"))

        thread = threading.Thread(target=publish_from_thread)
        thread.start()
        thread.join()
        await asyncio.sleep(0.2)

        assert "Excepción no capturada en suscriptor del bus" in caplog.text
        assert "error simulado en suscriptor" in caplog.text
