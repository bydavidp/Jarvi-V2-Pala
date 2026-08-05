from pathlib import Path
from unittest import mock

import pytest
import yaml

from brain.llm import LlmResponse, OllamaClient, OllamaError
from brain.router import IntentType, Router, RouterDecision


def _make_patterns_yaml(tmp_path: Path, patterns: list[dict]) -> str:
    path = tmp_path / "patterns.yaml"
    path.write_text(yaml.dump({"patterns": patterns}), encoding="utf-8")
    return str(path)


def _mock_client(responses: list[LlmResponse] | None = None) -> mock.MagicMock:
    client = mock.MagicMock(spec=OllamaClient)
    client.generate = mock.AsyncMock(side_effect=responses or [])
    return client


# ─── Params extraction con tipos ────────────────────────────────────────

class TestParamsExtraction:
    @pytest.mark.asyncio
    async def test_volume_set_level_int(self, tmp_path: Path) -> None:
        path = _make_patterns_yaml(tmp_path, [
            {"regex": "(?i)^\\s*pon\\s+(?:el\\s+)?volumen\\s*(?:a|al|en)\\s*(\\d+)\\s*$",
             "skill": "volume", "operation": "set", "params": {"level": {"group": 1, "type": "int"}}}
        ])
        client = _mock_client()
        router = Router(client, patterns_path=path)

        decision = await router.route("pon el volumen a 30")

        assert decision.resolved_by == "matcher"
        assert decision.params["level"] == 30
        assert isinstance(decision.params["level"], int)
        client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_volume_adjust_sin_numero(self, tmp_path: Path) -> None:
        path = _make_patterns_yaml(tmp_path, [
            {"regex": "(?i)^\\s*(sube|baja|subir|bajar)\\s+(?:el\\s+)?volumen\\s*$",
             "skill": "volume", "operation": "adjust", "params": {"delta": {"group": 1, "type": "str"}}}
        ])
        client = _mock_client()
        router = Router(client, patterns_path=path)

        decision = await router.route("sube el volumen")

        assert decision.resolved_by == "matcher"
        assert decision.skill_name == "volume"
        assert decision.operation == "adjust"
        assert decision.params["delta"] == "sube"

    @pytest.mark.asyncio
    async def test_volume_adjust_con_cantidad(self, tmp_path: Path) -> None:
        path = _make_patterns_yaml(tmp_path, [
            {"regex": "(?i)^\\s*(sube|baja)\\s+(?:el\\s+)?volumen\\s+(\\d+)\\s*$",
             "skill": "volume", "operation": "adjust",
             "params": {"delta": {"group": 1, "type": "str"}, "amount": {"group": 2, "type": "int"}}}
        ])
        client = _mock_client()
        router = Router(client, patterns_path=path)

        decision = await router.route("sube el volumen 10")

        assert decision.resolved_by == "matcher"
        assert decision.params["delta"] == "sube"
        assert decision.params["amount"] == 10

    @pytest.mark.asyncio
    async def test_conversion_int_falla_degrada(self, tmp_path: Path) -> None:
        path = _make_patterns_yaml(tmp_path, [
            {"regex": "(?i)^\\s*pon\\s+(?:el\\s+)?volumen\\s*(?:a|al|en)\\s*(\\S+)\\s*$",
             "skill": "volume", "operation": "set", "params": {"level": {"group": 1, "type": "int"}}}
        ])
        client = _mock_client([LlmResponse(text='{"intent": "CHAT"}')])
        router = Router(client, patterns_path=path)

        decision = await router.route("pon el volumen a mucho")

        assert decision.resolved_by == "llm"
        client.generate.assert_called_once()


# ─── Matcher básico ─────────────────────────────────────────────────────

class TestMatcher:
    @pytest.mark.asyncio
    async def test_matcher_sin_llm(self, tmp_path: Path) -> None:
        path = _make_patterns_yaml(tmp_path, [
            {"regex": "(?i)que hora es", "skill": "time", "operation": "get_current_time", "params": {}}
        ])
        client = _mock_client()
        router = Router(client, patterns_path=path)

        decision = await router.route("que hora es")
        assert decision.resolved_by == "matcher"
        client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_match_usa_llm(self, tmp_path: Path) -> None:
        path = _make_patterns_yaml(tmp_path, [])
        client = _mock_client([LlmResponse(text='{"intent": "CHAT"}')])
        router = Router(client, patterns_path=path)

        decision = await router.route("que opinas de Python")
        assert decision.resolved_by == "llm"


# ─── LLM Routing ─────────────────────────────────────────────────────────

class TestLlmRouting:
    @pytest.mark.asyncio
    async def test_json_valido_skill(self, tmp_path: Path) -> None:
        path = _make_patterns_yaml(tmp_path, [])
        client = _mock_client([
            LlmResponse(text='{"intent": "SKILL", "skill": "volume", "operation": "set", "params": {"level": 50}}')
        ])
        router = Router(client, patterns_path=path)
        decision = await router.route("sube el volumen al 50")
        assert decision.intent == IntentType.SKILL
        assert decision.params == {"level": 50}

    @pytest.mark.asyncio
    async def test_json_backtick(self, tmp_path: Path) -> None:
        path = _make_patterns_yaml(tmp_path, [])
        client = _mock_client([LlmResponse(text='```json\n{"intent": "CHAT"}\n```')])
        router = Router(client, patterns_path=path)
        decision = await router.route("hola")
        assert decision.intent == IntentType.CHAT

    @pytest.mark.asyncio
    async def test_json_invalido_degrade(self, tmp_path: Path) -> None:
        path = _make_patterns_yaml(tmp_path, [])
        client = _mock_client([
            LlmResponse(text="basura"),
            LlmResponse(text="mas basura"),
        ])
        router = Router(client, patterns_path=path)
        decision = await router.route("que hora es en marte")
        assert decision.intent == IntentType.CHAT
        assert client.generate.call_count == 2


# ─── Parametrizado: 25+ frases que deben resolver por matcher ──────────

MATCHER_PHRASES = [
    # Hora
    ("que hora es", "time", "get_current_time"),
    ("dime la hora", "time", "get_current_time"),
    ("cual es la hora", "time", "get_current_time"),
    ("quiero saber la hora", "time", "get_current_time"),
    ("me dices la hora por favor", "time", "get_current_time"),

    # Fecha
    ("que fecha es", "time", "get_current_time"),
    ("dime la fecha", "time", "get_current_time"),
    ("cual es la fecha de hoy", "time", "get_current_time"),

    # Volumen set
    ("pon el volumen a 50", "volume", "set"),
    ("poner volumen al 30", "volume", "set"),
    ("deja el volumen en 75", "volume", "set"),
    ("cambia el volumen a 20", "volume", "set"),

    # Volumen adjust
    ("sube el volumen", "volume", "adjust"),
    ("baja el volumen", "volume", "adjust"),
    ("aumenta el volumen", "volume", "adjust"),
    ("disminuye el volumen por favor", "volume", "adjust"),
    ("sube el volumen 15", "volume", "adjust"),
    ("baja el volumen 5", "volume", "adjust"),

    # Mute
    ("silencia el volumen", "volume", "mute"),
    ("mutea el sonido", "volume", "mute"),
    ("apaga el audio", "volume", "mute"),
    ("quita el sonido por favor", "volume", "mute"),

    # Unmute
    ("activa el sonido", "volume", "unmute"),
    ("enciende el audio", "volume", "unmute"),
    ("desmutea el volumen", "volume", "unmute"),

    # Abrir app
    ("abre chrome", "apps", "open"),
    ("abrir calculadora", "apps", "open"),
    ("abreme spotify", "apps", "open"),
    ("inicia el navegador", "apps", "open"),

    # Buscar
    ("busca tutoriales de Flask", "browser", "search"),
    ("googlea recetas de paella", "browser", "search"),
    ("investiga que es FastAPI", "browser", "search"),
    ("buscame en internet como instalar Ollama", "browser", "search"),

    # URL
    ("abre https://google.com", "browser", "open_url"),
    ("ve a https://github.com", "browser", "open_url"),
    ("navega a https://fastapi.tiangolo.com", "browser", "open_url"),
]


class TestMatcherCoverage:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("phrase,expected_skill,expected_op", MATCHER_PHRASES)
    async def test_matcher_resuelve(self, phrase: str, expected_skill: str, expected_op: str) -> None:
        router = Router(_mock_client())
        decision = await router.route(phrase)
        assert decision.resolved_by == "matcher", f"'{phrase}' fue resuelto por {decision.resolved_by}, no matcher"
        assert decision.skill_name == expected_skill, f"'{phrase}': skill={decision.skill_name}, esperado={expected_skill}"
        assert decision.operation == expected_op, f"'{phrase}': op={decision.operation}, esperado={expected_op}"
