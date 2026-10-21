"""Tests de los clientes LLM y de la rotación DeepSeek / NVIDIA / Ollama.

Todo va con HTTP mockeado: no se gasta ni un token real ni se necesita red.
"""

import asyncio
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest

from brain.llm import (
    DeepSeekClient,
    DeepSeekError,
    HybridClient,
    LlmError,
    LlmResponse,
    NvidiaClient,
    NvidiaError,
    OllamaClient,
    OllamaError,
    build_llm_client,
    load_latency_rank,
)


def _mock_resp(data: dict) -> mock.AsyncMock:
    resp = mock.AsyncMock()
    resp.json = mock.MagicMock(return_value=data)
    resp.raise_for_status = mock.MagicMock()
    resp.status_code = 200
    resp.text = ""
    return resp


def _http_error(status: int, body: str) -> httpx.HTTPStatusError:
    resp = mock.AsyncMock()
    resp.status_code = status
    resp.text = body
    return httpx.HTTPStatusError("error", request=mock.MagicMock(), response=resp)


class FakeClient:
    """Cliente LLM falso: latencia, error y nombre controlados."""

    def __init__(
        self,
        model: str = "fake-model",
        provider: str = "fake",
        latency_ms: float = 100.0,
        error: Exception | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.provider_name = provider
        self._model = model
        self._latency = latency_ms
        self._error = error
        self._delay_s = delay_s
        self.calls = 0

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, model="", prompt="", system="", temperature=0.7, max_tokens=512, num_ctx=4096):
        self.calls += 1
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if self._error is not None:
            raise self._error
        return LlmResponse(
            text="hola",
            model=model or self._model,
            provider=self.provider_name,
            latency_ms=self._latency,
            total_tokens=10,
        )

    async def generate_stream(self, model="", prompt="", system="", temperature=0.7, max_tokens=512):
        self.calls += 1
        if self._error is not None:
            raise self._error
        yield "hola "
        yield "mundo"


# ─── DeepSeek ──────────────────────────────────────────────────────────


class TestDeepSeekClient:
    @pytest.mark.asyncio
    async def test_generate_ok(self) -> None:
        client = DeepSeekClient(api_key="sk-test", default_model="deepseek-flash")
        data = {
            "model": "deepseek-flash",
            "choices": [{"message": {"content": "  El cielo es azul.  "}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        }
        with mock.patch("httpx.AsyncClient.post", return_value=_mock_resp(data)) as post:
            result = await client.generate(prompt="¿por qué?", system="sys")

        assert result.text == "El cielo es azul."
        assert result.provider == "deepseek"
        assert result.total_tokens == 8
        assert result.tokens_per_second > 0

        payload = post.call_args.kwargs["json"]
        assert payload["model"] == "deepseek-flash"
        assert payload["messages"][0] == {"role": "system", "content": "sys"}
        assert payload["messages"][1] == {"role": "user", "content": "¿por qué?"}
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"

    @pytest.mark.asyncio
    async def test_402_habla_de_saldo(self) -> None:
        client = DeepSeekClient(api_key="sk-test")
        with mock.patch(
            "httpx.AsyncClient.post",
            side_effect=_http_error(402, '{"error":{"message":"Insufficient Balance"}}'),
        ):
            with pytest.raises(DeepSeekError, match="saldo insuficiente"):
                await client.generate(prompt="hola")

    @pytest.mark.asyncio
    async def test_sin_api_key_no_llama_a_la_red(self) -> None:
        client = DeepSeekClient(api_key="")
        with mock.patch("httpx.AsyncClient.post") as post:
            with pytest.raises(DeepSeekError, match="Falta la API key"):
                await client.generate(prompt="hola")
        post.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_claro(self) -> None:
        client = DeepSeekClient(api_key="sk-test", timeout=1)
        with mock.patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("t")):
            with pytest.raises(DeepSeekError, match="Timeout"):
                await client.generate(prompt="hola")

    @pytest.mark.asyncio
    async def test_respuesta_vacia_es_error(self) -> None:
        client = DeepSeekClient(api_key="sk-test")
        data = {"choices": [{"message": {"content": ""}}]}
        with mock.patch("httpx.AsyncClient.post", return_value=_mock_resp(data)):
            with pytest.raises(DeepSeekError, match="vacía"):
                await client.generate(prompt="hola")

    @pytest.mark.asyncio
    async def test_generate_stream_parsea_sse(self) -> None:
        client = DeepSeekClient(api_key="sk-test")
        lineas = [
            'data: {"choices":[{"delta":{"content":"Hola"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":", jefe"}}]}',
            "data: [DONE]",
        ]

        class FakeStream:
            status_code = 200

            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *args):
                return False

            async def aiter_lines(self_inner):
                for linea in lineas:
                    yield linea

            async def aread(self_inner):
                return b""

        with mock.patch("httpx.AsyncClient.stream", return_value=FakeStream()):
            trozos = [t async for t in client.generate_stream(prompt="hola")]

        assert "".join(trozos) == "Hola, jefe"

    def test_effort_solo_si_esta_configurado(self) -> None:
        sin_effort = DeepSeekClient(api_key="k")
        con_effort = DeepSeekClient(api_key="k", effort="low")
        assert "effort" not in sin_effort._payload("m", "p", "", 0.7, 10, False)
        assert con_effort._payload("m", "p", "", 0.7, 10, False)["effort"] == "low"


# ─── NVIDIA ────────────────────────────────────────────────────────────


class TestNvidiaClient:
    @pytest.mark.asyncio
    async def test_generate_ok(self) -> None:
        client = NvidiaClient(api_key="nvapi-test", default_model="nvidia/nemotron-3.5-lightning-30b-a3b")
        data = {
            "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
            "choices": [{"message": {"content": "Azul por la dispersión de Rayleigh."}}],
            "usage": {"completion_tokens": 12},
        }
        with mock.patch("httpx.AsyncClient.post", return_value=_mock_resp(data)) as post:
            result = await client.generate(prompt="¿por qué?")

        assert result.provider == "nvidia"
        assert result.model == "nvidia/nemotron-3.5-lightning-30b-a3b"
        assert result.total_tokens == 12
        payload = post.call_args.kwargs["json"]
        assert payload["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer nvapi-test"

    @pytest.mark.asyncio
    async def test_403_explica_permisos(self) -> None:
        client = NvidiaClient(api_key="nvapi-test")
        with mock.patch(
            "httpx.AsyncClient.post",
            side_effect=_http_error(403, '{"detail":"Authorization failed"}'),
        ):
            with pytest.raises(NvidiaError, match="sin permisos"):
                await client.generate(prompt="hola")

    @pytest.mark.asyncio
    async def test_410_modelo_retirado(self) -> None:
        client = NvidiaClient(api_key="nvapi-test")
        with mock.patch(
            "httpx.AsyncClient.post",
            side_effect=_http_error(410, "The model has reached its end of life"),
        ):
            with pytest.raises(NvidiaError, match="fin de vida"):
                await client.generate(prompt="hola")


# ─── Rotación ──────────────────────────────────────────────────────────


class TestHybridClient:
    @pytest.mark.asyncio
    async def test_primer_proveedor_sano_responde(self) -> None:
        primary = FakeClient("deepseek-flash", "deepseek", latency_ms=800)
        fallback = FakeClient("llama3.2", "ollama", latency_ms=300)
        hybrid = HybridClient(primary, fallback)

        r = await hybrid.generate(prompt="hola")

        assert r.provider == "deepseek"
        assert primary.calls == 1
        assert fallback.calls == 0

    @pytest.mark.asyncio
    async def test_si_el_primero_falla_usa_el_segundo(self) -> None:
        primary = FakeClient("deepseek-flash", "deepseek", error=DeepSeekError("402 saldo"))
        fallback = FakeClient("llama3.2", "ollama")
        hybrid = HybridClient(primary, fallback)

        r = await hybrid.generate(prompt="hola")

        assert r.provider == "ollama"
        assert hybrid.status()["providers"][0]["failures"] == 1

    @pytest.mark.asyncio
    async def test_rota_entre_tres_proveedores(self) -> None:
        caido1 = FakeClient("deepseek-flash", "deepseek", error=DeepSeekError("402"))
        caido2 = FakeClient("nvidia/x", "nvidia", error=NvidiaError("403"))
        bueno = FakeClient("llama3.2", "ollama")
        hybrid = HybridClient(providers=[caido1, caido2, bueno])

        r = await hybrid.generate(prompt="hola")

        assert r.provider == "ollama"
        assert caido1.calls == 1 and caido2.calls == 1 and bueno.calls == 1

    @pytest.mark.asyncio
    async def test_dos_fallos_aparcan_el_proveedor(self) -> None:
        reloj = [1000.0]
        primary = FakeClient("deepseek-flash", "deepseek", error=DeepSeekError("boom"))
        fallback = FakeClient("llama3.2", "ollama")
        hybrid = HybridClient(primary, fallback, failure_trip_count=2, cooldown_s=60, clock=lambda: reloj[0])

        await hybrid.generate(prompt="1")
        await hybrid.generate(prompt="2")
        assert primary.calls == 2

        r = await hybrid.generate(prompt="3")
        assert primary.calls == 2
        assert r.provider == "ollama"

    @pytest.mark.asyncio
    async def test_lentitud_aparca_el_primero(self) -> None:
        reloj = [1000.0]
        primary = FakeClient("deepseek-flash", "deepseek", latency_ms=4000)
        fallback = FakeClient("llama3.2", "ollama", latency_ms=300)
        hybrid = HybridClient(
            primary, fallback, switch_after_slow_ms=2500, slow_trip_count=2, clock=lambda: reloj[0]
        )

        assert (await hybrid.generate(prompt="1")).provider == "deepseek"
        assert (await hybrid.generate(prompt="2")).provider == "deepseek"

        r = await hybrid.generate(prompt="3")
        assert r.provider == "ollama"
        assert primary.calls == 2

    @pytest.mark.asyncio
    async def test_una_rapida_reinicia_el_contador_de_lentitud(self) -> None:
        reloj = [1000.0]
        primary = FakeClient("deepseek-flash", "deepseek", latency_ms=4000)
        fallback = FakeClient("llama3.2", "ollama")
        hybrid = HybridClient(
            primary, fallback, switch_after_slow_ms=2500, slow_trip_count=2, clock=lambda: reloj[0]
        )

        await hybrid.generate(prompt="1")
        primary._latency = 500
        await hybrid.generate(prompt="2")
        primary._latency = 4000
        await hybrid.generate(prompt="3")

        assert hybrid.status()["providers"][0]["slow_streak"] == 1
        assert primary.calls == 3

    @pytest.mark.asyncio
    async def test_cooldown_expirado_reintenta(self) -> None:
        reloj = [1000.0]
        primary = FakeClient("deepseek-flash", "deepseek", error=DeepSeekError("boom"))
        fallback = FakeClient("llama3.2", "ollama")
        hybrid = HybridClient(
            primary, fallback, failure_trip_count=2, cooldown_s=60, clock=lambda: reloj[0]
        )

        await hybrid.generate(prompt="1")
        await hybrid.generate(prompt="2")
        assert primary.calls == 2

        reloj[0] += 61
        primary._error = None
        r = await hybrid.generate(prompt="3")

        assert primary.calls == 3
        assert r.provider == "deepseek"
        assert hybrid.status()["providers"][0]["failures"] == 0

    @pytest.mark.asyncio
    async def test_timeout_conmuta(self) -> None:
        primary = FakeClient("deepseek-flash", "deepseek", delay_s=0.2)
        fallback = FakeClient("llama3.2", "ollama")
        hybrid = HybridClient(primary, fallback, primary_timeout=0.01)

        r = await hybrid.generate(prompt="hola")

        assert r.provider == "ollama"
        assert "timeout" in hybrid.status()["providers"][0]["last_error"]

    @pytest.mark.asyncio
    async def test_ollama_local_no_lleva_tope_artificial(self) -> None:
        local = FakeClient("llama3.2", "ollama", delay_s=0.05)
        hybrid = HybridClient(providers=[local], timeouts={"ollama": None}, primary_timeout=0.001)

        r = await hybrid.generate(prompt="hola")

        assert r.provider == "ollama"

    @pytest.mark.asyncio
    async def test_todos_fallan_lanza_llm_error(self) -> None:
        a = FakeClient(error=DeepSeekError("deep caído"))
        b = FakeClient(error=OllamaError("ollama caído"))
        hybrid = HybridClient(providers=[a, b])

        with pytest.raises(LlmError, match="Todos los proveedores"):
            await hybrid.generate(prompt="hola")

    @pytest.mark.asyncio
    async def test_todos_degradados_se_intenta_igual(self) -> None:
        reloj = [1000.0]
        a = FakeClient(error=DeepSeekError("boom"))
        b = FakeClient("llama3.2", "ollama")
        hybrid = HybridClient(
            providers=[a, b], failure_trip_count=1, cooldown_s=60, clock=lambda: reloj[0]
        )

        await hybrid.generate(prompt="1")
        assert hybrid.status()["active"] == "ollama"

        # Ahora también se degrada Ollama: no debe quedarse sin intentar nada.
        b._error = OllamaError("boom")
        with pytest.raises(LlmError, match="Todos los proveedores"):
            await hybrid.generate(prompt="2")

    @pytest.mark.asyncio
    async def test_round_robin_reparte_el_primero(self) -> None:
        a = FakeClient("deepseek-flash", "deepseek")
        b = FakeClient("llama3.2", "ollama")
        hybrid = HybridClient(providers=[a, b], rotation="round_robin")

        primeros = {(await hybrid.generate(prompt="x")).provider for _ in range(4)}

        assert primeros == {"deepseek", "ollama"}

    def test_model_for_solo_sobrescribe_ollama(self) -> None:
        deep = FakeClient("deepseek-flash", "deepseek")
        nv = FakeClient("nvidia/x", "nvidia")
        local = FakeClient("llama3.2:latest", "ollama")
        hybrid = HybridClient(providers=[deep, nv, local])

        assert hybrid._model_for(deep, "llama3.2:3b") == "deepseek-flash"
        assert hybrid._model_for(nv, "llama3.2:3b") == "nvidia/x"
        assert hybrid._model_for(local, "llama3.2:3b") == "llama3.2:3b"
        assert hybrid._model_for(local, "") == "llama3.2:latest"

    def test_status_lista_los_tres(self) -> None:
        hybrid = HybridClient(
            providers=[
                FakeClient("deepseek-flash", "deepseek"),
                FakeClient("nvidia/x", "nvidia"),
                FakeClient("llama3.2", "ollama"),
            ]
        )
        status = hybrid.status()

        assert [p["name"] for p in status["providers"]] == ["deepseek", "nvidia", "ollama"]
        assert status["active"] == "deepseek"

    @pytest.mark.asyncio
    async def test_stream_cae_al_siguiente_si_no_emitio(self) -> None:
        primary = FakeClient(error=DeepSeekError("boom"))
        fallback = FakeClient("llama3.2", "ollama")
        hybrid = HybridClient(primary, fallback)

        trozos = [t async for t in hybrid.generate_stream(prompt="hola")]

        assert "".join(trozos) == "hola mundo"
        assert fallback.calls == 1


# ─── Fábrica ───────────────────────────────────────────────────────────


def _llm_config(**overrides):
    base = dict(
        mode="auto",
        rotation="fastest",
        providers=["deepseek", "nvidia", "ollama"],
        switch_after_slow_ms=2500,
        slow_trip_count=2,
        failure_trip_count=2,
        cooldown_s=120,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _config(**overrides):
    base = dict(
        ollama=SimpleNamespace(
            url="http://localhost:11434",
            model="llama3.2:latest",
            keep_alive="30m",
            timeout=30,
        ),
        deepseek=SimpleNamespace(
            enabled=True,
            url="https://api.deepseek.com",
            model="deepseek-flash",
            models=["deepseek-flash"],
            effort="",
            timeout=20,
            primary_timeout=10,
        ),
        nvidia=SimpleNamespace(
            enabled=True,
            url="https://integrate.api.nvidia.com/v1",
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            models=[],
            timeout=30,
            primary_timeout=8,
        ),
        groq=SimpleNamespace(
            enabled=False,
            url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            timeout=20,
            primary_timeout=6,
        ),
        llm=_llm_config(),
        deepseek_api_key="sk-test",
        nvidia_api_key="nvapi-test",
        groq_api_key="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestBuildLlmClient:
    def _build(self, cfg):
        # Sin archivo de ranking: el orden lo decide settings.
        with mock.patch("brain.llm.load_latency_rank", return_value={}):
            return build_llm_client(cfg)

    def test_sin_keys_solo_ollama(self) -> None:
        client = self._build(_config(deepseek_api_key="", nvidia_api_key=""))
        assert isinstance(client, OllamaClient)

    def test_modo_ollama_ignora_la_nube(self) -> None:
        client = self._build(_config(llm=_llm_config(mode="ollama")))
        assert isinstance(client, OllamaClient)

    def test_con_las_tres_keys_rota_entre_tres(self) -> None:
        client = self._build(_config())
        assert isinstance(client, HybridClient)
        assert [p["name"] for p in client.status()["providers"]] == ["deepseek", "nvidia", "ollama"]

    def test_sin_key_nvidia_rota_entre_dos(self) -> None:
        client = self._build(_config(nvidia_api_key=""))
        assert isinstance(client, HybridClient)
        assert [p["name"] for p in client.status()["providers"]] == ["deepseek", "ollama"]

    def test_el_ranking_reordena_por_velocidad(self) -> None:
        cfg = _config()
        with mock.patch(
            "brain.llm.load_latency_rank",
            return_value={"nvidia": 700.0, "deepseek": 1200.0, "ollama": 4000.0},
        ):
            client = build_llm_client(cfg)
        assert [p["name"] for p in client.status()["providers"]] == ["nvidia", "deepseek", "ollama"]

    def test_ranking_ausente_respeta_settings(self) -> None:
        cfg = _config(llm=_llm_config(providers=["ollama", "nvidia", "deepseek"]))
        client = self._build(cfg)
        assert [p["name"] for p in client.status()["providers"]] == ["ollama", "nvidia", "deepseek"]

    def test_load_latency_rank_ignora_basura(self, tmp_path) -> None:
        archivo = tmp_path / "rank.json"
        archivo.write_text(
            '{"providers": {"deepseek": 900, "nvidia": "rápido", "ollama": 0}}',
            encoding="utf-8",
        )
        assert load_latency_rank(archivo) == {"deepseek": 900.0}

    def test_load_latency_rank_sin_archivo(self, tmp_path) -> None:
        assert load_latency_rank(tmp_path / "no-existe.json") == {}

class TestHintDeModelo:
    def test_ollama_no_recibe_el_modelo_de_nvidia(self) -> None:
        nv = FakeClient(model="nvidia/nemotron-x", provider="nvidia")
        local = FakeClient(model="llama3.2:latest", provider="ollama")
        hybrid = HybridClient(providers=[nv, local])

        assert hybrid._model_for(local, "nvidia/nemotron-x") == "llama3.2:latest"

    def test_ollama_acepta_un_modelo_local_como_hint(self) -> None:
        nv = FakeClient(model="nvidia/nemotron-x", provider="nvidia")
        local = FakeClient(model="llama3.2:latest", provider="ollama")
        hybrid = HybridClient(providers=[nv, local])

        assert hybrid._model_for(local, "qwen2.5-coder:0.5b") == "qwen2.5-coder:0.5b"

class TestGroq:
    def test_cliente_groq_usa_dialecto_openai(self) -> None:
        from brain.llm import GroqClient, GroqError, RemoteLlmError

        cliente = GroqClient(api_key="gsk_test")

        assert cliente.provider_name == "groq"
        assert cliente.model == "llama-3.1-8b-instant"
        assert issubclass(GroqError, RemoteLlmError)

    def test_fabrica_incluye_groq_si_hay_key(self, monkeypatch) -> None:
        from brain.llm import HybridClient, build_llm_client
        from core.config import Config

        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        cliente = build_llm_client(Config())

        assert isinstance(cliente, HybridClient)
        assert "groq" in [p["name"] for p in cliente.status()["providers"]]

    def test_fabrica_deja_groq_fuera_sin_key(self, monkeypatch) -> None:
        from brain.llm import HybridClient, build_llm_client
        from core.config import Config

        monkeypatch.setenv("GROQ_API_KEY", "")
        cliente = build_llm_client(Config())

        if isinstance(cliente, HybridClient):
            assert "groq" not in [p["name"] for p in cliente.status()["providers"]]
