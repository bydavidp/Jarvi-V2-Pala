"""Clientes LLM: Ollama local, DeepSeek, NVIDIA y rotación automática.

Diseño:
- Todo cliente implementa el protocolo ``LlmClient`` (mismo ``generate`` /
  ``generate_stream``), así el router y el orchestrator no saben a quién llaman.
- ``DeepSeekClient`` y ``NvidiaClient`` comparten ``OpenAICompatibleClient``:
  ambas APIs hablan el dialecto de OpenAI (``/chat/completions``).
- ``HybridClient`` ordena los proveedores por velocidad medida y rota entre
  ellos: intenta el más rápido, y si falla, tarda o se degrada pasa al
  siguiente. La degradación es un *circuit breaker* con reintento half-open.
- El orden sale de ``data/models_rank.json`` (lo escribe ``scripts/bench_models.py``).
  Sin ranking, se usa el orden de ``llm.providers`` en ``settings.yaml``.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, Callable, Protocol, Sequence, runtime_checkable

import httpx

if TYPE_CHECKING:
    from core.config import Config

logger = logging.getLogger("jarvis.brain.llm")

RANK_PATH = Path("data/models_rank.json")


@dataclass
class LlmResponse:
    text: str
    tokens_per_second: float = 0.0
    total_tokens: int = 0
    model: str = ""
    eval_count: int = 0
    eval_duration_ns: int = 0
    provider: str = ""
    latency_ms: float = 0.0


class LlmError(Exception):
    """Error genérico de cualquier proveedor LLM."""


class OllamaError(LlmError):
    """Error de conexión o respuesta del servicio Ollama."""


class RemoteLlmError(LlmError):
    """Error de una API remota (DeepSeek, NVIDIA)."""

    provider = "remote"


class DeepSeekError(RemoteLlmError):
    provider = "deepseek"


class NvidiaError(RemoteLlmError):
    provider = "nvidia"


@runtime_checkable
class LlmClient(Protocol):
    provider_name: str

    @property
    def model(self) -> str: ...

    async def generate(
        self,
        model: str = "",
        prompt: str = "",
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        num_ctx: int = 4096,
    ) -> LlmResponse: ...

    def generate_stream(
        self,
        model: str = "",
        prompt: str = "",
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncGenerator[str, None]: ...


# ─── Ollama (local) ────────────────────────────────────────────────────


class OllamaClient:
    provider_name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        keep_alive: str = "30m",
        default_model: str = "llama3.2:latest",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._default_model = default_model

    @property
    def model(self) -> str:
        return self._default_model

    async def generate(
        self,
        model: str = "",
        prompt: str = "",
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        num_ctx: int = 4096,
    ) -> LlmResponse:
        model = model or self._default_model
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
            },
        }

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError:
            raise OllamaError(
                f"No se pudo conectar a Ollama en {self._base_url}. "
                "¿Está corriendo? Ejecuta: ollama serve"
            ) from None
        except httpx.TimeoutException:
            raise OllamaError(
                f"Timeout esperando respuesta de Ollama ({self._timeout}s)"
            ) from None
        except httpx.HTTPStatusError as e:
            raise OllamaError(
                f"Ollama devolvió error {e.response.status_code}: {e.response.text[:200]}"
            ) from e

        latency_ms = (time.perf_counter() - start) * 1000
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)

        return LlmResponse(
            text=data.get("response", "").strip(),
            tokens_per_second=eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0,
            total_tokens=eval_count,
            model=data.get("model", model),
            eval_count=eval_count,
            eval_duration_ns=eval_duration_ns,
            provider=self.provider_name,
            latency_ms=latency_ms,
        )

    async def generate_stream(
        self,
        model: str = "",
        prompt: str = "",
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncGenerator[str, None]:
        model = model or self._default_model
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        token = chunk.get("response", "")
                        if token:
                            yield token
        except httpx.ConnectError:
            raise OllamaError(
                f"No se pudo conectar a Ollama en {self._base_url}."
            ) from None
        except httpx.TimeoutException:
            raise OllamaError(
                f"Timeout esperando respuesta de Ollama ({self._timeout}s)"
            ) from None
        except httpx.HTTPStatusError as e:
            raise OllamaError(
                f"Ollama devolvió error {e.response.status_code}: {e.response.text[:200]}"
            ) from e


# ─── APIs remotas (DeepSeek, NVIDIA: dialecto OpenAI) ──────────────────


class OpenAICompatibleClient:
    """Base para APIs que exponen ``/chat/completions`` estilo OpenAI."""

    provider_name = "openai_compatible"
    error_cls: type[RemoteLlmError] = RemoteLlmError
    top_up_url = ""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        default_model: str,
        timeout: float = 20.0,
        effort: str = "",
    ) -> None:
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._timeout = timeout
        self._effort = effort.strip().lower()

    @property
    def model(self) -> str:
        return self._default_model

    def _fail(self, message: str) -> RemoteLlmError:
        return self.error_cls(message)

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise self._fail(
                f"Falta la API key de {self.provider_name}: no se puede llamar a {self._base_url}"
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        model: str,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": model or self._default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        # "effort" solo se envía si está configurado (low/high/max).
        if self._effort:
            payload["effort"] = self._effort
        return payload

    def _describe_status(self, status: int, body: str) -> str:
        nombre = self.provider_name
        if status == 402:
            extra = f" Recarga en {self.top_up_url}" if self.top_up_url else ""
            return f"{nombre}: saldo insuficiente.{extra} ({body[:120]})"
        if status in (401, 403):
            return f"{nombre}: API key inválida o sin permisos para inferencia"
        if status == 404:
            return f"{nombre}: el modelo no existe o está retirado ({body[:100]})"
        if status == 410:
            return f"{nombre}: el modelo alcanzó su fin de vida ({body[:100]})"
        if status == 429:
            return f"{nombre}: límite de peticiones alcanzado"
        return f"{nombre} devolvió error {status}: {body[:200]}"

    async def generate(
        self,
        model: str = "",
        prompt: str = "",
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        num_ctx: int = 4096,  # aceptado por compatibilidad; estas APIs no lo usan
    ) -> LlmResponse:
        payload = self._payload(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        url = f"{self._base_url}/chat/completions"
        headers = self._headers()

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError:
            raise self._fail(f"No se pudo conectar a {self.provider_name} en {self._base_url}") from None
        except httpx.TimeoutException:
            raise self._fail(
                f"Timeout esperando respuesta de {self.provider_name} ({self._timeout}s)"
            ) from None
        except httpx.HTTPStatusError as e:
            raise self._fail(
                self._describe_status(e.response.status_code, e.response.text)
            ) from e

        latency_ms = (time.perf_counter() - start) * 1000
        choices = data.get("choices") or []
        if not choices:
            raise self._fail(f"{self.provider_name} devolvió una respuesta sin 'choices'")

        message = choices[0].get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            raise self._fail(f"{self.provider_name} devolvió una respuesta vacía")

        usage = data.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        seconds = max(latency_ms / 1000, 1e-6)

        return LlmResponse(
            text=text,
            tokens_per_second=completion_tokens / seconds if completion_tokens else 0.0,
            total_tokens=completion_tokens,
            model=data.get("model", payload["model"]),
            eval_count=completion_tokens,
            provider=self.provider_name,
            latency_ms=latency_ms,
        )

    async def generate_stream(
        self,
        model: str = "",
        prompt: str = "",
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncGenerator[str, None]:
        payload = self._payload(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        url = f"{self._base_url}/chat/completions"
        headers = self._headers()

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST", url, json=payload, headers=headers
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        raise self._fail(self._describe_status(resp.status_code, body))
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if not chunk:
                            continue
                        if chunk == "[DONE]":
                            break
                        try:
                            obj = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        token = delta.get("content")
                        if token:
                            yield token
        except httpx.ConnectError:
            raise self._fail(f"No se pudo conectar a {self.provider_name} en {self._base_url}") from None
        except httpx.TimeoutException:
            raise self._fail(
                f"Timeout esperando respuesta de {self.provider_name} ({self._timeout}s)"
            ) from None


class DeepSeekClient(OpenAICompatibleClient):
    provider_name = "deepseek"
    error_cls = DeepSeekError
    top_up_url = "https://platform.deepseek.com/top_up"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        default_model: str = "deepseek-flash",
        timeout: float = 20.0,
        effort: str = "",
    ) -> None:
        super().__init__(api_key, base_url, default_model, timeout, effort)


class NvidiaClient(OpenAICompatibleClient):
    provider_name = "nvidia"
    error_cls = NvidiaError

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        default_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
        timeout: float = 30.0,
        effort: str = "",
    ) -> None:
        super().__init__(api_key, base_url, default_model, timeout, effort)


# ─── Rotación automática ───────────────────────────────────────────────


@dataclass
class ProviderHealth:
    """Estado observado de un proveedor. Solo lo escribe ``HybridClient``."""

    failures: int = 0
    slow_streak: int = 0
    degraded_until: float = 0.0
    last_latency_ms: float = 0.0
    last_error: str = ""
    requests: int = 0
    successes: int = 0

    def is_degraded(self, now: float) -> bool:
        return now < self.degraded_until

    def as_dict(self) -> dict:
        return {
            "failures": self.failures,
            "slow_streak": self.slow_streak,
            "degraded": self.degraded_until > 0,
            "degraded_until": round(self.degraded_until, 1),
            "last_latency_ms": round(self.last_latency_ms, 1),
            "last_error": self.last_error,
            "requests": self.requests,
            "successes": self.successes,
        }


class HybridClient:
    """Rota entre varios proveedores según salud y latencia medidas.

    Reglas:
    1. Se prueban en orden; el primero sano que responde gana.
    2. Cada proveedor remoto tiene un tope de tiempo (``timeouts``) para que la
       voz no se quede esperando. Ollama local no lleva tope artificial.
    3. ``failure_trip_count`` fallos seguidos aparcan un proveedor ``cooldown_s``.
    4. ``slow_trip_count`` respuestas más lentas que ``switch_after_slow_ms``
       también lo aparcan: así "se cambia solo" cuando el rápido se vuelve lento.
    5. Al expirar el cooldown se reintenta (half-open); si va bien, se queda.
    6. ``rotation="round_robin"`` reparte la primera elección entre los sanos
       (útil para no quemar la cuota de uno solo). ``"fastest"`` siempre empieza
       por el mejor medido.
    7. Si todos están degradados se intenta igual: es peor fallar que rotar.
    """

    provider_name = "hybrid"

    def __init__(
        self,
        primary: LlmClient | None = None,
        fallback: LlmClient | None = None,
        *,
        providers: Sequence[LlmClient] | None = None,
        primary_timeout: float = 10.0,
        timeouts: dict[str, float | None] | None = None,
        switch_after_slow_ms: float = 2500.0,
        slow_trip_count: int = 2,
        failure_trip_count: int = 2,
        cooldown_s: float = 120.0,
        rotation: str = "fastest",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if providers is not None:
            self._providers: list[LlmClient] = list(providers)
        elif primary is not None:
            self._providers = [primary] + ([fallback] if fallback is not None else [])
        else:
            self._providers = [fallback] if fallback is not None else []
        if not self._providers:
            raise ValueError("HybridClient necesita al menos un proveedor")

        self._primary_timeout = primary_timeout
        self._timeouts = dict(timeouts or {})
        self._switch_after_slow_ms = switch_after_slow_ms
        self._slow_trip_count = max(1, slow_trip_count)
        self._failure_trip_count = max(1, failure_trip_count)
        self._cooldown_s = cooldown_s
        self._rotation = rotation if rotation in ("fastest", "round_robin") else "fastest"
        self._clock = clock
        self._rr = 0

        # Las claves son el nombre del proveedor; si se repite, sufijo numérico.
        self._names: list[str] = []
        vistos: dict[str, int] = {}
        for provider in self._providers:
            base = provider.provider_name
            vistos[base] = vistos.get(base, 0) + 1
            self._names.append(base if vistos[base] == 1 else f"{base}{vistos[base]}")
        self._health = {name: ProviderHealth() for name in self._names}

    # ─── Propiedades ───────────────────────────────────────────────────

    def _activos(self) -> list[tuple[str, LlmClient]]:
        ahora = self._clock()
        sanos = [
            (name, provider)
            for name, provider in zip(self._names, self._providers)
            if not self._health[name].is_degraded(ahora)
        ]
        if not sanos:
            # Todo aparcado: se intenta igual, es peor dejar sin respuesta.
            return list(zip(self._names, self._providers))
        if self._rotation == "round_robin" and len(sanos) > 1:
            k = self._rr % len(sanos)
            self._rr += 1
            sanos = sanos[k:] + sanos[:k]
        return sanos

    @property
    def model(self) -> str:
        return self._activos()[0][1].model

    @property
    def provider(self) -> str:
        return self._activos()[0][0]

    def status(self) -> dict:
        activos = dict(self._activos())
        return {
            "rotation": self._rotation,
            "active": self.provider,
            "providers": [
                {
                    "name": name,
                    "model": provider.model,
                    "selected": name in activos,
                    **self._health[name].as_dict(),
                }
                for name, provider in zip(self._names, self._providers)
            ],
        }

    # ─── API pública ───────────────────────────────────────────────────

    async def generate(
        self,
        model: str = "",
        prompt: str = "",
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        num_ctx: int = 4096,
    ) -> LlmResponse:
        errores: list[str] = []

        for name, provider in self._activos():
            try:
                coro = provider.generate(
                    model=self._model_for(provider, model),
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    num_ctx=num_ctx,
                )
                deadline = self._deadline_for(name)
                if deadline is None:
                    resp = await coro
                else:
                    resp = await asyncio.wait_for(coro, timeout=deadline)
            except (LlmError, asyncio.TimeoutError) as e:
                descripcion = self._describe(e, name)
                self._record_failure(name, descripcion)
                errores.append(f"{name}: {descripcion}")
                logger.warning("%s no usable (%s). Siguiente proveedor.", name, descripcion)
                continue

            self._record_success(name, resp.latency_ms)
            logger.info(
                "LLM proveedor=%s modelo=%s latencia=%.0fms",
                resp.provider,
                resp.model,
                resp.latency_ms,
            )
            return resp

        raise LlmError("Todos los proveedores fallaron: " + " | ".join(errores))

    async def generate_stream(
        self,
        model: str = "",
        prompt: str = "",
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncGenerator[str, None]:
        errores: list[str] = []

        for name, provider in self._activos():
            emitidos = 0
            try:
                async for token in provider.generate_stream(
                    model=self._model_for(provider, model),
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    emitidos += 1
                    yield token
            except LlmError as e:
                self._record_failure(name, str(e))
                errores.append(f"{name}: {e}")
                logger.warning("Stream de %s interrumpido: %s", name, e)
                if emitidos:
                    raise
                continue
            else:
                self._record_success(name, None)
                return

        raise LlmError("Todos los proveedores fallaron: " + " | ".join(errores))

    # ─── Interno ───────────────────────────────────────────────────────

    def _model_for(self, provider: LlmClient, hint: str) -> str:
        """El router puede pedir un modelo pequeño, pero eso solo aplica a Ollama.

        Los proveedores remotos usan su modelo configurado (el elegido por
        benchmark); un id de DeepSeek o NVIDIA no sirve en los otros.
        """
        hint = (hint or "").strip()
        remotos = {p.model for p in self._providers if p.provider_name != "ollama"}
        if hint and provider.provider_name == "ollama" and hint not in remotos:
            return hint
        return provider.model

    def _deadline_for(self, name: str) -> float | None:
        if name in self._timeouts:
            return self._timeouts[name]
        return self._primary_timeout

    def _record_failure(self, name: str, error: str) -> None:
        health = self._health[name]
        health.requests += 1
        health.failures += 1
        health.slow_streak = 0
        health.last_error = error
        if health.failures >= self._failure_trip_count:
            self._degrade(name, f"{health.failures} fallos seguidos")

    def _record_success(self, name: str, latency_ms: float | None) -> None:
        health = self._health[name]
        health.requests += 1
        health.successes += 1
        health.failures = 0
        health.last_error = ""
        if latency_ms is None:
            return
        health.last_latency_ms = latency_ms
        if latency_ms >= self._switch_after_slow_ms:
            health.slow_streak += 1
            logger.info(
                "%s lento: %.0fms >= %.0fms (%d/%d)",
                name,
                latency_ms,
                self._switch_after_slow_ms,
                health.slow_streak,
                self._slow_trip_count,
            )
            if health.slow_streak >= self._slow_trip_count:
                self._degrade(name, f"{health.slow_streak} respuestas lentas")
        else:
            health.slow_streak = 0

    def _degrade(self, name: str, motivo: str) -> None:
        health = self._health[name]
        health.degraded_until = self._clock() + self._cooldown_s
        logger.warning(
            "Proveedor %s aparcado %.0fs (%s).", name, self._cooldown_s, motivo
        )

    def _describe(self, error: BaseException, name: str) -> str:
        if isinstance(error, asyncio.TimeoutError):
            tope = self._deadline_for(name)
            return f"timeout de {tope:.0f}s" if tope else "timeout"
        return str(error) or error.__class__.__name__


# ─── Fábrica ───────────────────────────────────────────────────────────


def load_latency_rank(path: Path = RANK_PATH) -> dict[str, float]:
    """Lee las medianas medidas por ``scripts/bench_models.py``.

    Formato: ``{"providers": {"deepseek": 1200.0, "nvidia": 900.0}}``.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("No pude leer %s: %s", path, e)
        return {}
    providers = data.get("providers") or {}
    return {
        str(name): float(ms)
        for name, ms in providers.items()
        if isinstance(ms, (int, float)) and ms > 0
    }


def _order_providers(
    providers: list[LlmClient], preferred: Sequence[str], rank: dict[str, float]
) -> list[LlmClient]:
    """Ordena por latencia medida; sin medición, respeta ``llm.providers``."""
    por_nombre = {p.provider_name: p for p in providers}
    orden = [n for n in preferred if n in por_nombre]
    orden += [n for n in por_nombre if n not in orden]
    con_dato = [n for n in orden if n in rank]
    sin_dato = [n for n in orden if n not in rank]
    con_dato.sort(key=lambda n: rank[n])
    return [por_nombre[n] for n in con_dato + sin_dato]


def build_llm_client(config: "Config") -> LlmClient:
    """Arma el cliente LLM según ``settings.yaml``, ``.env`` y el benchmark.

    Modos:
    - ``auto``: rota entre los proveedores con key y ordena por velocidad medida.
    - ``ollama``: 100% local, la nube ni se toca.

    Un proveedor sin API key simplemente no entra en la rotación.
    """
    ollama = OllamaClient(
        base_url=config.ollama.url,
        timeout=config.ollama.timeout,
        keep_alive=config.ollama.keep_alive,
        default_model=config.ollama.model,
    )

    mode = (config.llm.mode or "auto").strip().lower()
    if mode == "ollama":
        logger.info("LLM: modo 'ollama' (100%% local, sin nube)")
        return ollama

    candidatos: list[LlmClient] = []
    timeouts: dict[str, float | None] = {"ollama": None}

    ds = config.deepseek
    if ds.enabled and config.deepseek_api_key.strip():
        candidatos.append(
            DeepSeekClient(
                api_key=config.deepseek_api_key,
                base_url=ds.url,
                default_model=ds.model,
                timeout=ds.timeout,
                effort=ds.effort,
            )
        )
        timeouts["deepseek"] = float(ds.primary_timeout)
    elif ds.enabled:
        logger.warning("LLM: falta DEEPSEEK_API_KEY en .env; DeepSeek queda fuera")

    nv = config.nvidia
    if nv.enabled and config.nvidia_api_key.strip():
        candidatos.append(
            NvidiaClient(
                api_key=config.nvidia_api_key,
                base_url=nv.url,
                default_model=nv.model,
                timeout=nv.timeout,
            )
        )
        timeouts["nvidia"] = float(nv.primary_timeout)
    elif nv.enabled:
        logger.warning("LLM: falta NVIDIA_API_KEY en .env; NVIDIA queda fuera")

    candidatos.append(ollama)

    ordenados = _order_providers(
        candidatos, list(config.llm.providers), load_latency_rank()
    )
    logger.info(
        "LLM: rotación %s -> %s",
        config.llm.rotation,
        ", ".join(f"{p.provider_name}/{p.model}" for p in ordenados),
    )

    if len(ordenados) == 1:
        return ordenados[0]

    return HybridClient(
        providers=ordenados,
        timeouts=timeouts,
        switch_after_slow_ms=float(config.llm.switch_after_slow_ms),
        slow_trip_count=config.llm.slow_trip_count,
        failure_trip_count=config.llm.failure_trip_count,
        cooldown_s=config.llm.cooldown_s,
        rotation=config.llm.rotation,
    )
