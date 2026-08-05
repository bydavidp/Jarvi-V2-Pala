import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx

logger = logging.getLogger("jarvis.brain.llm")


@dataclass
class LlmResponse:
    text: str
    tokens_per_second: float = 0.0
    total_tokens: int = 0
    model: str = ""
    eval_count: int = 0
    eval_duration_ns: int = 0


class OllamaError(Exception):
    """Error de conexión o respuesta del servicio Ollama."""


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        keep_alive: str = "30m",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._keep_alive = keep_alive

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        num_ctx: int = 4096,
    ) -> LlmResponse:
        url = f"{self._base_url}/api/generate"
        payload: dict[str, Any] = {
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

        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)

        return LlmResponse(
            text=data.get("response", "").strip(),
            tokens_per_second=eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0,
            total_tokens=eval_count,
            model=data.get("model", model),
            eval_count=eval_count,
            eval_duration_ns=eval_duration_ns,
        )

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncGenerator[str, None]:
        url = f"{self._base_url}/api/generate"
        payload: dict[str, Any] = {
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
                        if line:
                            try:
                                chunk = __import__("json").loads(line)
                                token = chunk.get("response", "")
                                if token:
                                    yield token
                            except Exception:
                                continue
        except httpx.ConnectError:
            raise OllamaError(
                f"No se pudo conectar a Ollama en {self._base_url}."
            ) from None
