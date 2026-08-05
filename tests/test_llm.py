from unittest import mock

import httpx
import pytest

from brain.llm import LlmResponse, OllamaClient, OllamaError


def _mock_httpx_response(data: dict) -> mock.AsyncMock:
    resp = mock.AsyncMock()
    resp.json = mock.MagicMock(return_value=data)
    resp.raise_for_status = mock.MagicMock()
    return resp


class TestOllamaClient:
    @pytest.mark.asyncio
    async def test_generate_ok(self) -> None:
        client = OllamaClient(keep_alive="0m")
        mock_data = {
            "response": "Hola, en que te ayudo?",
            "eval_count": 25,
            "eval_duration": 500_000_000,
            "model": "llama3.2:3b",
        }

        mock_resp = _mock_httpx_response(mock_data)
        with mock.patch("httpx.AsyncClient.post", return_value=mock_resp):
            result = await client.generate("llama3.2:3b", "Hola")

        assert result.text == "Hola, en que te ayudo?"
        assert result.total_tokens == 25
        assert result.tokens_per_second > 0

    @pytest.mark.asyncio
    async def test_connect_error_clear(self) -> None:
        client = OllamaClient(base_url="http://localhost:99999", keep_alive="0m")
        with mock.patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OllamaError, match="conectar"):
                await client.generate("m", "p")

    @pytest.mark.asyncio
    async def test_timeout_clear(self) -> None:
        client = OllamaClient(timeout=1.0, keep_alive="0m")
        with mock.patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("timeout")):
            with pytest.raises(OllamaError, match="Timeout"):
                await client.generate("m", "p")

    @pytest.mark.asyncio
    async def test_keep_alive_in_payload(self) -> None:
        client = OllamaClient(keep_alive="30m")
        mock_resp = _mock_httpx_response({"response": "ok", "eval_count": 1, "eval_duration": 1e9})

        with mock.patch("httpx.AsyncClient.post", return_value=mock_resp) as post_mock:
            await client.generate("m", "p")
            payload = post_mock.call_args.kwargs["json"]
            assert payload["keep_alive"] == "30m"

    def test_llm_response_dataclass(self) -> None:
        r = LlmResponse(text="hola", tokens_per_second=12.5, total_tokens=10)
        assert r.text == "hola"
