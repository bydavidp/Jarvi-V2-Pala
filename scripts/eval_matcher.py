"""Evaluacion honesta del matcher con frases Colombianas reales."""
import asyncio
from unittest import mock

from brain.llm import OllamaClient
from brain.router import Router
from tests.test_honesto import PHRASES


async def main():
    client = mock.MagicMock(spec=OllamaClient)

    async def fake_generate(**kwargs):
        from brain.llm import LlmResponse
        return LlmResponse(text='{"intent": "CHAT"}')

    client.generate = fake_generate
    router = Router(client)

    ok = 0
    for phrase, exp_skill, exp_op in PHRASES:
        d = await router.route(phrase)
        hit = d.resolved_by == "matcher" and d.skill_name == exp_skill and d.operation == exp_op
        status = "OK " if hit else "FAIL"
        print(f"  {status} '{phrase}' -> {d.resolved_by}/{d.skill_name}.{d.operation} (esperado {exp_skill}.{exp_op})")
        ok += 1 if hit else 0

    print(f"\n{ok}/{len(PHRASES)} = {ok/len(PHRASES)*100:.0f}%")


asyncio.run(main())
