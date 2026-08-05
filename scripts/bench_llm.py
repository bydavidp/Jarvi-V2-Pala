"""Comparativa de latencia router + chat entre modelos Ollama."""
import asyncio
import time

from brain.llm import OllamaClient
from brain.prompts import CHAT_PROMPT, SYSTEM_PROMPT
from brain.router import Router


async def measure(model: str, rounds: int = 3) -> dict:
    client = OllamaClient(keep_alive="30m")
    router = Router(client, model=model)

    print(f"\n{'='*50}")
    print(f"Modelo: {model}")
    print(f"{'='*50}")

    # Warmup
    print("  Calentando...")
    await client.generate(model, "Hola", system=SYSTEM_PROMPT, max_tokens=10)

    # Test 1: matcher
    t0 = time.perf_counter()
    d = await router.route("que hora es")
    matcher_ms = d.latency_ms
    print(f"  Matcher (que hora es): {matcher_ms:.0f}ms -> {d.resolved_by}")

    # Test 2: chat corto
    t0 = time.perf_counter()
    resp = await client.generate(model, CHAT_PROMPT.format(user_text="Hola, como estas?"),
                                  system=SYSTEM_PROMPT, max_tokens=100)
    chat_ms = (time.perf_counter() - t0) * 1000
    print(f"  Chat (hola):          {chat_ms:.0f}ms {resp.total_tokens} tok @ {resp.tokens_per_second:.0f} tok/s")

    # Test 3: routing LLM
    t0 = time.perf_counter()
    d = await router.route("que opinas de Python como lenguaje de programacion")
    route_ms = d.latency_ms
    print(f"  Router LLM:           {route_ms:.0f}ms -> {d.intent.value}")

    return {
        "model": model,
        "matcher_ms": matcher_ms,
        "chat_ms": chat_ms,
        "route_ms": route_ms,
        "tok_s": resp.tokens_per_second,
    }


async def main():
    results = []
    for model in ["llama3.2:latest", "llama3.1:latest"]:
        r = await measure(model)
        results.append(r)

    print(f"\n{'='*50}")
    print("RESUMEN")
    print(f"{'='*50}")
    print(f"{'Modelo':<20s} {'Matcher':>8s} {'Router LLM':>10s} {'Chat':>8s} {'tok/s':>8s}")
    for r in results:
        print(f"{r['model']:<20s} {r['matcher_ms']:7.0f}ms {r['route_ms']:9.0f}ms {r['chat_ms']:7.0f}ms {r['tok_s']:7.0f}")


asyncio.run(main())
