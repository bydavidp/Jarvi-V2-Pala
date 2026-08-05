#!/usr/bin/env python3
"""REPL de chat por consola. Muestra etapa, intención y latencia por turno.

Uso:
    python3.14 scripts/chat.py
    python3.14 scripts/chat.py --model llama3.1:8b
"""

import argparse
import asyncio
import time

from brain.llm import OllamaClient, OllamaError
from brain.prompts import SYSTEM_PROMPT, CHAT_PROMPT
from brain.router import IntentType, Router


async def main() -> None:
    parser = argparse.ArgumentParser(description="Chat con Jarvis")
    parser.add_argument("--model", default="llama3.2:3b", help="Modelo de Ollama")
    parser.add_argument("--url", default="http://localhost:11434", help="URL de Ollama")
    args = parser.parse_args()

    client = OllamaClient(base_url=args.url)
    router = Router(client, model=args.model)

    history: list[str] = []

    print(f"Jarvis v0.1 — modelo: {args.model}")
    print("Escribe 'salir' para terminar.\n")

    while True:
        try:
            user = input("Tú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAdiós.")
            break

        if not user:
            continue
        if user.lower() in ("salir", "exit", "quit"):
            print("Adiós.")
            break

        t0 = time.perf_counter()

        # ── Router ──
        decision = await router.route(user)
        print(f"  [{decision.resolved_by}] intent={decision.intent.value} "
              f"lat={decision.latency_ms:.0f}ms", end="")

        if decision.intent == IntentType.SKILL:
            print(f" skill={decision.skill_name}.{decision.operation}")
            print(f"Jarvis > [SKILL] {decision.skill_name}.{decision.operation}({decision.params})")

        elif decision.intent == IntentType.SEARCH:
            print(f' query="{decision.search_query[:40]}"')
            print(f"Jarvis > [SEARCH] {decision.search_query}")

        else:
            print()
            # ── Chat ──
            context = "\n".join(history[-10:])
            prompt = CHAT_PROMPT.format(user_text=user)
            if context:
                prompt = f"Historial:\n{context}\n\n{prompt}"

            try:
                response = await client.generate(
                    model=args.model,
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                )
                history.append(f"Usuario: {user}")
                history.append(f"Jarvis: {response.text}")
                total_ms = (time.perf_counter() - t0) * 1000
                print(f"Jarvis > {response.text}")
                print(f"  [{response.tokens_per_second:.0f} tok/s, total={total_ms:.0f}ms]")
            except OllamaError as e:
                print(f"  [ERROR] {e}")

        print()


if __name__ == "__main__":
    asyncio.run(main())
