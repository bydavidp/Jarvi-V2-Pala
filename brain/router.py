import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from brain.llm import LlmResponse, OllamaClient, OllamaError
from brain.prompts import ROUTER_SYSTEM

logger = logging.getLogger("jarvis.brain.router")


class IntentType(str, Enum):
    CHAT = "CHAT"
    SKILL = "SKILL"
    SEARCH = "SEARCH"


@dataclass
class RouterDecision:
    intent: IntentType
    skill_name: str = ""
    operation: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    search_query: str = ""
    resolved_by: str = ""  # "matcher" o "llm"
    latency_ms: float = 0.0
    llm_response: LlmResponse | None = None


class RouterOutput(BaseModel):
    intent: str
    skill: str = ""
    operation: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    search_query: str = ""


def _load_patterns(path: str) -> list[dict[str, Any]]:
    full = Path(path)
    if full.exists():
        raw = yaml.safe_load(full.read_text(encoding="utf-8"))
        return raw.get("patterns", []) if isinstance(raw, dict) else []
    return []


class Router:
    def __init__(
        self,
        ollama: OllamaClient,
        patterns_path: str = "brain/patterns.yaml",
        model: str = "llama3.2:3b",
        classifier: Any = None,
    ) -> None:
        self._ollama = ollama
        self._patterns = _load_patterns(patterns_path)
        self._model = model
        self._classifier = classifier

    async def route(self, user_text: str) -> RouterDecision:
        t0 = time.perf_counter()

        decision = self._matcher_route(user_text)
        if decision is not None:
            decision.latency_ms = (time.perf_counter() - t0) * 1000
            decision.resolved_by = "matcher"
            return decision

        if self._classifier is not None:
            cls_result = self._classifier.classify(user_text)
            if cls_result is not None:
                return RouterDecision(
                    intent=IntentType.SKILL,
                    skill_name=cls_result["skill"],
                    operation=cls_result["operation"],
                    params={},
                    resolved_by="classifier",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        decision = await self._llm_route(user_text)
        decision.latency_ms = (time.perf_counter() - t0) * 1000
        decision.resolved_by = "llm"
        return decision

    def _matcher_route(self, text: str) -> RouterDecision | None:
        for pattern in self._patterns:
            regex = pattern.get("regex", "")
            if not regex:
                continue
            match = re.search(regex, text)
            if not match:
                continue

            skill = pattern.get("skill", "")
            operation = pattern.get("operation", "")
            param_specs: dict[str, dict] = pattern.get("params", {})

            extracted: dict[str, Any] = {}
            for param_name, spec in param_specs.items():
                group_ref = spec.get("group") if isinstance(spec, dict) else spec
                type_name = spec.get("type", "str") if isinstance(spec, dict) else "str"

                if isinstance(group_ref, int):
                    value = match.group(group_ref)
                else:
                    value = match.group(str(group_ref))

                if value is None or value.strip() == "":
                    logger.info("Matcher: match pero param '%s' vacío, degradando", param_name)
                    return None

                try:
                    value = value.strip()
                    if type_name == "int":
                        value = int(value)
                    elif type_name == "float":
                        value = float(value)
                    # str stays as-is
                    extracted[param_name] = value
                except (ValueError, TypeError):
                    logger.info("Matcher: fallo conversión '%s' a %s, degradando", value, type_name)
                    return None

            logger.info("Matcher: %s -> %s.%s params=%s", text[:60], skill, operation, extracted)
            return RouterDecision(
                intent=IntentType.SKILL,
                skill_name=skill,
                operation=operation,
                params=extracted,
            )
        return None

    async def _llm_route(self, user_text: str) -> RouterDecision:
        system = ROUTER_SYSTEM.format(user_text=user_text)

        for attempt in (1, 2):
            try:
                response = await self._ollama.generate(
                    model=self._model,
                    prompt="JSON:",
                    system=system,
                    temperature=0.1,
                    max_tokens=128,
                    num_ctx=512,
                )
            except OllamaError as e:
                logger.error("Error de Ollama en routing (intento %d): %s", attempt, e)
                return RouterDecision(intent=IntentType.CHAT, resolved_by="llm")

            try:
                parsed = self._parse_json(response.text)
            except ValueError as e:
                logger.warning(
                    "JSON inválido del LLM (intento %d): %s\nRespuesta: %s",
                    attempt, e, response.text[:200],
                )
                if attempt == 2:
                    logger.error("Dos intentos fallidos de JSON. Degradando a CHAT.")
                    return RouterDecision(intent=IntentType.CHAT, resolved_by="llm")
                continue

            intent = parsed.intent.upper()
            if intent not in ("CHAT", "SKILL", "SEARCH"):
                intent = "CHAT"

            return RouterDecision(
                intent=IntentType(intent),
                skill_name=parsed.skill if intent == "SKILL" else "",
                operation=parsed.operation if intent == "SKILL" else "",
                params=parsed.params if intent == "SKILL" else {},
                search_query=parsed.search_query if intent == "SEARCH" else "",
                llm_response=response if attempt == 1 else None,
            )

        return RouterDecision(intent=IntentType.CHAT, resolved_by="llm")

    def _parse_json(self, text: str) -> RouterOutput:
        cleaned = text.strip()
        for opener, closer in [("{", "}"), ("```json", "```"), ("```", "```")]:
            if opener in cleaned:
                start = cleaned.index(opener)
                end = cleaned.rindex(closer) + len(closer)
                cleaned = cleaned[start:end]
                break
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        try:
            return RouterOutput.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError) as e:
            raise ValueError(f"No se pudo parsear JSON: {e}") from e
