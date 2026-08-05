import inspect
import logging
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any

from security.audit import AuditLog
from security.policy import DecisionType, Policy

logger = logging.getLogger("jarvis.skills")

_AUTH_TOKEN_PARAM = "_auth_token"


@dataclass
class SkillResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    needs_confirmation: bool = False


class SkillAuthError(Exception):
    """Intento de ejecutar una skill sin token de autorización válido."""


class Skill:
    name: str = ""
    description: str = ""
    level: str = ""

    def __init__(self) -> None:
        self._registry: Any = None

    def _bind_registry(self, registry: "SkillRegistry") -> None:
        self._registry = registry

    async def _execute(self, _auth_token: str = "", **params: Any) -> SkillResult:
        if self._registry is None:
            logger.error("Skill '%s' ejecutada sin registry enlazado", self.name)
            raise SkillAuthError(
                f"Skill '{self.name}' no tiene registry enlazado. "
                "Usar registry.execute()."
            )

        if not self._registry.validate_token(self.name, _auth_token):
            self._registry.audit_bypass_attempt(self.name, params)
            logger.warning("Intento de bypass detectado en skill '%s'", self.name)
            raise SkillAuthError(
                f"Skill '{self.name}' ejecutada sin token de autorización válido. "
                "Usar registry.execute()."
            )

        return await self._do_execute(**params)

    async def _do_execute(self, **params: Any) -> SkillResult:
        raise NotImplementedError


class SkillRegistry:
    def __init__(self, policy: Policy, audit: AuditLog | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        self._policy = policy
        self._audit = audit
        self._tokens: dict[str, str] = {}
        self._token_lock = threading.Lock()

    def register(self, skill: Skill) -> None:
        self._validate_skill_signature(skill)
        self._skills[skill.name] = skill
        skill._bind_registry(self)
        logger.info("Skill registrada: %s (nivel %s)", skill.name, skill.level)

    def _validate_skill_signature(self, skill: Skill) -> None:
        sig = inspect.signature(skill._do_execute)
        for name in sig.parameters:
            if name == _AUTH_TOKEN_PARAM:
                raise ValueError(
                    f"Skill '{skill.name}' declara el parámetro reservado "
                    f"'{_AUTH_TOKEN_PARAM}' en _do_execute(). "
                    "Renombra el parámetro de la skill."
                )

    def _issue_token(self, skill_name: str) -> str:
        token = secrets.token_hex(16)
        with self._token_lock:
            self._tokens[skill_name] = token
        return token

    def validate_token(self, skill_name: str, token: str) -> bool:
        with self._token_lock:
            expected = self._tokens.get(skill_name)
            if expected is None:
                return False
            if not secrets.compare_digest(expected, token):
                return False
            del self._tokens[skill_name]
            return True

    def audit_bypass_attempt(self, skill_name: str, params: dict[str, Any]) -> None:
        if self._audit is not None:
            self._audit.log(
                skill_name, "_execute", params,
                "DENY", "bypass_attempt",
                "Intento de ejecución directa sin token válido",
            )

    async def execute(
        self,
        skill_name: str,
        operation: str,
        params: dict[str, Any] | None = None,
    ) -> SkillResult:
        skill = self._skills.get(skill_name)
        if skill is None:
            return SkillResult(success=False, error=f"Skill desconocida: {skill_name}")

        decision = self._policy.check(skill_name, operation, params)

        if decision.type == DecisionType.DENY:
            self._audit_log(skill_name, operation, params, decision.type.value, "blocked", decision.reason)
            logger.warning("Skill bloqueada: %s.%s → %s", skill_name, operation, decision.reason)
            return SkillResult(success=False, error=decision.reason)

        if decision.type == DecisionType.NEEDS_CONFIRMATION:
            return SkillResult(
                success=False,
                error="Operación sensible: requiere confirmación por PIN",
                needs_confirmation=True,
            )

        token = self._issue_token(skill_name)
        result = await skill._execute(_auth_token=token, **decision.resolved_params)

        result_label = "executed" if result.success else "failed"
        self._audit_log(
            skill_name, operation, decision.resolved_params,
            decision.type.value, result_label, result.error,
        )
        return result

    def _audit_log(
        self,
        skill: str,
        operation: str,
        params: dict[str, Any] | None,
        decision: str,
        result: str,
        details: str = "",
    ) -> None:
        if self._audit is not None:
            self._audit.log(skill, operation, params, decision, result, details)

    def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())
