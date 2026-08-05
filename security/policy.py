import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

logger = logging.getLogger("jarvis.security.policy")


class PermissionLevel(str, Enum):
    SAFE = "SAFE"
    SYSTEM = "SYSTEM"
    SENSITIVE = "SENSITIVE"
    FORBIDDEN = "FORBIDDEN"


class DecisionType(str, Enum):
    ALLOW = "ALLOW"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    DENY = "DENY"


@dataclass
class Decision:
    type: DecisionType
    reason: str = ""
    effective_level: PermissionLevel | None = None
    resolved_params: dict[str, Any] = field(default_factory=dict)


class PolicyError(Exception):
    """Error de configuración o evaluación de política de permisos."""


class Policy:
    _KNOWN_CONDITIONS: set[str] = {"not_in_whitelist", "scheme_not_allowed"}
    _LEVEL_ORDER: dict[PermissionLevel, int] = {
        PermissionLevel.SAFE: 0,
        PermissionLevel.SYSTEM: 1,
        PermissionLevel.SENSITIVE: 2,
        PermissionLevel.FORBIDDEN: 3,
    }

    def __init__(self, permissions_path: str, whitelist_apps: dict[str, str] | None = None) -> None:
        self._permissions = self._load_and_validate(permissions_path)
        self._whitelist_apps = whitelist_apps or {}

    def _load_and_validate(self, path: str) -> dict[str, Any]:
        full_path = Path(path)
        if not full_path.exists():
            raise PolicyError(f"Archivo de permisos no encontrado: {full_path.resolve()}")

        raw = yaml.safe_load(full_path.read_text(encoding="utf-8"))

        skills = raw.get("skills", {}) if isinstance(raw, dict) else {}
        for skill_name, operations in skills.items():
            if not isinstance(operations, dict):
                continue
            for op_name, op_config in operations.items():
                if not isinstance(op_config, dict):
                    continue
                for rule in op_config.get("escalate", []):
                    condition = rule.get("condition")
                    if condition not in self._KNOWN_CONDITIONS:
                        raise PolicyError(
                            f"Condición desconocida '{condition}' en "
                            f"skills.{skill_name}.{op_name}.escalate. "
                            f"Catálogo permitido: {sorted(self._KNOWN_CONDITIONS)}"
                        )

        return skills

    def check(self, skill_name: str, operation: str, params: dict[str, Any] | None = None) -> Decision:
        if params is None:
            params = {}

        op_config = self._get_operation_config(skill_name, operation)
        if op_config is None:
            reason = f"Operación no autorizada: {skill_name}.{operation}"
            logger.warning("DENY: %s", reason)
            return Decision(type=DecisionType.DENY, reason=reason, effective_level=PermissionLevel.FORBIDDEN)

        base_level = PermissionLevel(op_config["level"])

        if base_level == PermissionLevel.FORBIDDEN:
            reason = f"Operación prohibida por política: {skill_name}.{operation}"
            logger.warning("DENY: %s", reason)
            return Decision(type=DecisionType.DENY, reason=reason, effective_level=PermissionLevel.FORBIDDEN)

        resolved = dict(params)
        escalate_rules = op_config.get("escalate", [])
        self._resolve_aliases(escalate_rules, resolved)

        escalated = self._evaluate_escalations(escalate_rules, resolved, base_level)

        if escalated == PermissionLevel.FORBIDDEN:
            reason = f"Operación escalada a FORBIDDEN: {skill_name}.{operation}"
            logger.warning("DENY: %s", reason)
            return Decision(
                type=DecisionType.DENY,
                reason=reason,
                effective_level=PermissionLevel.FORBIDDEN,
                resolved_params=resolved,
            )

        if escalated == PermissionLevel.SENSITIVE:
            return Decision(
                type=DecisionType.NEEDS_CONFIRMATION,
                reason="Operación sensible requiere confirmación por PIN",
                effective_level=PermissionLevel.SENSITIVE,
                resolved_params=resolved,
            )

        return Decision(type=DecisionType.ALLOW, reason="", effective_level=escalated, resolved_params=resolved)

    def _get_operation_config(self, skill_name: str, operation: str) -> dict[str, Any] | None:
        skill = self._permissions.get(skill_name)
        if not isinstance(skill, dict):
            return None
        return skill.get(operation)

    def _evaluate_escalations(
        self, rules: list[dict[str, Any]], params: dict[str, Any], current: PermissionLevel
    ) -> PermissionLevel:
        final = current
        for rule in rules:
            condition = rule["condition"]
            param_name = rule["param"]
            to_level = PermissionLevel(rule["to"])

            param_value = params.get(param_name)

            triggered = False
            if condition == "not_in_whitelist":
                triggered = self._check_not_in_whitelist(param_value)
            elif condition == "scheme_not_allowed":
                allowed = rule.get("allowed_schemes", ["http", "https"])
                triggered = self._check_scheme_not_allowed(param_value, allowed)

            if triggered:
                final = self._most_restrictive(final, to_level)

        return final

    def _check_not_in_whitelist(self, value: Any) -> bool:
        if value is None:
            return True
        return str(value) not in self._whitelist_apps.values()

    def _check_scheme_not_allowed(self, url_value: Any, allowed: list[str]) -> bool:
        try:
            parsed = urlparse(str(url_value))
            return parsed.scheme not in allowed
        except Exception:
            return True

    def _most_restrictive(self, a: PermissionLevel, b: PermissionLevel) -> PermissionLevel:
        return b if self._LEVEL_ORDER[b] > self._LEVEL_ORDER[a] else a

    def _resolve_aliases(self, rules: list[dict[str, Any]], params: dict[str, Any]) -> None:
        for rule in rules:
            if rule.get("condition") == "not_in_whitelist":
                param_name = rule["param"]
                alias = params.get(param_name)
                if alias is not None and alias in self._whitelist_apps:
                    params[param_name] = self._whitelist_apps[alias]

    def resolve_alias(self, app_alias: str) -> str:
        if app_alias not in self._whitelist_apps:
            raise PolicyError(f"Alias no reconocido: '{app_alias}'. No está en la whitelist de apps.")
        return self._whitelist_apps[app_alias]
