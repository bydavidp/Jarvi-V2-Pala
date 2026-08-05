import json
import secrets
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest
import yaml

from security.audit import AuditLog, REDACTED
from security.confirm import ConfirmResult, PinVerifier
from security.policy import (
    Decision,
    DecisionType,
    PermissionLevel,
    Policy,
    PolicyError,
)

WHITELIST_APPS = {
    "navegador": "chrome",
    "spotify": "spotify",
    "vscode": "code",
    "calculadora": "calc",
}


# ─── helpers ────────────────────────────────────────────────────────────

def _make_permissions_yaml(tmp_path: Path, content: dict) -> str:
    path = tmp_path / "permissions.yaml"
    path.write_text(yaml.dump(content), encoding="utf-8")
    return str(path)


def _assert_deny(decision: Decision, *, reason_contains: str = "") -> None:
    assert decision.type == DecisionType.DENY
    if reason_contains:
        assert reason_contains in decision.reason


def _assert_allow(decision: Decision) -> None:
    assert decision.type == DecisionType.ALLOW


def _assert_needs_confirmation(decision: Decision) -> None:
    assert decision.type == DecisionType.NEEDS_CONFIRMATION


# ─── Policy ──────────────────────────────────────────────────────────────

class TestPolicyBasic:
    def test_safe_returns_allow(self) -> None:
        policy = Policy("config/permissions.yaml")
        decision = policy.check("time", "get_current_time")
        _assert_allow(decision)
        assert decision.effective_level == PermissionLevel.SAFE

    def test_system_returns_allow(self) -> None:
        policy = Policy("config/permissions.yaml")
        decision = policy.check("volume", "set", {"level": 50})
        _assert_allow(decision)
        assert decision.effective_level == PermissionLevel.SYSTEM

    def test_sensitive_returns_needs_confirmation(self) -> None:
        policy = Policy("config/permissions.yaml")
        decision = policy.check("devices", "discover")
        _assert_needs_confirmation(decision)
        assert decision.effective_level == PermissionLevel.SENSITIVE

    def test_unknown_skill_is_deny(self) -> None:
        policy = Policy("config/permissions.yaml")
        decision = policy.check("hack", "root")
        _assert_deny(decision, reason_contains="no autorizada")

    def test_unknown_operation_is_deny(self) -> None:
        policy = Policy("config/permissions.yaml")
        decision = policy.check("time", "set_timezone")
        _assert_deny(decision, reason_contains="no autorizada")


class TestPolicyEscalation:
    def test_not_in_whitelist_escalates_to_forbidden(self) -> None:
        policy = Policy("config/permissions.yaml", whitelist_apps=WHITELIST_APPS)
        decision = policy.check("apps", "open", {"app_name": "cmd.exe"})
        _assert_deny(decision, reason_contains="FORBIDDEN")
        assert decision.effective_level == PermissionLevel.FORBIDDEN

    def test_in_whitelist_passes(self) -> None:
        policy = Policy("config/permissions.yaml", whitelist_apps=WHITELIST_APPS)
        decision = policy.check("apps", "open", {"app_name": "chrome"})
        _assert_allow(decision)

    def test_alias_resuelto_viaja_en_decision(self) -> None:
        """El alias 'navegador' se resuelve a 'chrome' dentro de Decision."""
        policy = Policy("config/permissions.yaml", whitelist_apps=WHITELIST_APPS)
        decision = policy.check("apps", "open", {"app_name": "navegador"})
        _assert_allow(decision)
        assert decision.resolved_params["app_name"] == "chrome"

    def test_alias_fuera_de_whitelist_es_deny(self) -> None:
        """Alias no reconocido no resuelve y escala a FORBIDDEN."""
        policy = Policy("config/permissions.yaml", whitelist_apps=WHITELIST_APPS)
        decision = policy.check("apps", "open", {"app_name": "internet_explorer"})
        _assert_deny(decision, reason_contains="FORBIDDEN")
        # El resolved_params contiene el alias original sin resolver
        assert decision.resolved_params["app_name"] == "internet_explorer"

    def test_scheme_not_allowed_blocks_ftp(self) -> None:
        policy = Policy("config/permissions.yaml")
        decision = policy.check("browser", "open_url", {"url": "ftp://malware.exe"})
        _assert_deny(decision, reason_contains="FORBIDDEN")

    def test_scheme_allowed_passes(self) -> None:
        policy = Policy("config/permissions.yaml")
        decision = policy.check("browser", "open_url", {"url": "https://google.com"})
        _assert_allow(decision)

    def test_scheme_none_escalates(self) -> None:
        policy = Policy("config/permissions.yaml")
        decision = policy.check("browser", "open_url", {"url": "not-a-url"})
        _assert_deny(decision)

    def test_multiple_escalations_take_most_restrictive(self, tmp_path: Path) -> None:
        permissions = {
            "skills": {
                "test_skill": {
                    "test_op": {
                        "level": "SAFE",
                        "escalate": [
                            {"condition": "not_in_whitelist", "param": "app_name", "to": "SYSTEM"},
                            {"condition": "scheme_not_allowed", "param": "url", "allowed_schemes": ["http"], "to": "FORBIDDEN"},
                        ],
                    }
                }
            }
        }
        path = _make_permissions_yaml(tmp_path, permissions)
        policy = Policy(path, whitelist_apps={"chrome": "chrome"})

        decision = policy.check("test_skill", "test_op", {"app_name": "cmd.exe", "url": "ftp://x"})
        _assert_deny(decision)
        assert decision.effective_level == PermissionLevel.FORBIDDEN


class TestPolicyFailClosed:
    def test_param_missing_scales_anyway(self) -> None:
        """Si una regla declara param: X y la llamada no incluye X, escala."""
        policy = Policy("config/permissions.yaml", whitelist_apps=WHITELIST_APPS)
        decision = policy.check("apps", "open", {})
        # app_name ausente → None → not_in_whitelist → FORBIDDEN
        _assert_deny(decision, reason_contains="FORBIDDEN")

    def test_unknown_condition_crashes_at_load(self, tmp_path: Path) -> None:
        permissions = {
            "skills": {
                "test": {
                    "op": {
                        "level": "SAFE",
                        "escalate": [
                            {"condition": "condicion_que_no_existe", "param": "x", "to": "FORBIDDEN"}
                        ],
                    }
                }
            }
        }
        path = _make_permissions_yaml(tmp_path, permissions)
        with pytest.raises(PolicyError, match="Condición desconocida"):
            Policy(path)


class TestPolicyAlias:
    def test_resolve_alias_returns_executable(self) -> None:
        policy = Policy("config/permissions.yaml", whitelist_apps=WHITELIST_APPS)
        assert policy.resolve_alias("navegador") == "chrome"

    def test_resolve_alias_unknown_raises(self) -> None:
        policy = Policy("config/permissions.yaml", whitelist_apps=WHITELIST_APPS)
        with pytest.raises(PolicyError, match="Alias no reconocido"):
            policy.resolve_alias("internet_explorer")


# ─── AuditLog ────────────────────────────────────────────────────────────

class TestAuditLog:
    @pytest.fixture
    def audit(self, tmp_path: Path) -> AuditLog:
        db = str(tmp_path / "audit.db")
        return AuditLog(db)

    def test_log_allowed(self, audit: AuditLog) -> None:
        audit.log("volume", "set", {"level": 50}, "ALLOW", "executed")
        entries = audit.get_recent()
        assert len(entries) == 1
        assert entries[0]["skill"] == "volume"
        assert entries[0]["decision"] == "ALLOW"

    def test_log_deny(self, audit: AuditLog) -> None:
        audit.log("apps", "open", {"app_name": "cmd.exe"}, "DENY", "blocked",
                   "app fuera de whitelist")
        entries = audit.get_recent()
        assert len(entries) == 1
        assert entries[0]["result"] == "blocked"
        assert entries[0]["decision"] == "DENY"

    def test_log_needs_confirmation(self, audit: AuditLog) -> None:
        audit.log("devices", "discover", {}, "NEEDS_CONFIRMATION", "confirmed",
                   "PIN validado")
        entries = audit.get_recent()
        assert len(entries) == 1
        assert entries[0]["decision"] == "NEEDS_CONFIRMATION"
        assert entries[0]["result"] == "confirmed"

    def test_params_sanitize_sensitive_keys(self, audit: AuditLog) -> None:
        audit.log("test", "op", {"pin": "1234", "api_token": "abc", "normal": "ok"},
                   "ALLOW", "executed")
        entries = audit.get_recent()
        params = json.loads(entries[0]["params"])
        assert params["pin"] == REDACTED
        assert params["api_token"] == REDACTED
        assert params["normal"] == "ok"

    def test_params_truncate_long_values(self, audit: AuditLog) -> None:
        long_value = "x" * 300
        audit.log("test", "op", {"data": long_value}, "ALLOW", "executed")
        entries = audit.get_recent()
        params = json.loads(entries[0]["params"])
        assert len(params["data"]) == 203  # 200 + "..."
        assert params["data"].endswith("...")

    def test_pin_never_in_log(self, audit: AuditLog) -> None:
        audit.log("confirm", "verify_pin", {"pin": "9999"}, "DENY", "blocked")
        entries = audit.get_recent()
        params_raw = entries[0]["params"]
        assert "9999" not in params_raw
        assert REDACTED in params_raw

    def test_update_fails(self, audit: AuditLog) -> None:
        audit.log("test", "op", {}, "ALLOW", "executed")
        with pytest.raises(sqlite3.IntegrityError, match="UPDATE"):
            with sqlite3.connect(str(audit._db_path)) as conn:
                conn.execute("UPDATE audit_log SET result = 'hacked' WHERE id = 1")
                conn.commit()

    def test_delete_fails(self, audit: AuditLog) -> None:
        audit.log("test", "op", {}, "ALLOW", "executed")
        with pytest.raises(sqlite3.IntegrityError, match="DELETE"):
            with sqlite3.connect(str(audit._db_path)) as conn:
                conn.execute("DELETE FROM audit_log WHERE id = 1")
                conn.commit()

    def test_audit_covers_multiple_operations(self, audit: AuditLog) -> None:
        audit.log("volume", "set", {"level": 50}, "ALLOW", "executed")
        audit.log("apps", "open", {"app_name": "cmd.exe"}, "DENY", "blocked")
        audit.log("devices", "discover", {}, "NEEDS_CONFIRMATION", "pending")
        assert audit.count_by_skill("volume") == 1
        assert audit.count_by_skill("apps") == 1
        assert len(audit.get_recent()) == 3


# ─── PinVerifier ─────────────────────────────────────────────────────────

class TestPinVerifier:
    def test_correct_pin_approved(self) -> None:
        verifier = PinVerifier(pin="1234")
        assert verifier.verify("1234") == ConfirmResult.APPROVED

    def test_incorrect_pin_denied(self) -> None:
        verifier = PinVerifier(pin="1234")
        assert verifier.verify("0000") == ConfirmResult.DENIED

    def test_empty_pin_raises(self) -> None:
        with pytest.raises(ValueError, match="vacío"):
            PinVerifier(pin="")

    def test_uses_compare_digest(self) -> None:
        verifier = PinVerifier(pin="abcd")
        with mock.patch.object(secrets, "compare_digest", wraps=secrets.compare_digest) as spy:
            verifier.verify("abcd")
            spy.assert_called_once_with("abcd", "abcd")

    def test_forbidden_not_reversed_by_valid_pin(self) -> None:
        policy = Policy("config/permissions.yaml")
        decision = policy.check("hack", "root")
        assert decision.type == DecisionType.DENY
        # Incluso si el PIN es correcto, un DENY no se revierte
        verifier = PinVerifier(pin="1234")
        assert verifier.verify("1234") == ConfirmResult.APPROVED
        assert decision.type == DecisionType.DENY

    def test_remaining_attempts(self) -> None:
        verifier = PinVerifier(pin="1234")
        assert verifier.remaining_attempts() == 3
        verifier.verify("0000")
        assert verifier.remaining_attempts() == 2

    def test_max_attempts_triggers_cooldown(self) -> None:
        verifier = PinVerifier(pin="1234", max_attempts=2, cooldown_seconds=10.0)
        assert verifier.verify("0000") == ConfirmResult.DENIED
        assert verifier.verify("0000") == ConfirmResult.DENIED
        assert verifier.remaining_attempts() == 0
        assert verifier.verify("0000") == ConfirmResult.PIN_EXPIRED

    def test_correct_pin_resets_attempts(self) -> None:
        verifier = PinVerifier(pin="1234", max_attempts=3)
        verifier.verify("0000")
        verifier.verify("0000")
        assert verifier.remaining_attempts() == 1
        verifier.verify("1234")
        assert verifier.remaining_attempts() == 3


# ─── Integración ─────────────────────────────────────────────────────────

class TestIntegration:
    def test_audit_logs_deny_decisions(self, tmp_path: Path) -> None:
        db = str(tmp_path / "audit.db")
        audit = AuditLog(db)
        policy = Policy("config/permissions.yaml")

        decision = policy.check("hack", "root")
        assert decision.type == DecisionType.DENY
        audit.log("hack", "root", {}, decision.type.value, "blocked",
                   decision.reason)

        entries = audit.get_recent()
        assert len(entries) == 1
        assert entries[0]["decision"] == "DENY"

    def test_audit_logs_failed_pin_attempts(self, tmp_path: Path) -> None:
        db = str(tmp_path / "audit.db")
        audit = AuditLog(db)
        verifier = PinVerifier(pin="1234")

        verifier.verify("0000")
        audit.log("confirm", "verify_pin", {"pin": "0000"},
                   "NEEDS_CONFIRMATION", "pin_denied", "PIN incorrecto")

        entries = audit.get_recent()
        assert len(entries) == 1
        assert entries[0]["result"] == "pin_denied"

    def test_sensitive_flow_with_pin(self, tmp_path: Path) -> None:
        db = str(tmp_path / "audit.db")
        audit = AuditLog(db)
        policy = Policy("config/permissions.yaml")
        verifier = PinVerifier(pin="1234")

        decision = policy.check("devices", "discover")
        assert decision.type == DecisionType.NEEDS_CONFIRMATION

        audit.log("devices", "discover", {},
                   decision.type.value, "pending")

        result = verifier.verify("1234")
        assert result == ConfirmResult.APPROVED

        audit.log("devices", "discover", {},
                   decision.type.value, "confirmed", "PIN validado")

        entries = audit.get_recent()
        assert len(entries) == 2
        assert entries[0]["result"] == "confirmed"
        assert entries[1]["result"] == "pending"
