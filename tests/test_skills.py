import asyncio
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from core.orchestrator import Orchestrator as Orchestrator_
from security.audit import AuditLog
from security.policy import Policy
from skills.base import Skill, SkillAuthError, SkillRegistry, SkillResult
from skills.time_skill import TimeSkill


# ─── helpers ────────────────────────────────────────────────────────────

WHITELIST_APPS = {"navegador": "chrome"}

def _make_policy(whitelist: dict[str, str] | None = None) -> Policy:
    return Policy("config/permissions.yaml", whitelist_apps=whitelist or {})

def _make_registry(policy: Policy) -> SkillRegistry:
    registry = SkillRegistry(policy)
    registry.register(TimeSkill())
    return registry

def _make_registry_with_audit(policy: Policy, db_path: str) -> SkillRegistry:
    audit = AuditLog(db_path)
    registry = SkillRegistry(policy, audit=audit)
    registry.register(TimeSkill())
    return registry


# ─── SkillResult ─────────────────────────────────────────────────────────

class TestSkillResult:
    def test_defaults(self) -> None:
        result = SkillResult(success=True)
        assert result.success is True
        assert result.data == {}
        assert result.error == ""
        assert result.needs_confirmation is False

    def test_with_data(self) -> None:
        result = SkillResult(success=True, data={"hora": "12:00"})
        assert result.data["hora"] == "12:00"

    def test_with_error(self) -> None:
        result = SkillResult(success=False, error="falló algo")
        assert result.error == "falló algo"


# ─── Skill base ─────────────────────────────────────────────────────────

class TestSkillBase:
    def test_skill_has_metadata(self) -> None:
        skill = TimeSkill()
        assert skill.name == "time"
        assert skill.description != ""
        assert skill.level == "SAFE"

    def test_skill_no_tiene_run_publico(self) -> None:
        assert not hasattr(Skill, "run") or getattr(Skill, "run", None) is None

    def test_do_execute_no_implementado(self) -> None:
        skill = Skill()
        with pytest.raises(NotImplementedError):
            asyncio.run(skill._do_execute())

    @pytest.mark.asyncio
    async def test_execute_directo_sin_token_falla(self, tmp_path: Path) -> None:
        db = str(tmp_path / "audit.db")
        audit = AuditLog(db)
        policy = _make_policy()
        registry = SkillRegistry(policy, audit=audit)
        registry.register(TimeSkill())

        skill = registry.get_skill("time")
        assert skill is not None

        with pytest.raises(SkillAuthError, match="sin token"):
            await skill._execute()

    @pytest.mark.asyncio
    async def test_execute_directo_sin_token_auditado(self, tmp_path: Path) -> None:
        db = str(tmp_path / "audit.db")
        audit = AuditLog(db)
        policy = _make_policy()
        registry = SkillRegistry(policy, audit=audit)
        registry.register(TimeSkill())

        skill = registry.get_skill("time")
        assert skill is not None

        try:
            await skill._execute()
        except SkillAuthError:
            pass

        entries = audit.get_recent()
        assert len(entries) >= 1
        bypass = [e for e in entries if e.get("result") == "bypass_attempt"]
        assert len(bypass) == 1
        assert bypass[0]["skill"] == "time"
        assert bypass[0]["decision"] == "DENY"

    @pytest.mark.asyncio
    async def test_execute_con_token_invalido_falla(self, tmp_path: Path) -> None:
        db = str(tmp_path / "audit.db")
        audit = AuditLog(db)
        policy = _make_policy()
        registry = SkillRegistry(policy, audit=audit)
        registry.register(TimeSkill())

        skill = registry.get_skill("time")
        assert skill is not None

        with pytest.raises(SkillAuthError, match="sin token"):
            await skill._execute(_auth_token="token_falso")

    def test_registro_falla_si_skill_declara_auth_token(self) -> None:
        policy = _make_policy()
        registry = SkillRegistry(policy)

        class BadSkill(Skill):
            name = "bad"
            description = "mala"
            async def _do_execute(self, _auth_token: str = "", **params: object) -> SkillResult:
                return SkillResult(success=True)

        with pytest.raises(ValueError, match="_auth_token"):
            registry.register(BadSkill())

    @pytest.mark.asyncio
    async def test_skill_sin_conflicto_se_registra_bien(self) -> None:
        policy = _make_policy()
        registry = SkillRegistry(policy)

        class GoodSkill(Skill):
            name = "good"
            description = "buena"
            async def _do_execute(self, **params: object) -> SkillResult:
                return SkillResult(success=True)

        registry.register(GoodSkill())  # no lanza excepción
        assert "good" in registry.list_skills()


# ─── TimeSkill ──────────────────────────────────────────────────────────

class TestTimeSkill:
    @pytest.mark.asyncio
    async def test_returns_time_and_date_in_spanish(self) -> None:
        policy = _make_policy()
        registry = _make_registry(policy)
        result = await registry.execute("time", "get_current_time")

        assert result.success
        assert "hora" in result.data
        assert "fecha" in result.data
        assert "iso" in result.data
        assert any(mes in result.data["fecha"] for mes in [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
        ])
        assert any(dia in result.data["fecha"] for dia in [
            "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo",
        ])


# ─── SkillRegistry ──────────────────────────────────────────────────────

class TestSkillRegistry:
    def test_register_and_list(self) -> None:
        policy = _make_policy()
        registry = SkillRegistry(policy)
        registry.register(TimeSkill())
        assert "time" in registry.list_skills()
        assert registry.get_skill("time") is not None

    @pytest.mark.asyncio
    async def test_execute_safe_skill_allows(self) -> None:
        policy = _make_policy()
        registry = _make_registry(policy)
        result = await registry.execute("time", "get_current_time")
        assert result.success
        assert "hora" in result.data

    @pytest.mark.asyncio
    async def test_execute_unknown_skill_returns_error(self) -> None:
        policy = _make_policy()
        registry = _make_registry(policy)
        result = await registry.execute("fantasma", "run")
        assert not result.success
        assert "desconocida" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_deny_blocks_execution(self) -> None:
        policy = Policy("config/permissions.yaml", whitelist_apps=WHITELIST_APPS)
        registry = SkillRegistry(policy)

        class AppSkill(Skill):
            name = "apps"
            description = "dummy"
            async def _do_execute(self, **params: object) -> SkillResult:
                return SkillResult(success=True)

        registry.register(AppSkill())
        result = await registry.execute("apps", "open", {"app_name": "cmd.exe"})
        assert not result.success
        assert "FORBIDDEN" in result.error

    @pytest.mark.asyncio
    async def test_execute_sensitive_returns_needs_confirmation(self) -> None:
        policy = _make_policy()
        registry = SkillRegistry(policy)

        class DeviceSkill(Skill):
            name = "devices"
            description = "dummy"
            async def _do_execute(self, **params: object) -> SkillResult:
                return SkillResult(success=True)

        registry.register(DeviceSkill())
        result = await registry.execute("devices", "discover")
        assert not result.success
        assert result.needs_confirmation is True

    @pytest.mark.asyncio
    async def test_execute_siempre_llama_policy_check(self) -> None:
        policy = _make_policy()
        registry = SkillRegistry(policy)

        class SpySkill(Skill):
            name = "volume"
            description = "spy"
            async def _do_execute(self, **params: object) -> SkillResult:
                return SkillResult(success=True, data=dict(params))

        registry.register(SpySkill())

        with mock.patch.object(policy, "check", wraps=policy.check) as check_spy:
            result = await registry.execute("volume", "get")
            assert result.success
            check_spy.assert_called_once_with("volume", "get", None)


# ─── Resolved params ────────────────────────────────────────────────────

class TestResolvedParams:
    @pytest.mark.asyncio
    async def test_resolved_params_llegan_a_skill(self) -> None:
        """_do_execute recibe SIEMPRE decision.resolved_params."""
        policy = _make_policy()
        registry = SkillRegistry(policy)

        received: dict[str, object] = {}

        class SpySkill(Skill):
            name = "volume"
            description = "spy"
            async def _do_execute(self, **params: object) -> SkillResult:
                nonlocal received
                received = dict(params)
                return SkillResult(success=True)

        registry.register(SpySkill())
        await registry.execute("volume", "get", {"nivel": 77})
        assert received == {"nivel": 77}

    @pytest.mark.asyncio
    async def test_alias_resuelto_llega_a_skill(self) -> None:
        """El alias resuelto viaja de policy.check() a _do_execute()."""
        policy = Policy("config/permissions.yaml", whitelist_apps=WHITELIST_APPS)
        registry = SkillRegistry(policy)

        received: dict[str, object] = {}

        class SpySkill(Skill):
            name = "apps"
            description = "spy"
            async def _do_execute(self, **params: object) -> SkillResult:
                nonlocal received
                received = dict(params)
                return SkillResult(success=True)

        registry.register(SpySkill())
        await registry.execute("apps", "open", {"app_name": "navegador"})
        assert received["app_name"] == "chrome"

    @pytest.mark.asyncio
    async def test_params_crudos_nunca_llegan(self) -> None:
        """Con alias, _do_execute recibe el valor resuelto, no el crudo."""
        policy = Policy("config/permissions.yaml", whitelist_apps=WHITELIST_APPS)
        registry = SkillRegistry(policy)

        received: dict[str, object] = {}

        class SpySkill(Skill):
            name = "apps"
            description = "spy"
            async def _do_execute(self, **params: object) -> SkillResult:
                nonlocal received
                received = dict(params)
                return SkillResult(success=True)

        registry.register(SpySkill())
        await registry.execute("apps", "open", {"app_name": "navegador"})
        # "navegador" nunca aparece en _do_execute — solo "chrome"
        assert "navegador" not in received.values()


# ─── Concurrencia ───────────────────────────────────────────────────────

class TestConcurrency:
    @pytest.mark.asyncio
    async def test_ejecuciones_concurrentes_no_se_pisan(self) -> None:
        """N ejecuciones simultáneas: cada una con su propio token y params."""
        policy = _make_policy()
        registry = SkillRegistry(policy)

        executed: list[int] = []

        class ConcurrentSkill(Skill):
            name = "volume"
            description = "concurrent"
            async def _do_execute(self, **params: object) -> SkillResult:
                value = params.get("level")
                executed.append(int(value))
                await asyncio.sleep(0.01)
                return SkillResult(success=True, data={"received": value})

        registry.register(ConcurrentSkill())

        tasks = [
            registry.execute("volume", "get", {"level": i})
            for i in range(20)
        ]
        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            assert result.success
            assert result.data["received"] == i

        assert sorted(executed) == list(range(20))

    @pytest.mark.asyncio
    async def test_concurrente_desde_hilo_y_corutina(self) -> None:
        """Token distinto para cada invocación, incluido desde run_coroutine_threadsafe."""
        policy = _make_policy()
        registry = SkillRegistry(policy)
        loop = asyncio.get_running_loop()

        received: list[int] = []

        class ConcurrentSkill(Skill):
            name = "volume"
            description = "concurrent"
            async def _do_execute(self, **params: object) -> SkillResult:
                received.append(int(params.get("level", 0)))
                await asyncio.sleep(0.01)
                return SkillResult(success=True)

        registry.register(ConcurrentSkill())

        future = asyncio.run_coroutine_threadsafe(
            registry.execute("volume", "get", {"level": 999}), loop
        )

        async_result = await registry.execute("volume", "get", {"level": 1})

        thread_result = await asyncio.wrap_future(future)

        assert async_result.success
        assert thread_result.success
        assert sorted(received) == [1, 999]


# ─── Orchestrator (con API nueva) ──────────────────────────────────────

class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_dispatch_delega_en_registry(self) -> None:
        from unittest import mock
        policy = _make_policy()
        registry = _make_registry(policy)
        deps = {
            "capture": None, "stt": None, "router": mock.MagicMock(), "llm": mock.MagicMock(),
            "tts": mock.MagicMock(), "registry": registry, "bus": None,
        }
        deps["tts"].speak = mock.AsyncMock()
        deps["router"].route = mock.AsyncMock(return_value=mock.MagicMock(
            intent=mock.MagicMock(value="SKILL"),
            skill_name="time", operation="get_current_time", params={},
            resolved_by="matcher",
        ))
        o = Orchestrator_(**deps)
        await o._process_text("que hora es")
        # La skill se ejecuto via registry (verify indirectamente: TTS fue llamado)
        deps["tts"].speak.assert_called()

    @pytest.mark.asyncio
    async def test_dispatch_deny(self) -> None:
        from unittest import mock
        policy = Policy("config/permissions.yaml", whitelist_apps=WHITELIST_APPS)
        registry = SkillRegistry(policy)

        class AppSkill(Skill):
            name = "apps"
            description = "dummy"
            async def _do_execute(self, **params: object) -> SkillResult:
                return SkillResult(success=True)

        registry.register(AppSkill())
        deps = {
            "capture": None, "stt": None, "router": mock.MagicMock(), "llm": mock.MagicMock(),
            "tts": mock.MagicMock(), "registry": registry, "bus": None,
        }
        deps["tts"].speak = mock.AsyncMock()
        deps["router"].route = mock.AsyncMock(return_value=mock.MagicMock(
            intent=mock.MagicMock(value="SKILL"),
            skill_name="apps", operation="open", params={"app_name": "cmd.exe"},
            resolved_by="matcher",
        ))
        o = Orchestrator_(**deps)
        await o._process_text("abre cmd.exe")
        # Se ejecuto, pero policy rechazo app_name no en whitelist -> TTS dice error
        assert deps["tts"].speak.called
