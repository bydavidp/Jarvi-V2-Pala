"""Tests de la skill de código y del sandbox que la protege."""

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from security.policy import Policy
from skills.base import Skill, SkillRegistry
from skills.code import CodeAccessError, CodeSkill


# ─── helpers ────────────────────────────────────────────────────────────


def _skill(root: Path, **kwargs: object) -> CodeSkill:
    opciones = {
        "roots": [str(root)],
        "max_file_bytes": 1024,
        "blocked_extensions": None,
        "vs_code_command": "code",
    }
    opciones.update(kwargs)
    return CodeSkill(**opciones)  # type: ignore[arg-type]


async def _run(skill: CodeSkill, operation: str, **params: object):
    skill._registry = mock.MagicMock()  # type: ignore[assignment]
    skill._registry.validate_token.return_value = True  # type: ignore[attr-defined]
    return await skill._execute(_auth_token="t", _operation=operation, **params)


# ─── construcción ───────────────────────────────────────────────────────


class TestConstruccion:
    def test_sin_raices_falla(self) -> None:
        with pytest.raises(CodeAccessError):
            CodeSkill(roots=[])

    def test_normaliza_raiz_inexistente(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path / "no-existe")
        assert skill._roots[0].is_absolute()


# ─── escritura ──────────────────────────────────────────────────────────


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_escribe_relativo_en_la_raiz(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path)
        result = await _run(skill, "write_file", path="hola.py", content="print(1)\n")

        assert result.success is True
        assert (tmp_path / "hola.py").read_text(encoding="utf-8") == "print(1)\n"
        assert result.data["bytes"] == 9
        assert result.data["overwritten"] is False

    @pytest.mark.asyncio
    async def test_crea_subcarpetas(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path)
        result = await _run(skill, "write_file", path="a/b/c.txt", content="x")

        assert result.success is True
        assert (tmp_path / "a" / "b" / "c.txt").exists()

    @pytest.mark.asyncio
    async def test_no_sobrescribe_sin_permiso(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("viejo", encoding="utf-8")
        skill = _skill(tmp_path)

        result = await _run(skill, "write_file", path="x.py", content="nuevo")

        assert result.success is False
        assert "ya existe" in result.error
        assert (tmp_path / "x.py").read_text(encoding="utf-8") == "viejo"

    @pytest.mark.asyncio
    async def test_sobrescribe_con_overwrite(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("viejo", encoding="utf-8")
        skill = _skill(tmp_path)

        result = await _run(skill, "write_file", path="x.py", content="nuevo", overwrite=True)

        assert result.success is True
        assert result.data["overwritten"] is True
        assert (tmp_path / "x.py").read_text(encoding="utf-8") == "nuevo"

    @pytest.mark.asyncio
    async def test_sobrescribe_con_overwrite_string(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("viejo", encoding="utf-8")
        skill = _skill(tmp_path)

        result = await _run(skill, "write_file", path="x.py", content="nuevo", overwrite="true")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_rechaza_contenido_no_texto(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path)
        result = await _run(skill, "write_file", path="x.py", content=123)

        assert result.success is False
        assert "texto" in result.error

    @pytest.mark.asyncio
    async def test_rechaza_archivo_demasiado_grande(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path, max_file_bytes=10)
        result = await _run(skill, "write_file", path="x.py", content="a" * 11)

        assert result.success is False
        assert "demasiado grande" in result.error
        assert not (tmp_path / "x.py").exists()


# ─── sandbox ────────────────────────────────────────────────────────────


class TestSandbox:
    @pytest.mark.asyncio
    async def test_rechaza_punto_punto(self, tmp_path: Path) -> None:
        root = tmp_path / "permitido"
        root.mkdir()
        skill = _skill(root)

        result = await _run(skill, "write_file", path="../fuera.py", content="x")

        assert result.success is False
        assert "fuera de las carpetas permitidas" in result.error
        assert not (tmp_path / "fuera.py").exists()

    @pytest.mark.asyncio
    async def test_rechaza_ruta_absoluta_fuera(self, tmp_path: Path) -> None:
        root = tmp_path / "permitido"
        root.mkdir()
        skill = _skill(root)

        result = await _run(skill, "write_file", path=str(tmp_path / "fuera.py"), content="x")

        assert result.success is False
        assert "fuera de las carpetas permitidas" in result.error

    @pytest.mark.asyncio
    async def test_rechaza_ruta_vacia(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path)
        result = await _run(skill, "write_file", path="", content="x")

        assert result.success is False
        assert "Falta la ruta" in result.error

    @pytest.mark.asyncio
    async def test_rechaza_extension_ejecutable(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path)

        for nombre in ("virus.exe", "script.bat", "tarea.ps1", "x.cmd", "y.vbs"):
            result = await _run(skill, "write_file", path=nombre, content="x")
            assert result.success is False, nombre
            assert "Extensión bloqueada" in result.error

    @pytest.mark.asyncio
    async def test_rechaza_extension_en_mayusculas(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path)
        result = await _run(skill, "write_file", path="VIRUS.EXE", content="x")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_permite_extensiones_de_codigo(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path)

        for nombre in ("a.py", "b.ts", "c.md", "Dockerfile", "d.json"):
            result = await _run(skill, "write_file", path=nombre, content="x")
            assert result.success is True, nombre
            assert (tmp_path / nombre).exists()


# ─── lectura y listado ──────────────────────────────────────────────────


class TestReadFile:
    @pytest.mark.asyncio
    async def test_lee_dentro_de_la_raiz(self, tmp_path: Path) -> None:
        (tmp_path / "nota.md").write_text("contenido", encoding="utf-8")
        skill = _skill(tmp_path)

        result = await _run(skill, "read_file", path="nota.md")

        assert result.success is True
        assert result.data["content"] == "contenido"
        assert result.data["truncated"] is False

    @pytest.mark.asyncio
    async def test_trunca_contenido_largo(self, tmp_path: Path) -> None:
        (tmp_path / "nota.md").write_text("a" * 100, encoding="utf-8")
        skill = _skill(tmp_path, max_file_bytes=4096)

        result = await _run(skill, "read_file", path="nota.md", max_chars=10)

        assert result.success is True
        assert result.data["content"] == "a" * 10
        assert result.data["truncated"] is True

    @pytest.mark.asyncio
    async def test_no_lee_fuera(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path)
        result = await _run(skill, "read_file", path="..\\secreto.txt")

        assert result.success is False
        assert "fuera de las carpetas permitidas" in result.error

    @pytest.mark.asyncio
    async def test_archivo_inexistente(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path)
        result = await _run(skill, "read_file", path="no.txt")

        assert result.success is False
        assert "No existe" in result.error


class TestListFiles:
    @pytest.mark.asyncio
    async def test_lista_ordenado_con_carpetas_primero(self, tmp_path: Path) -> None:
        (tmp_path / "zeta.py").write_text("", encoding="utf-8")
        (tmp_path / "alfa").mkdir()
        (tmp_path / "Beta.py").write_text("", encoding="utf-8")
        skill = _skill(tmp_path)

        result = await _run(skill, "list_files")

        assert result.success is True
        nombres = [e["name"] for e in result.data["entries"]]
        assert nombres[0] == "alfa"
        assert result.data["entries"][0]["dir"] is True
        assert result.data["total"] == 3

    @pytest.mark.asyncio
    async def test_respeta_max_entries(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text("", encoding="utf-8")
        skill = _skill(tmp_path)

        result = await _run(skill, "list_files", max_entries=2)

        assert len(result.data["entries"]) == 2
        assert result.data["truncated"] is True

    @pytest.mark.asyncio
    async def test_no_lista_fuera(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path)
        result = await _run(skill, "list_files", path="C:\\Windows")

        assert result.success is False


# ─── VS Code ────────────────────────────────────────────────────────────


class TestOpenInVsCode:
    @pytest.mark.asyncio
    async def test_abre_con_lista_de_argumentos(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("x", encoding="utf-8")
        skill = _skill(tmp_path)
        proceso = mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("skills.code.subprocess.run", return_value=proceso) as run:
            result = await _run(skill, "open_in_vscode", path="x.py")

        assert result.success is True
        args, kwargs = run.call_args
        assert args[0] == ["code", "--reuse-window", str(tmp_path / "x.py")]
        assert kwargs["shell"] is False
        assert "timeout" in kwargs

    @pytest.mark.asyncio
    async def test_sin_vscode_en_el_path(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("x", encoding="utf-8")
        skill = _skill(tmp_path)

        with mock.patch("skills.code.subprocess.run", side_effect=FileNotFoundError):
            result = await _run(skill, "open_in_vscode", path="x.py")

        assert result.success is False
        assert "PATH" in result.error

    @pytest.mark.asyncio
    async def test_codigo_de_error_de_vscode(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("x", encoding="utf-8")
        skill = _skill(tmp_path)
        proceso = mock.Mock(returncode=1, stdout="", stderr="boom")

        with mock.patch("skills.code.subprocess.run", return_value=proceso):
            result = await _run(skill, "open_in_vscode", path="x.py")

        assert result.success is False
        assert "1" in result.error

    @pytest.mark.asyncio
    async def test_no_abre_ruta_inexistente(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path)
        result = await _run(skill, "open_in_vscode", path="no.py")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_no_abre_fuera(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path)
        result = await _run(skill, "open_in_vscode", path="..")

        assert result.success is False
        assert "fuera de las carpetas permitidas" in result.error


# ─── contrato del registry ──────────────────────────────────────────────


class TestRegistro:
    @pytest.mark.asyncio
    async def test_operacion_desconocida(self, tmp_path: Path) -> None:
        skill = _skill(tmp_path)
        result = await _run(skill, "borrar_todo")

        assert result.success is False
        assert "no soportada" in result.error

    @pytest.mark.asyncio
    async def test_registry_pasa_la_operacion(self, tmp_path: Path) -> None:
        policy = Policy("config/permissions.yaml")
        registry = SkillRegistry(policy)
        registry.register(_skill(tmp_path))

        result = await registry.execute("code", "write_file", {"path": "ok.py", "content": "x"})

        assert result.success is True
        assert (tmp_path / "ok.py").exists()

    @pytest.mark.asyncio
    async def test_registry_rechaza_parametros_reservados(self, tmp_path: Path) -> None:
        policy = Policy("config/permissions.yaml")
        registry = SkillRegistry(policy)
        registry.register(_skill(tmp_path))

        result = await registry.execute(
            "code", "write_file", {"path": "ok.py", "content": "x", "_auth_token": "falso"}
        )

        assert result.success is False
        assert "no permitido" in result.error
        assert not (tmp_path / "ok.py").exists()

    def test_skill_no_puede_declarar_auth_token(self) -> None:
        class MalaSkill(Skill):
            name = "mala"
            description = "x"
            level = "SAFE"

            async def _do_execute(self, _auth_token: str = "", **params: object):  # type: ignore[override]
                raise AssertionError("no debería ejecutarse")

        policy = Policy("config/permissions.yaml")
        registry = SkillRegistry(policy)
        with pytest.raises(ValueError):
            registry.register(MalaSkill())

    def test_subprocess_no_usa_shell(self) -> None:
        fuente = Path("skills/code.py").read_text(encoding="utf-8")
        assert "shell=True" not in fuente
        assert "os.system" not in fuente
