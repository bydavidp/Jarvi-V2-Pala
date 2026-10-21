"""Skill de código: escribe, lee y abre archivos dentro de raíces permitidas.

La contención vive aquí, no en ``policy.py``: la política autoriza *la
operación*, esta skill decide si *la ruta concreta* es aceptable. Reglas duras:

1. Toda ruta se resuelve (symlinks y ``..`` incluidos) y debe quedar dentro de
   una de las raíces configuradas. Si no, se rechaza.
2. Se bloquean extensiones ejecutables: Jarvis escribe código fuente, no
   binarios ni scripts que se autoejecuten.
3. Tope de tamaño por archivo, para no llenar el disco por accidente.
4. VS Code se lanza con lista de argumentos y ``shell=False``.
5. Nada de borrar archivos: no hay operación de borrado.
"""

import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from skills.base import Skill, SkillResult

logger = logging.getLogger("jarvis.skills.code")

# Extensiones que Jarvis nunca escribe: binarios o scripts que se ejecutan solos.
_BLOCKED_DEFAULT = [
    ".exe", ".dll", ".so", ".dylib", ".msi", ".com", ".scr",
    ".bat", ".cmd", ".ps1", ".vbs", ".js?", ".jar", ".pyc",
]


class CodeAccessError(Exception):
    """Ruta fuera del sandbox o contenido inválido."""


class CodeSkill(Skill):
    name = "code"
    description = "Crea, lee y abre archivos de código en las carpetas permitidas"
    level = "SYSTEM"

    def __init__(
        self,
        roots: list[str],
        max_file_bytes: int = 262_144,
        blocked_extensions: list[str] | None = None,
        vs_code_command: str = "code",
        open_timeout: float = 20.0,
    ) -> None:
        super().__init__()
        if not roots:
            raise CodeAccessError("La skill de código necesita al menos una raíz permitida")

        self._roots = [self._normalizar_raiz(r) for r in roots]
        self._max_bytes = max(1, max_file_bytes)
        self._blocked = {
            ext.lower() for ext in (blocked_extensions or _BLOCKED_DEFAULT)
        }
        self._code_cmd = vs_code_command
        self._open_timeout = open_timeout

    # ─── Sandbox ───────────────────────────────────────────────────────

    @staticmethod
    def _normalizar_raiz(raw: str) -> Path:
        expandida = os.path.expandvars(os.path.expanduser(str(raw)))
        if not expandida:
            raise CodeAccessError(f"Raíz inválida: {raw!r}")
        ruta = Path(expandida)
        if not ruta.is_absolute():
            ruta = Path.cwd() / ruta
        return ruta.resolve()

    @staticmethod
    def _dentro_de(target: Path, root: Path) -> bool:
        # normcase evita que 'C:\\Users\\DPala' y 'C:\\Users\\dpala' difieran.
        t = os.path.normcase(str(target))
        r = os.path.normcase(str(root))
        return t == r or t.startswith(r + os.sep)

    def _resolve(self, raw_path: Any, allow_root: bool = False) -> Path:
        if allow_root and (raw_path is None or (isinstance(raw_path, str) and not raw_path.strip())):
            return self._roots[0]

        if not isinstance(raw_path, str) or not raw_path.strip():
            raise CodeAccessError("Falta la ruta del archivo")

        texto = raw_path.strip()
        if "\x00" in texto:
            raise CodeAccessError("Ruta inválida")

        expandida = os.path.expandvars(os.path.expanduser(texto))
        candidata = Path(expandida)
        if not candidata.is_absolute():
            candidata = self._roots[0] / candidata

        try:
            resuelta = candidata.resolve()
        except (OSError, RuntimeError) as e:
            raise CodeAccessError(f"No pude resolver la ruta {texto!r}: {e}") from e

        for root in self._roots:
            if self._dentro_de(resuelta, root):
                return resuelta

        raise CodeAccessError(
            f"Ruta fuera de las carpetas permitidas: {resuelta}"
        )

    def _validar_extension(self, destino: Path) -> None:
        if destino.suffix.lower() in self._blocked:
            raise CodeAccessError(
                f"Extensión bloqueada por seguridad: {destino.suffix}"
            )

    # ─── Operaciones ───────────────────────────────────────────────────

    async def _do_execute(self, _operation: str = "", **params: Any) -> SkillResult:
        handlers: dict[str, Callable[..., SkillResult]] = {
            "write_file": self._write_file,
            "read_file": self._read_file,
            "list_files": self._list_files,
            "open_in_vscode": self._open_in_vscode,
        }
        handler = handlers.get(_operation)
        if handler is None:
            return SkillResult(
                success=False,
                error=f"Operación de código no soportada: {_operation!r}",
            )

        try:
            return handler(**params)
        except CodeAccessError as e:
            logger.warning("code.%s rechazado: %s", _operation, e)
            return SkillResult(success=False, error=str(e))
        except (OSError, UnicodeError) as e:
            logger.exception("code.%s falló", _operation)
            return SkillResult(success=False, error=f"Error de archivo: {e}")

    def _write_file(
        self,
        path: Any = "",
        content: Any = "",
        overwrite: Any = False,
        **_: Any,
    ) -> SkillResult:
        destino = self._resolve(path)
        self._validar_extension(destino)

        if not isinstance(content, str):
            raise CodeAccessError("El contenido debe ser texto")

        datos = content.encode("utf-8")
        if len(datos) > self._max_bytes:
            raise CodeAccessError(
                f"Contenido demasiado grande: {len(datos)} bytes "
                f"(máximo {self._max_bytes})"
            )

        existe = destino.exists()
        if existe and not self._es_verdadero(overwrite):
            raise CodeAccessError(
                f"El archivo ya existe: {destino}. Pide sobrescribir explícitamente."
            )
        if existe and destino.is_dir():
            raise CodeAccessError(f"La ruta es una carpeta, no un archivo: {destino}")

        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(content, encoding="utf-8")

        logger.info("Archivo escrito: %s (%d bytes)", destino, len(datos))
        return SkillResult(
            success=True,
            data={
                "path": str(destino),
                "bytes": len(datos),
                "overwritten": existe,
            },
        )

    def _read_file(self, path: Any = "", max_chars: Any = 4000, **_: Any) -> SkillResult:
        destino = self._resolve(path)
        self._validar_extension(destino)

        if not destino.exists() or not destino.is_file():
            raise CodeAccessError(f"No existe el archivo: {destino}")

        tamano = destino.stat().st_size
        if tamano > self._max_bytes:
            raise CodeAccessError(
                f"Archivo demasiado grande para leerlo entero: {tamano} bytes"
            )

        limite = max_chars if isinstance(max_chars, int) and max_chars > 0 else 4000
        texto = destino.read_text(encoding="utf-8", errors="replace")

        return SkillResult(
            success=True,
            data={
                "path": str(destino),
                "content": texto[:limite],
                "truncated": len(texto) > limite,
                "bytes": tamano,
            },
        )

    def _list_files(self, path: Any = "", max_entries: Any = 100, **_: Any) -> SkillResult:
        destino = self._resolve(path, allow_root=True)
        if not destino.is_dir():
            raise CodeAccessError(f"No es una carpeta: {destino}")

        limite = max_entries if isinstance(max_entries, int) and max_entries > 0 else 100
        entradas = sorted(destino.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

        return SkillResult(
            success=True,
            data={
                "path": str(destino),
                "entries": [
                    {"name": p.name, "dir": p.is_dir()} for p in entradas[:limite]
                ],
                "truncated": len(entradas) > limite,
                "total": len(entradas),
            },
        )

    def _open_in_vscode(self, path: Any = "", **_: Any) -> SkillResult:
        destino = self._resolve(path, allow_root=True)
        if not destino.exists():
            raise CodeAccessError(f"No existe la ruta: {destino}")

        try:
            proceso = subprocess.run(
                [self._code_cmd, "--reuse-window", str(destino)],
                capture_output=True,
                text=True,
                timeout=self._open_timeout,
                shell=False,
                check=False,
            )
        except FileNotFoundError:
            return SkillResult(
                success=False,
                error=f"No encontré el comando '{self._code_cmd}'. ¿Está VS Code en el PATH?",
            )
        except subprocess.TimeoutExpired:
            return SkillResult(
                success=False,
                error="VS Code tardó demasiado en abrir el archivo",
            )

        if proceso.returncode != 0:
            detalle = (proceso.stderr or proceso.stdout or "").strip()[:200]
            return SkillResult(
                success=False,
                error=f"VS Code devolvió {proceso.returncode}: {detalle}",
            )

        return SkillResult(success=True, data={"path": str(destino), "opened": True})

    @staticmethod
    def _es_verdadero(valor: Any) -> bool:
        if isinstance(valor, bool):
            return valor
        if isinstance(valor, str):
            return valor.strip().lower() in {"1", "true", "si", "sí", "yes", "ok"}
        return bool(valor)
