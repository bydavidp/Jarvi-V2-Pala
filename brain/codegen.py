"""Interpreta el JSON que el LLM devuelve cuando pide escribir código.

El contenido del LLM es **dato**, no instrucciones: aquí solo se valida su
forma. La seguridad de la ruta la aplica ``skills/code.py``, no este módulo.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger("jarvis.brain.codegen")


class CodeArtifact(BaseModel):
    """Un archivo que el LLM propone escribir."""

    path: str = Field(min_length=1)
    content: str = Field(min_length=1)
    language: str = ""

    @field_validator("path")
    @classmethod
    def _ruta_limpia(cls, valor: str) -> str:
        limpia = valor.strip().strip('"').strip("'")
        if not limpia:
            raise ValueError("ruta vacía")
        # El LLM no decide rutas absolutas: solo relativas al sandbox.
        if limpia.startswith(("/", "\\")) or (len(limpia) > 1 and limpia[1] == ":"):
            raise ValueError("la ruta debe ser relativa a la carpeta permitida")
        return limpia

    @field_validator("language")
    @classmethod
    def _lenguaje(cls, valor: str) -> str:
        return valor.strip().lower()[:24]


def _extraer_json(texto: str) -> str:
    limpio = texto.strip()
    if "```" in limpio:
        inicio = limpio.find("```")
        resto = limpio[inicio + 3 :]
        if resto.lstrip().lower().startswith("json"):
            resto = resto.lstrip()[4:]
        fin = resto.find("```")
        if fin != -1:
            limpio = resto[:fin]
    inicio = limpio.find("{")
    fin = limpio.rfind("}")
    if inicio == -1 or fin <= inicio:
        raise ValueError("la respuesta no contiene un objeto JSON")
    return limpio[inicio : fin + 1]


def parse_code_artifact(texto: str) -> CodeArtifact:
    """Convierte la respuesta del LLM en un artefacto validado.

    Lanza ``ValueError`` con un motivo en español si no se puede.
    """
    crudo = _extraer_json(texto)
    try:
        data: Any = json.loads(crudo)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("se esperaba un objeto JSON")

    try:
        artefacto = CodeArtifact.model_validate(data)
    except ValidationError as e:
        primero = e.errors()[0]
        campo = ".".join(str(p) for p in primero.get("loc", ())) or "?"
        raise ValueError(f"campo inválido '{campo}': {primero.get('msg', '')}") from e

    logger.info(
        "Código generado: %s (%d caracteres, lenguaje=%s)",
        artefacto.path,
        len(artefacto.content),
        artefacto.language or "?",
    )
    return artefacto
