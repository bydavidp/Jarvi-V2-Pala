"""Tests del parser de código del LLM."""

import json

import pytest

from brain.codegen import CodeArtifact, parse_code_artifact


def _json(**datos: object) -> str:
    return json.dumps(datos, ensure_ascii=False)


class TestParseValido:
    def test_json_plano(self) -> None:
        artefacto = parse_code_artifact(
            _json(path="script.py", content="print('hola')\n", language="Python")
        )

        assert artefacto.path == "script.py"
        assert artefacto.content == "print('hola')\n"
        assert artefacto.language == "python"

    def test_json_con_bloque_markdown(self) -> None:
        texto = "```json\n" + _json(path="a.py", content="x = 1") + "\n```"

        artefacto = parse_code_artifact(texto)

        assert artefacto.path == "a.py"

    def test_json_con_texto_alrededor(self) -> None:
        texto = "Claro, aquí va:\n" + _json(path="a.py", content="x") + "\nEspero que sirva."

        artefacto = parse_code_artifact(texto)

        assert artefacto.path == "a.py"

    def test_contenido_con_saltos_y_comillas(self) -> None:
        contenido = 'def f():\n    return "hola"\n'
        artefacto = parse_code_artifact(_json(path="f.py", content=contenido))

        assert artefacto.content == contenido

    def test_ruta_entre_comillas(self) -> None:
        artefacto = parse_code_artifact(_json(path='"a.py"', content="x"))

        assert artefacto.path == "a.py"

    def test_sin_lenguaje(self) -> None:
        artefacto = parse_code_artifact(_json(path="a.py", content="x"))

        assert artefacto.language == ""


class TestParseInvalido:
    def test_sin_json(self) -> None:
        with pytest.raises(ValueError, match="JSON"):
            parse_code_artifact("no hay nada aquí")

    def test_json_malformado(self) -> None:
        with pytest.raises(ValueError, match="JSON inválido"):
            parse_code_artifact('{"path": "a.py", "content": }')

    def test_json_no_objeto(self) -> None:
        with pytest.raises(ValueError, match="objeto JSON"):
            parse_code_artifact("[1, 2, 3]")

    def test_falta_path(self) -> None:
        with pytest.raises(ValueError, match="path"):
            parse_code_artifact(_json(content="x"))

    def test_path_vacio(self) -> None:
        with pytest.raises(ValueError):
            parse_code_artifact(_json(path="   ", content="x"))

    def test_content_vacio(self) -> None:
        with pytest.raises(ValueError, match="content"):
            parse_code_artifact(_json(path="a.py", content=""))

    def test_ruta_absoluta_posix(self) -> None:
        with pytest.raises(ValueError, match="relativa"):
            parse_code_artifact(_json(path="/etc/passwd", content="x"))

    def test_ruta_absoluta_windows(self) -> None:
        with pytest.raises(ValueError, match="relativa"):
            parse_code_artifact(_json(path="C:/Users/dpala/x.py", content="x"))

    def test_ruta_absoluta_windows_backslash(self) -> None:
        with pytest.raises(ValueError, match="relativa"):
            parse_code_artifact(_json(path="\\\\servidor\\share\\x.py", content="x"))


class TestModelo:
    def test_lenguaje_se_normaliza(self) -> None:
        artefacto = CodeArtifact(path="a.py", content="x", language="  Python  ")

        assert artefacto.language == "python"

    def test_lenguaje_se_recorta(self) -> None:
        artefacto = CodeArtifact(path="a.py", content="x", language="a" * 100)

        assert len(artefacto.language) == 24
