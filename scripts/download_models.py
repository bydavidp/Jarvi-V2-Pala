#!/usr/bin/env python3
"""Descarga el binario de Piper y el modelo de voz española.

Uso:
    python scripts/download_models.py
"""

import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

PIPER_VERSION = "2023.11.14-2"
PIPER_ZIP = f"piper_windows_amd64.zip"
PIPER_URL = f"https://github.com/rhasspy/piper/releases/download/{PIPER_VERSION}/{PIPER_ZIP}"

VOICE_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/carlfm/x_low"
VOICE_ONNX = "es_ES-carlfm-x_low.onnx"
VOICE_JSON = f"{VOICE_ONNX}.json"

BIN_DIR = Path("bin")
PIPER_EXE_DIR = BIN_DIR / "piper"
MODELS_DIR = Path("models/piper")


def download(url: str, dest: Path) -> None:
    print(f"Descargando {url} ...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  -> {dest}")


def main() -> None:
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    piper_exe = PIPER_EXE_DIR / "piper.exe"
    if not piper_exe.exists():
        zip_path = BIN_DIR / PIPER_ZIP
        if not zip_path.exists():
            download(PIPER_URL, zip_path)
        print("Extrayendo...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(BIN_DIR)
        zip_path.unlink(missing_ok=True)
        print(f"  -> {piper_exe}")
    else:
        print(f"Piper ya existe: {piper_exe}")

    onnx_path = MODELS_DIR / VOICE_ONNX
    json_path = MODELS_DIR / VOICE_JSON

    if not onnx_path.exists():
        download(f"{VOICE_BASE}/{VOICE_ONNX}", onnx_path)
    else:
        print(f"Modelo ONNX ya existe: {onnx_path}")

    if not json_path.exists():
        download(f"{VOICE_BASE}/{VOICE_JSON}", json_path)
    else:
        print(f"Modelo JSON ya existe: {json_path}")

    print("Listo.")


if __name__ == "__main__":
    main()
