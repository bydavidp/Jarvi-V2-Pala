import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings


class OllamaConfig(BaseModel):
    url: str = "http://localhost:11434"
    model: str = "llama3.2:latest"
    keep_alive: str = "30m"
    timeout: int = 30


class AudioConfig(BaseModel):
    input_device: int | None = None
    output_device: int | None = None
    vad_silence_ms: int = 800
    vad_min_speech_ms: int = 300
    vad_threshold: float = 0.04
    sample_rate: int = 16000


class WhisperConfig(BaseModel):
    model: str = "base"
    compute_type: str = "int8"
    language: str = "es"


class PiperConfig(BaseModel):
    model_path: str
    binary_path: str


class WakeWordConfig(BaseModel):
    model_path: str
    sensitivity: float = 0.5


class WhitelistConfig(BaseModel):
    apps: dict[str, str] = Field(default_factory=dict)


class SkillToggle(BaseModel):
    enabled: bool = True


class SkillsConfig(BaseModel):
    time: SkillToggle = Field(default_factory=SkillToggle)


class Settings(BaseModel):
    idioma: str = "es"
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    piper: PiperConfig
    wakeword: WakeWordConfig
    whitelist: WhitelistConfig = Field(default_factory=WhitelistConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)


class Secrets(BaseSettings):
    pin: str = ""
    search_api_key: str = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class Config:
    def __init__(self, settings_path: str = "config/settings.yaml") -> None:
        self.settings = load_settings(settings_path)
        self.secrets = Secrets()

    @property
    def idioma(self) -> str:
        return self.settings.idioma

    @property
    def ollama(self) -> OllamaConfig:
        return self.settings.ollama

    @property
    def audio(self) -> AudioConfig:
        return self.settings.audio

    @property
    def whisper(self) -> WhisperConfig:
        return self.settings.whisper

    @property
    def piper(self) -> PiperConfig:
        return self.settings.piper

    @property
    def wakeword(self) -> WakeWordConfig:
        return self.settings.wakeword

    @property
    def whitelist(self) -> WhitelistConfig:
        return self.settings.whitelist

    @property
    def skills(self) -> SkillsConfig:
        return self.settings.skills

    @property
    def pin(self) -> str:
        return self.secrets.pin

    @property
    def search_api_key(self) -> str:
        return self.secrets.search_api_key


def load_settings(path: str) -> Settings:
    full_path = Path(path)
    if not full_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: {full_path.resolve()}"
        )

    raw = yaml.safe_load(full_path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"El archivo de configuración está vacío: {full_path.resolve()}")

    try:
        return Settings.model_validate(raw)
    except ValidationError as e:
        raise ValueError(
            f"Error de validación en {full_path.resolve()}:\n{e}"
        ) from e
