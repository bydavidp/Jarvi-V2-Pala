from unittest import mock

import numpy as np
import pytest

from audio.capture import AudioCapture
from audio.stt import SttEngine, _detect_gpu_available, _resolve_compute_type
from core.config import WhisperConfig


# ─── GPU detection ──────────────────────────────────────────────────────

class TestGpuDetection:
    def test_resolve_auto_sin_gpu_fuerza_int8(self) -> None:
        config = WhisperConfig(compute_type="auto")
        with mock.patch("audio.stt._detect_gpu_available", return_value=False):
            result = _resolve_compute_type(config)
            assert result == "int8"

    def test_resolve_auto_con_gpu_usa_auto(self) -> None:
        config = WhisperConfig(compute_type="auto")
        with mock.patch("audio.stt._detect_gpu_available", return_value=True):
            result = _resolve_compute_type(config)
            assert result == "auto"

    def test_resolve_explicito_respeta(self) -> None:
        config = WhisperConfig(compute_type="float32")
        result = _resolve_compute_type(config)
        assert result == "float32"


# ─── AudioCapture ───────────────────────────────────────────────────────

class TestAudioCapture:
    def test_list_devices(self) -> None:
        devices = AudioCapture.list_devices()
        assert isinstance(devices, list)
        assert len(devices) > 0
        for dev in devices:
            assert "index" in dev
            assert "name" in dev
            assert "max_input_channels" in dev
            assert "default_samplerate" in dev

    def test_config_defaults(self) -> None:
        cap = AudioCapture()
        assert cap._sample_rate == 16000
        assert cap._silence_samples > 0
        assert cap._min_speech_samples > 0


# ─── SttEngine ──────────────────────────────────────────────────────────

class TestSttEngine:
    def test_compute_type_detectado(self) -> None:
        config = WhisperConfig(model="base", compute_type="auto", language="es")
        engine = SttEngine(config)
        engine._load_model()
        assert engine._model is not None

    def test_audio_vacio(self) -> None:
        config = WhisperConfig(model="base", compute_type="int8", language="es")
        engine = SttEngine(config)
        result = engine.transcribe(b"")
        assert result["text"] == ""
        assert result["confidence"] == 0.0
