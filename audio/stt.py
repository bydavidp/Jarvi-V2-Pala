import logging
import time
from typing import Any

import numpy as np

from core.config import WhisperConfig

logger = logging.getLogger("jarvis.audio.stt")


def _detect_gpu_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _resolve_compute_type(config: WhisperConfig) -> str:
    if config.compute_type == "auto":
        if _detect_gpu_available():
            logger.info("GPU detectada, usando compute_type=auto")
            return "auto"
        else:
            logger.info("Sin GPU NVIDIA, forzando compute_type=int8")
            return "int8"
    return config.compute_type


class SttEngine:
    def __init__(self, config: WhisperConfig) -> None:
        self._config = config
        self._model = None
        self._compute_type = _resolve_compute_type(config)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info(
            "Cargando modelo faster-whisper '%s' con compute_type='%s'...",
            self._config.model, self._compute_type,
        )
        start = time.perf_counter()
        self._model = WhisperModel(
            self._config.model,
            device="cpu",
            compute_type=self._compute_type,
        )
        elapsed = time.perf_counter() - start
        logger.info("Modelo cargado en %.2fs", elapsed)

    def transcribe(self, audio: bytes, sample_rate: int = 16000) -> dict[str, Any]:
        start = time.perf_counter()
        self._load_model()

        if not audio or len(audio) == 0:
            return {"text": "", "confidence": 0.0, "duration": 0.0, "transcription_latency": 0.0}

        audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        segments, info = self._model.transcribe(
            audio_np,
            language=self._config.language,
            beam_size=5,
            vad_filter=True,
        )

        texts = []
        confidences = []
        for segment in segments:
            texts.append(segment.text.strip())
            confidences.append(segment.avg_logprob)

        transcription_end = time.perf_counter()
        latency = transcription_end - start

        text = " ".join(texts).strip()
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        audio_duration = len(audio_np) / sample_rate

        return {
            "text": text,
            "confidence": avg_confidence,
            "duration": audio_duration,
            "transcription_latency": latency,
            "language": info.language if info else self._config.language,
        }
