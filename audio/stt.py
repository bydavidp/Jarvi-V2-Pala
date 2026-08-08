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
        load_start = time.perf_counter()
        self._load_model()
        load_ms = (time.perf_counter() - load_start) * 1000

        if not audio or len(audio) == 0:
            return {"text": "", "avg_logprob": 0.0, "duration": 0.0, "transcription_ms": 0.0, "load_ms": load_ms}

        audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        peak_before = float(np.abs(audio_np).max())
        # Normalizar pico a 0.9 si el audio esta por debajo de nivel
        if 0.01 < peak_before < 0.5:
            gain = 0.9 / peak_before
            audio_np = audio_np * gain
            logger.info("Audio normalizado: peak %.3f -> %.3f (gain %.1fx)", peak_before, 0.9, gain)

        peak_after = float(np.abs(audio_np).max())

        segments, info = self._model.transcribe(
            audio_np,
            language=self._config.language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            no_repeat_ngram_size=3,
        )

        texts = []
        logprobs = []
        for segment in segments:
            texts.append(segment.text.strip())
            logprobs.append(segment.avg_logprob)

        infer_end = time.perf_counter()
        infer_ms = (infer_end - load_start) * 1000 - load_ms

        text = " ".join(texts).strip()
        avg_logprob = float(sum(logprobs) / len(logprobs)) if logprobs else 0.0
        audio_duration = len(audio_np) / sample_rate

        return {
            "text": text,
            "avg_logprob": avg_logprob,
            "duration": audio_duration,
            "transcription_ms": round(infer_ms, 1),
            "load_ms": round(load_ms, 1),
            "language": info.language if info else self._config.language,
            "peak_before": peak_before,
            "peak_after": peak_after,
        }
