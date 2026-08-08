import logging
import time
from typing import Any

import numpy as np
import sounddevice as sd
import torch

from silero_vad import load_silero_vad

logger = logging.getLogger("jarvis.audio.capture")

SILERO_WINDOW = 512  # muestras a 16kHz = 32ms
SILERO_SAMPLE_RATE = 16000


class AudioCapture:
    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = 16000,
        vad_silence_ms: int = 800,
        vad_min_speech_ms: int = 300,
        vad_threshold: float = 0.5,
    ) -> None:
        if sample_rate != SILERO_SAMPLE_RATE:
            raise ValueError(f"Silero VAD requiere {SILERO_SAMPLE_RATE}Hz, se recibió {sample_rate}Hz")

        self._device = device
        self._sample_rate = sample_rate
        self._silence_samples = int(vad_silence_ms * sample_rate / 1000)
        self._min_speech_samples = int(vad_min_speech_ms * sample_rate / 1000)
        self._vad_threshold = vad_threshold
        self._vad_model: Any = None
        self._vad_state: Any = None

    def _ensure_vad_loaded(self) -> None:
        if self._vad_model is not None:
            return
        logger.info("Cargando modelo Silero VAD...")
        self._vad_model = load_silero_vad()

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            devices.append({
                "index": i,
                "name": dev["name"],
                "max_input_channels": dev["max_input_channels"],
                "max_output_channels": dev["max_output_channels"],
                "default_samplerate": int(dev["default_samplerate"]),
            })
        return devices

    def capture_phrase(self, timeout_seconds: float = 10.0) -> dict[str, Any]:
        self._ensure_vad_loaded()
        started = time.perf_counter()
        buffer: list[np.ndarray] = []
        chunk_buffer = np.zeros(0, dtype=np.float32)
        silent_samples = 0
        speech_samples = 0
        triggered = False
        vad_latencies: list[float] = []

        def callback(indata: np.ndarray, frames: int, time_info: dict, status: int) -> None:
            nonlocal silent_samples, speech_samples, triggered, chunk_buffer

            if status:
                logger.warning("Stream status: %s", status)

            mono = indata.mean(axis=1) if indata.ndim > 1 and indata.shape[1] > 1 else indata[:, 0]
            chunk = mono.astype(np.float32) / 32768.0

            buffer.append(indata.copy())
            chunk_buffer = np.concatenate([chunk_buffer, chunk])

            while len(chunk_buffer) >= SILERO_WINDOW:
                window = chunk_buffer[:SILERO_WINDOW]
                chunk_buffer = chunk_buffer[SILERO_WINDOW:]

                t0 = time.perf_counter()
                try:
                    prob = self._vad_model(torch.from_numpy(window), self._sample_rate).item()
                except Exception:
                    prob = 0.0
                vad_latencies.append(time.perf_counter() - t0)

                is_speech = prob >= self._vad_threshold
                if is_speech:
                    speech_samples += SILERO_WINDOW
                    silent_samples = 0
                    triggered = True
                elif triggered:
                    silent_samples += SILERO_WINDOW

            if triggered and silent_samples >= self._silence_samples:
                pass  # signal to stop (handled in main loop)

        try:
            stream = sd.InputStream(
                device=self._device,
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                callback=callback,
            )
            stream.start()

            capture_start = time.perf_counter()

            while stream.active:
                if time.perf_counter() - started > timeout_seconds:
                    logger.warning("Timeout de captura")
                    break

                if triggered and silent_samples >= self._silence_samples:
                    break

                sd.sleep(10)

            stream.stop()
            stream.close()

            capture_end = time.perf_counter()

            if not buffer or not triggered or speech_samples < self._min_speech_samples:
                logger.info("Sin habla suficiente: %d muestras", speech_samples)
                return {
                    "audio": b"", "duration": 0,
                    "capture_latency": capture_end - capture_start,
                    "speech_samples": speech_samples,
                }

            audio_raw = np.concatenate(buffer, axis=0)
            duration = len(audio_raw) / self._sample_rate

            avg_vad = sum(vad_latencies) / len(vad_latencies) * 1000 if vad_latencies else 0
            logger.info("VAD: %d chunks, latencia avg %.2fms", len(vad_latencies), avg_vad)

            return {
                "audio": audio_raw.tobytes(),
                "duration": duration,
                "capture_latency": capture_end - capture_start,
                "speech_samples": speech_samples,
                "speech_ms": speech_samples / self._sample_rate * 1000,
                "vad_latency_ms": avg_vad,
            }

        except Exception:
            logger.exception("Error en captura de audio")
            return {"audio": b"", "duration": 0, "capture_latency": 0}
