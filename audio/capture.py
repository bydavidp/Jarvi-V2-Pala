import logging
import time
from typing import Any

import numpy as np
import sounddevice as sd

logger = logging.getLogger("jarvis.audio.capture")


class AudioCapture:
    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = 16000,
        vad_silence_ms: int = 800,
        vad_min_speech_ms: int = 300,
        vad_threshold: float = 0.02,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._silence_samples = int(vad_silence_ms * sample_rate / 1000)
        self._min_speech_samples = int(vad_min_speech_ms * sample_rate / 1000)
        self._threshold = vad_threshold

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
        started = time.perf_counter()

        buffer: list[np.ndarray] = []
        silent_frames = 0
        speech_frames = 0
        triggered = False

        def callback(indata: np.ndarray, frames: int, time_info: dict, status: int) -> None:
            nonlocal silent_frames, speech_frames, triggered

            if status:
                logger.warning("Stream status: %s", status)

            mono = indata.mean(axis=1) if indata.ndim > 1 and indata.shape[1] > 1 else indata[:, 0]
            mono_float = mono.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(mono_float**2) + 1e-10))

            is_speech = rms > self._threshold
            buffer.append(indata.copy())

            if is_speech:
                speech_frames += len(indata)
                silent_frames = 0
                triggered = True
            elif triggered:
                silent_frames += len(indata)

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

                if triggered and silent_frames >= self._silence_samples:
                    break

                sd.sleep(50)

            stream.stop()
            stream.close()

            capture_end = time.perf_counter()

            if not buffer or not triggered or speech_frames < self._min_speech_samples:
                logger.info("Frase demasiado corta o sin habla: %d frames", speech_frames)
                return {"audio": b"", "duration": 0, "capture_latency": capture_end - capture_start}

            audio_raw = np.concatenate(buffer, axis=0)
            duration = len(audio_raw) / self._sample_rate

            return {
                "audio": audio_raw.tobytes(),
                "duration": duration,
                "capture_latency": capture_end - capture_start,
                "speech_frames": speech_frames,
            }

        except Exception:
            logger.exception("Error en captura de audio")
            return {"audio": b"", "duration": 0, "capture_latency": 0}
