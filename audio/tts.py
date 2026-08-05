import asyncio
import logging
from typing import Any

import numpy as np
import sounddevice as sd

from core.bus import Event, EventBus
from core.config import PiperConfig

logger = logging.getLogger("jarvis.audio.tts")


class TtsError(Exception):
    """Error en la síntesis o reproducción de voz."""


class TtsEngine:
    def __init__(
        self,
        config: PiperConfig,
        bus: EventBus | None = None,
        sample_rate: int = 22050,
    ) -> None:
        self._binary = config.binary_path
        self._model = config.model_path
        self._bus = bus
        self._sample_rate = sample_rate
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._stopped = False

    async def speak(self, text: str) -> None:
        if not text.strip():
            return

        if not self._model_path_valid():
            logger.error("Modelo Piper no encontrado en %s", self._model)
            return

        await self._queue.put(text)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.ensure_future(self._drain_queue())

    def stop(self) -> None:
        self._stopped = True
        sd.stop()
        self._drain_queue_clear()

    def _drain_queue_clear(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _model_path_valid(self) -> bool:
        import os
        return os.path.isfile(self._model)

    async def _drain_queue(self) -> None:
        try:
            while not self._stopped:
                try:
                    text = await self._queue.get()
                except asyncio.CancelledError:
                    break

                if self._bus is not None:
                    await self._bus.publish(Event(type="SPEAKING_START", data={"text": text}))

                try:
                    raw_audio = await self._synthesize(text)
                    await asyncio.to_thread(self._play_blocking, raw_audio)
                except TtsError as e:
                    logger.error("Error en TTS: %s", e)
                except Exception:
                    logger.exception("Error inesperado en TTS")
                finally:
                    if self._bus is not None:
                        await self._bus.publish(Event(type="SPEAKING_END"))

                self._queue.task_done()
        finally:
            self._stopped = False

    async def _synthesize(self, text: str) -> bytes:
        try:
            proc = await asyncio.create_subprocess_exec(
                self._binary,
                "--model", self._model,
                "--output-raw",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate(input=text.encode("utf-8"))

            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace").strip()
                raise TtsError(f"Piper finalizó con código {proc.returncode}: {err_msg}")

            return stdout
        except FileNotFoundError:
            raise TtsError(f"Binario Piper no encontrado: {self._binary}") from None
        except OSError as e:
            raise TtsError(f"Error al ejecutar Piper: {e}") from e

    def _play_blocking(self, raw_audio: bytes) -> None:
        if self._stopped:
            return

        audio = np.frombuffer(raw_audio, dtype=np.int16)
        if len(audio) == 0:
            return

        sd.play(audio, samplerate=self._sample_rate, blocking=False)
        try:
            while sd.get_stream().active:
                if self._stopped:
                    break
                sd.sleep(50)
        except Exception:
            pass
        finally:
            sd.stop()
