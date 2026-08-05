import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("jarvis.bus")


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


Subscriber = Callable[[Event], Coroutine[Any, Any, None]]


class EventBusError(Exception):
    """Error al publicar en el bus desde un hilo sin loop enlazado o con loop cerrado."""


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, event_type: str, callback: Subscriber) -> None:
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Subscriber) -> None:
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                cb for cb in self._subscribers[event_type] if cb is not callback
            ]

    async def publish(self, event: Event) -> None:
        callbacks = self._subscribers.get(event.type, [])
        for callback in callbacks:
            await callback(event)

    def _log_publish_error(self, future: asyncio.Future[None]) -> None:
        exc = future.exception()
        if exc is not None:
            logger.exception("Excepción no capturada en suscriptor del bus: %s", exc)

    def publish_threadsafe(self, event: Event) -> None:
        if self._loop is None:
            msg = (
                "El bus no tiene un event loop enlazado. "
                "Llama a bus.bind_loop(loop) antes de usar publish_threadsafe()."
            )
            logger.error("publish_threadsafe falló: %s", msg)
            raise EventBusError(msg)

        try:
            future = asyncio.run_coroutine_threadsafe(self.publish(event), self._loop)
            future.add_done_callback(self._log_publish_error)
        except RuntimeError as e:
            msg = f"No se pudo publicar evento {event.type!r} desde hilo: {e}"
            logger.error("publish_threadsafe falló: %s", msg)
            raise EventBusError(msg) from e
