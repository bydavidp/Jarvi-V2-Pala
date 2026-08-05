import logging
import secrets
import time
from enum import Enum

logger = logging.getLogger("jarvis.security.confirm")


class ConfirmResult(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    PIN_EXPIRED = "pin_expired"


class PinVerifier:
    def __init__(
        self,
        pin: str,
        max_attempts: int = 3,
        cooldown_seconds: float = 5.0,
    ) -> None:
        if not pin:
            raise ValueError("PIN no puede estar vacío")
        self._pin = pin
        self._max_attempts = max_attempts
        self._cooldown = cooldown_seconds
        self._attempts: list[float] = []

    def verify(self, provided_pin: str) -> ConfirmResult:
        now = time.monotonic()
        self._attempts = [t for t in self._attempts if now - t < self._cooldown]

        if len(self._attempts) >= self._max_attempts:
            logger.warning("PIN bloqueado por cooldown: %d intentos en %.1fs", len(self._attempts), self._cooldown)
            return ConfirmResult.PIN_EXPIRED

        self._attempts.append(now)

        if secrets.compare_digest(self._pin, provided_pin):
            self._attempts.clear()
            return ConfirmResult.APPROVED

        logger.warning("Intento de PIN fallido. Restantes: %d", self.remaining_attempts())
        return ConfirmResult.DENIED

    def remaining_attempts(self) -> int:
        now = time.monotonic()
        self._attempts = [t for t in self._attempts if now - t < self._cooldown]
        return max(0, self._max_attempts - len(self._attempts))
