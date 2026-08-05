import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.security.audit")

SENSITIVE_KEY_PATTERN = re.compile(r"(pin|password|token|key|secret)", re.IGNORECASE)
REDACTED = "***REDACTADO***"
MAX_PARAM_LENGTH = 200


class AuditLog:
    def __init__(self, db_path: str = "data/audit.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    skill TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    params TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    result TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS audit_no_update
                BEFORE UPDATE ON audit_log
                BEGIN
                    SELECT RAISE(ABORT, 'UPDATE no permitido en audit_log (tabla append-only)');
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS audit_no_delete
                BEFORE DELETE ON audit_log
                BEGIN
                    SELECT RAISE(ABORT, 'DELETE no permitido en audit_log (tabla append-only)');
                END
            """)
            conn.commit()

    def log(
        self,
        skill: str,
        operation: str,
        params: dict[str, Any] | None,
        decision: str,
        result: str,
        details: str = "",
    ) -> None:
        sanitized = self._sanitize_params(params or {})
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "INSERT INTO audit_log (timestamp, skill, operation, params, decision, result, details) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (time.time(), skill, operation, sanitized, decision, result, details),
                )
                conn.commit()
        except Exception as e:
            logger.error("Error al escribir en audit log: %s", e)

    def _sanitize_params(self, params: dict[str, Any]) -> str:
        sanitized: dict[str, str] = {}
        for key, value in params.items():
            if SENSITIVE_KEY_PATTERN.search(key):
                sanitized[key] = REDACTED
            else:
                str_value = str(value)
                if len(str_value) > MAX_PARAM_LENGTH:
                    str_value = str_value[:MAX_PARAM_LENGTH] + "..."
                sanitized[key] = str_value
        return json.dumps(sanitized, ensure_ascii=False)

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def count_by_skill(self, skill: str) -> int:
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE skill = ?", (skill,)
            ).fetchone()
        return row[0] if row else 0

    def clear_for_test(self) -> None:
        """Solo para tests. Borra el archivo de base de datos."""
        if self._db_path.exists():
            self._db_path.unlink()
            self._init_db()
