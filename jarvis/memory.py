"""SQLite-backed local facts, notes, reminders, and to-do items."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Literal
from uuid import uuid4

from jarvis.config import get_settings

TodoStatus = Literal["open", "completed"]


class MemoryError(RuntimeError):
    """Raised when a memory operation cannot be completed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_text(value: str, name: str, max_length: int = 20_000) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} cannot be empty.")
    if len(cleaned) > max_length:
        raise ValueError(f"{name} cannot exceed {max_length} characters.")
    return cleaned


class MemoryStore:
    """Thread-safe SQLite storage for personal assistant memory."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        configured_path = database_path or get_settings().database_path
        self.database_path = Path(configured_path)
        if str(configured_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._connection = sqlite3.connect(
                str(configured_path),
                timeout=10,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 10000")
            if str(configured_path) != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._initialize_schema()
        except sqlite3.Error as exc:
            raise MemoryError(f"Could not initialize memory database: {exc}") from exc

        self._lock = threading.RLock()
        self._closed = False

    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                due_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS todo_items (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('open', 'completed')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            """
        )

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        with self._lock:
            if self._closed:
                raise MemoryError("MemoryStore is closed.")
            try:
                with self._connection:
                    yield self._connection
            except sqlite3.Error as exc:
                raise MemoryError(f"Memory database operation failed: {exc}") from exc

    @staticmethod
    def _rows(rows: list[sqlite3.Row]) -> list[dict[str, object]]:
        return [dict(row) for row in rows]

    def set_fact(self, key: str, value: str) -> None:
        fact_key = _validate_text(key, "Fact key", max_length=200).casefold()
        fact_value = _validate_text(value, "Fact value")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO user_facts (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (fact_key, fact_value, _now()),
            )

    def get_fact(self, key: str) -> str | None:
        fact_key = _validate_text(key, "Fact key", max_length=200).casefold()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT value FROM user_facts WHERE key = ?", (fact_key,)
            ).fetchone()
            return None if row is None else str(row["value"])

    def list_facts(self) -> dict[str, str]:
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT key, value FROM user_facts ORDER BY key"
            ).fetchall()
            return {str(row["key"]): str(row["value"]) for row in rows}

    def add_note(self, content: str) -> str:
        note = _validate_text(content, "Note")
        note_id = str(uuid4())
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO notes (id, content, created_at) VALUES (?, ?, ?)",
                (note_id, note, _now()),
            )
        return note_id

    def list_notes(self, limit: int = 100) -> list[dict[str, object]]:
        if not 1 <= limit <= 1000:
            raise ValueError("Note limit must be between 1 and 1000.")
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT id, content, created_at FROM notes "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return self._rows(rows)

    def add_reminder(self, content: str, due_at: str | None = None) -> str:
        reminder = _validate_text(content, "Reminder")
        if due_at is not None:
            due_at = _validate_text(due_at, "Reminder due time", max_length=100)
            try:
                datetime.fromisoformat(due_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(
                    "Reminder due time must be an ISO 8601 date or datetime."
                ) from exc

        reminder_id = str(uuid4())
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO reminders (id, content, due_at, completed_at, created_at)
                VALUES (?, ?, ?, NULL, ?)
                """,
                (reminder_id, reminder, due_at, _now()),
            )
        return reminder_id

    def list_reminders(
        self, include_completed: bool = False, limit: int = 100
    ) -> list[dict[str, object]]:
        if not 1 <= limit <= 1000:
            raise ValueError("Reminder limit must be between 1 and 1000.")
        query = (
            "SELECT id, content, due_at, completed_at, created_at FROM reminders "
        )
        parameters: tuple[object, ...]
        if include_completed:
            parameters = (limit,)
        else:
            query += "WHERE completed_at IS NULL "
            parameters = (limit,)
        query += "ORDER BY due_at IS NULL, due_at, created_at LIMIT ?"
        with self._transaction() as connection:
            return self._rows(connection.execute(query, parameters).fetchall())

    def complete_reminder(self, reminder_id: str) -> bool:
        identifier = _validate_text(reminder_id, "Reminder ID", max_length=100)
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE reminders SET completed_at = ? "
                "WHERE id = ? AND completed_at IS NULL",
                (_now(), identifier),
            )
            return cursor.rowcount == 1

    def add_todo(self, content: str) -> str:
        item = _validate_text(content, "To-do item")
        todo_id = str(uuid4())
        now = _now()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO todo_items
                    (id, content, status, created_at, updated_at, completed_at)
                VALUES (?, ?, 'open', ?, ?, NULL)
                """,
                (todo_id, item, now, now),
            )
        return todo_id

    def list_todos(
        self, status: TodoStatus | None = "open", limit: int = 100
    ) -> list[dict[str, object]]:
        if not 1 <= limit <= 1000:
            raise ValueError("To-do limit must be between 1 and 1000.")
        if status not in {None, "open", "completed"}:
            raise ValueError("To-do status must be open, completed, or None.")

        query = (
            "SELECT id, content, status, created_at, updated_at, completed_at "
            "FROM todo_items "
        )
        parameters: tuple[object, ...]
        if status is None:
            parameters = (limit,)
        else:
            query += "WHERE status = ? "
            parameters = (status, limit)
        query += "ORDER BY created_at DESC, rowid DESC LIMIT ?"
        with self._transaction() as connection:
            return self._rows(connection.execute(query, parameters).fetchall())

    def set_todo_status(self, todo_id: str, status: TodoStatus) -> bool:
        if status not in {"open", "completed"}:
            raise ValueError("To-do status must be open or completed.")
        identifier = _validate_text(todo_id, "To-do ID", max_length=100)
        completed_at = _now() if status == "completed" else None
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE todo_items
                SET status = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, _now(), completed_at, identifier),
            )
            return cursor.rowcount == 1

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                try:
                    self._connection.close()
                except sqlite3.Error as exc:
                    raise MemoryError(f"Could not close memory database: {exc}") from exc
                self._closed = True

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(
        self, _exc_type: object, _exc: object, _traceback: object
    ) -> None:
        self.close()
