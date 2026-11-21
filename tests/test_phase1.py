"""Focused tests for Phase 1 persistence, tool validation, and file safety."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from jarvis.config import get_settings
from jarvis.memory import MemoryStore
from jarvis.tools import close_memory_store, execute_tool
from jarvis.tools.file_tools import resolve_workspace_path


class PhaseOneTests(unittest.TestCase):
    def setUp(self) -> None:
        close_memory_store()
        self._previous_environment = {
            key: os.environ.get(key)
            for key in (
                "JARVIS_WORKSPACE",
                "JARVIS_DATABASE_PATH",
                "JARVIS_TEXT_ONLY",
                "JARVIS_AUTO_INDEX_APPROVED_SOURCES",
            )
        }
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.workspace = self.root / "workspace"
        self.database = self.root / "memory.sqlite3"
        os.environ["JARVIS_WORKSPACE"] = str(self.workspace)
        os.environ["JARVIS_DATABASE_PATH"] = str(self.database)
        os.environ["JARVIS_TEXT_ONLY"] = "true"
        os.environ["JARVIS_AUTO_INDEX_APPROVED_SOURCES"] = "true"
        get_settings.cache_clear()

    def tearDown(self) -> None:
        close_memory_store()
        for key, value in self._previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()
        self._temporary_directory.cleanup()

    def test_memory_persists_facts_notes_reminders_and_todos(self) -> None:
        with MemoryStore() as memory:
            memory.set_fact("Name", "Ada")
            note_id = memory.add_note("Keep this note")
            reminder_id = memory.add_reminder("Call home", "2026-11-20T10:00:00Z")
            todo_id = memory.add_todo("Ship Phase 1")

        with MemoryStore() as memory:
            self.assertEqual(memory.get_fact("name"), "Ada")
            self.assertEqual(memory.list_notes()[0]["id"], note_id)
            self.assertEqual(memory.list_reminders()[0]["id"], reminder_id)
            self.assertEqual(memory.list_todos()[0]["id"], todo_id)

    def test_workspace_rejects_traversal_and_confirm_gates_mutations(self) -> None:
        with self.assertRaises(ValueError):
            resolve_workspace_path("../outside.txt")

        created = execute_tool(
            "file_write", {"path": "notes/today.txt", "content": "original"}
        )
        self.assertTrue(created["ok"])

        overwrite = execute_tool(
            "file_write",
            {
                "path": "notes/today.txt",
                "content": "replacement",
                "overwrite": True,
            },
        )
        self.assertEqual(overwrite["error"]["code"], "confirmation_required")
        self.assertEqual(
            (self.workspace / "notes" / "today.txt").read_text(encoding="utf-8"),
            "original",
        )

        moved = execute_tool(
            "file_move",
            {"source": "notes/today.txt", "destination": "archive/today.txt"},
        )
        self.assertEqual(moved["error"]["code"], "confirmation_required")
        self.assertTrue(
            execute_tool(
                "file_move",
                {"source": "notes/today.txt", "destination": "archive/today.txt"},
                confirmed=True,
            )["ok"]
        )

    def test_tool_registry_rejects_unknown_tools_and_invalid_types(self) -> None:
        self.assertEqual(
            execute_tool("not_registered", {})["error"]["code"], "unknown_tool"
        )
        self.assertEqual(
            execute_tool("set_system_volume", {"level": True})["error"]["code"],
            "invalid_arguments",
        )

    def test_auto_index_setting_is_validated_and_configurable(self) -> None:
        self.assertTrue(get_settings().auto_index_approved_sources)
        os.environ["JARVIS_AUTO_INDEX_APPROVED_SOURCES"] = "false"
        get_settings.cache_clear()
        self.assertFalse(get_settings().auto_index_approved_sources)

    def test_workspace_rejects_symlink_escape_when_supported(self) -> None:
        outside = self.root / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        self.workspace.mkdir(parents=True)
        link = self.workspace / "escape.txt"
        try:
            link.symlink_to(outside)
        except (NotImplementedError, OSError):
            self.skipTest("This Windows environment does not permit symlink creation.")
        with self.assertRaises(ValueError):
            resolve_workspace_path("escape.txt")


if __name__ == "__main__":
    unittest.main()
