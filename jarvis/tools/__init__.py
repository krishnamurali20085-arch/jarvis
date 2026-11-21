"""Registry for the assistant's explicitly supported tools."""

from __future__ import annotations

import atexit
import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from jarvis.knowledge.downloads import (
    download_confirmation_required,
    download_resource,
)
from jarvis.knowledge.store import KnowledgeBase
from jarvis.memory import MemoryStore, TodoStatus
from jarvis.tools import file_tools, system_tools, web_tools

logger = logging.getLogger(__name__)

ToolHandler = Callable[..., dict[str, Any]]
ConfirmationRule = Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    confirmation_rule: ConfirmationRule | None = None


_TOOLS: dict[str, ToolDefinition] = {}


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    handler: ToolHandler,
    confirmation_rule: ConfirmationRule | None = None,
) -> None:
    if not name or name in _TOOLS:
        raise ValueError(f"Tool name must be non-empty and unique: {name!r}.")
    if parameters.get("type") != "object":
        raise ValueError("Tool parameters must use a JSON object schema.")
    _TOOLS[name] = ToolDefinition(
        name=name,
        description=description,
        parameters=deepcopy(parameters),
        handler=handler,
        confirmation_rule=confirmation_rule,
    )


def _validate_arguments(
    definition: ToolDefinition, arguments: dict[str, Any]
) -> str | None:
    schema = definition.parameters
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    missing = [key for key in required if key not in arguments]
    if missing:
        return f"Missing required argument(s): {', '.join(missing)}."
    unexpected = [key for key in arguments if key not in properties]
    if unexpected and schema.get("additionalProperties") is False:
        return f"Unexpected argument(s): {', '.join(unexpected)}."

    type_checks: dict[str, Callable[[Any], bool]] = {
        "string": lambda value: isinstance(value, str),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda value: isinstance(value, (int, float))
        and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
    }
    for key, value in arguments.items():
        property_schema = properties.get(key)
        if property_schema is None:
            continue
        expected_type = property_schema.get("type")
        check = type_checks.get(expected_type)
        if check is not None and not check(value):
            return f"Argument {key!r} must be a JSON {expected_type}."
        if "enum" in property_schema and value not in property_schema["enum"]:
            options = ", ".join(map(str, property_schema["enum"]))
            return f"Argument {key!r} must be one of: {options}."
        if expected_type in {"integer", "number"}:
            if "minimum" in property_schema and value < property_schema["minimum"]:
                return f"Argument {key!r} must be at least {property_schema['minimum']}."
            if "maximum" in property_schema and value > property_schema["maximum"]:
                return f"Argument {key!r} must be at most {property_schema['maximum']}."
        if expected_type == "string" and "maxLength" in property_schema:
            if len(value) > property_schema["maxLength"]:
                return (
                    f"Argument {key!r} cannot exceed "
                    f"{property_schema['maxLength']} characters."
                )
    return None


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return JSON-compatible tool descriptions for provider adapters."""
    return [
        {
            "name": definition.name,
            "description": definition.description,
            "parameters": deepcopy(definition.parameters),
        }
        for definition in _TOOLS.values()
    ]


def execute_tool(
    name: str, arguments: dict[str, Any], *, confirmed: bool = False
) -> dict[str, Any]:
    """Validate and execute a registered tool.

    `confirmed` is an application-layer decision and must never be copied from
    model-supplied arguments.
    """
    definition = _TOOLS.get(name)
    if definition is None:
        return {
            "ok": False,
            "error": {"code": "unknown_tool", "message": f"Unknown tool: {name}."},
        }
    if not isinstance(arguments, dict):
        return {
            "ok": False,
            "error": {
                "code": "invalid_arguments",
                "message": "Tool arguments must be a JSON object.",
            },
        }
    if any(not isinstance(key, str) for key in arguments):
        return {
            "ok": False,
            "error": {
                "code": "invalid_arguments",
                "message": "Tool argument names must be strings.",
            },
        }

    validation_error = _validate_arguments(definition, arguments)
    if validation_error:
        return {
            "ok": False,
            "error": {"code": "invalid_arguments", "message": validation_error},
        }

    try:
        needs_confirmation = (
            definition.confirmation_rule(arguments)
            if definition.confirmation_rule is not None
            else False
        )
    except (OSError, ValueError) as exc:
        return {
            "ok": False,
            "error": {"code": "invalid_arguments", "message": str(exc)},
        }
    if needs_confirmation and not confirmed:
        return {
            "ok": False,
            "error": {
                "code": "confirmation_required",
                "message": (
                    "This action needs explicit user confirmation from the "
                    "application interface before it can run."
                ),
                "tool": name,
            },
        }

    try:
        return definition.handler(**arguments)
    except Exception as exc:
        logger.exception("Tool %s failed unexpectedly", name)
        return {
            "ok": False,
            "error": {
                "code": "tool_error",
                "message": f"{type(exc).__name__}: {exc}",
            },
        }


def _object(
    properties: dict[str, dict[str, Any]], required: list[str] | None = None
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _requires_confirmation(_: dict[str, Any]) -> bool:
    return True


def _write_requires_confirmation(arguments: dict[str, Any]) -> bool:
    return bool(arguments.get("overwrite", False))


def _download_requires_confirmation(arguments: dict[str, Any]) -> bool:
    return download_confirmation_required(arguments["url"])


@lru_cache(maxsize=1)
def _memory_store() -> MemoryStore:
    return MemoryStore()


def close_memory_store() -> None:
    """Close the process-wide memory connection when the application exits."""
    if _memory_store.cache_info().currsize:
        _memory_store().close()
        _memory_store.cache_clear()


atexit.register(close_memory_store)


def _memory_fact_set(key: str, value: str) -> dict[str, Any]:
    _memory_store().set_fact(key, value)
    return {"ok": True, "data": {"key": key, "saved": True}}


def _memory_fact_get(key: str) -> dict[str, Any]:
    value = _memory_store().get_fact(key)
    return {"ok": True, "data": {"key": key, "value": value}}


def _memory_facts_list() -> dict[str, Any]:
    return {"ok": True, "data": {"facts": _memory_store().list_facts()}}


def _memory_note_add(content: str) -> dict[str, Any]:
    identifier = _memory_store().add_note(content)
    return {"ok": True, "data": {"id": identifier}}


def _memory_notes_list(limit: int = 100) -> dict[str, Any]:
    return {"ok": True, "data": {"notes": _memory_store().list_notes(limit)}}


def _memory_reminder_add(
    content: str, due_at: str | None = None
) -> dict[str, Any]:
    identifier = _memory_store().add_reminder(content, due_at)
    return {"ok": True, "data": {"id": identifier}}


def _memory_reminders_list(
    include_completed: bool = False, limit: int = 100
) -> dict[str, Any]:
    return {
        "ok": True,
        "data": {
            "reminders": _memory_store().list_reminders(include_completed, limit)
        },
    }


def _memory_reminder_complete(reminder_id: str) -> dict[str, Any]:
    completed = _memory_store().complete_reminder(reminder_id)
    return {"ok": True, "data": {"completed": completed}}


def _memory_todo_add(content: str) -> dict[str, Any]:
    identifier = _memory_store().add_todo(content)
    return {"ok": True, "data": {"id": identifier}}


def _memory_todos_list(status: str = "open", limit: int = 100) -> dict[str, Any]:
    if status == "all":
        selected_status: TodoStatus | None = None
    elif status == "completed":
        selected_status = "completed"
    else:
        selected_status = "open"
    return {
        "ok": True,
        "data": {
            "todos": _memory_store().list_todos(selected_status, limit)
        },
    }


def _memory_todo_status(todo_id: str, status: str) -> dict[str, Any]:
    todo_status: TodoStatus = "completed" if status == "completed" else "open"
    return {
        "ok": True,
        "data": {
            "updated": _memory_store().set_todo_status(todo_id, todo_status)
        },
    }


@lru_cache(maxsize=1)
def _knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()


def close_knowledge_base() -> None:
    """Close the process-wide knowledge client when the application exits."""
    if _knowledge_base.cache_info().currsize:
        _knowledge_base().close()
        _knowledge_base.cache_clear()


atexit.register(close_knowledge_base)


def _knowledge_search(
    query: str, top_k: int = 5, subject: str | None = None
) -> dict[str, Any]:
    return _knowledge_base().search(query, top_k, subject or None)


register_tool(
    "system_status",
    "Get CPU usage, memory usage, battery charge, and platform information.",
    _object({}),
    system_tools.get_system_status,
)
register_tool(
    "open_application",
    "Open an allowlisted built-in application.",
    _object({"application": {"type": "string", "maxLength": 40}}, ["application"]),
    system_tools.open_application,
)
register_tool(
    "close_application",
    "Gracefully close an allowlisted application; requires user confirmation.",
    _object({"application": {"type": "string", "maxLength": 40}}, ["application"]),
    system_tools.close_application,
    _requires_confirmation,
)
register_tool(
    "set_system_volume",
    "Set system output volume from 0 to 100 percent.",
    _object(
        {"level": {"type": "integer", "minimum": 0, "maximum": 100}}, ["level"]
    ),
    system_tools.set_system_volume,
)
register_tool(
    "set_screen_brightness",
    "Set screen brightness from 0 to 100 percent where supported.",
    _object(
        {"level": {"type": "integer", "minimum": 0, "maximum": 100}}, ["level"]
    ),
    system_tools.set_screen_brightness,
)
register_tool(
    "take_screenshot",
    "Capture the screen to the private workspace; requires user confirmation.",
    _object({}),
    system_tools.take_screenshot,
    _requires_confirmation,
)
register_tool(
    "web_search",
    "Search the public web and return a small set of results.",
    _object(
        {
            "query": {"type": "string", "maxLength": 500},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        ["query"],
    ),
    web_tools.web_search,
)
register_tool(
    "get_weather",
    "Get current weather for a city or the configured default city.",
    _object({"location": {"type": "string", "maxLength": 200}}),
    web_tools.get_weather,
)
register_tool(
    "open_url",
    "Open a validated public HTTP(S) URL in the default browser.",
    _object({"url": {"type": "string", "maxLength": 2000}}, ["url"]),
    web_tools.open_url,
)
register_tool(
    "browser_search",
    "Open a DuckDuckGo search in the default browser.",
    _object({"query": {"type": "string", "maxLength": 500}}, ["query"]),
    web_tools.open_browser_search,
)
register_tool(
    "youtube_search",
    "Open a YouTube search in the default browser.",
    _object({"query": {"type": "string", "maxLength": 500}}, ["query"]),
    web_tools.open_youtube_search,
)
register_tool(
    "fetch_webpage",
    "Fetch and extract readable text from a public web page.",
    _object(
        {
            "url": {"type": "string", "maxLength": 2000},
            "max_chars": {"type": "integer", "minimum": 500, "maximum": 30000},
        },
        ["url"],
    ),
    web_tools.fetch_webpage,
)
register_tool(
    "download_resource",
    "Download a document to local staging; exact verified educational sources may be indexed automatically.",
    _object({"url": {"type": "string", "maxLength": 2000}}, ["url"]),
    download_resource,
    _download_requires_confirmation,
)
register_tool(
    "summarize_webpage",
    "Create a short extractive summary of a public web page.",
    _object(
        {
            "url": {"type": "string", "maxLength": 2000},
            "max_chars": {"type": "integer", "minimum": 200, "maximum": 5000},
        },
        ["url"],
    ),
    web_tools.summarize_webpage,
)
register_tool(
    "file_list",
    "List files and folders inside the configured workspace.",
    _object({"directory": {"type": "string", "maxLength": 1000}}),
    file_tools.list_files,
)
register_tool(
    "file_read",
    "Read a UTF-8 text file inside the configured workspace.",
    _object({"path": {"type": "string", "maxLength": 1000}}, ["path"]),
    file_tools.read_file,
)
register_tool(
    "file_write",
    "Write a UTF-8 text file inside the workspace. Overwriting requires confirmation.",
    _object(
        {
            "path": {"type": "string", "maxLength": 1000},
            "content": {"type": "string", "maxLength": 2_000_000},
            "overwrite": {"type": "boolean"},
        },
        ["path", "content"],
    ),
    file_tools.write_file,
    _write_requires_confirmation,
)
register_tool(
    "file_move",
    "Move a file within the workspace; requires user confirmation.",
    _object(
        {
            "source": {"type": "string", "maxLength": 1000},
            "destination": {"type": "string", "maxLength": 1000},
            "overwrite": {"type": "boolean"},
        },
        ["source", "destination"],
    ),
    file_tools.move_file,
    _requires_confirmation,
)
register_tool(
    "file_search",
    "Search UTF-8 text files inside the configured workspace.",
    _object(
        {
            "query": {"type": "string", "maxLength": 500},
            "directory": {"type": "string", "maxLength": 1000},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        ["query"],
    ),
    file_tools.search_files,
)
register_tool(
    "knowledge_search",
    "Search local educational documents and return cited passages.",
    _object({"query": {"type": "string", "maxLength": 2000}}, ["query"]),
    _knowledge_search,
)
register_tool(
    "memory_set_fact",
    "Remember or update a user fact or preference.",
    _object(
        {
            "key": {"type": "string", "maxLength": 200},
            "value": {"type": "string", "maxLength": 20000},
        },
        ["key", "value"],
    ),
    _memory_fact_set,
)
register_tool(
    "memory_get_fact",
    "Recall one saved user fact or preference.",
    _object({"key": {"type": "string", "maxLength": 200}}, ["key"]),
    _memory_fact_get,
)
register_tool(
    "memory_list_facts",
    "List all saved user facts and preferences.",
    _object({}),
    _memory_facts_list,
)
register_tool(
    "memory_add_note",
    "Save a personal note in local memory.",
    _object({"content": {"type": "string", "maxLength": 20000}}, ["content"]),
    _memory_note_add,
)
register_tool(
    "memory_list_notes",
    "List the most recent saved personal notes.",
    _object({"limit": {"type": "integer", "minimum": 1, "maximum": 1000}}),
    _memory_notes_list,
)
register_tool(
    "memory_add_reminder",
    "Save a reminder with an optional ISO 8601 due date or time.",
    _object(
        {
            "content": {"type": "string", "maxLength": 20000},
            "due_at": {"type": "string", "maxLength": 100},
        },
        ["content"],
    ),
    _memory_reminder_add,
)
register_tool(
    "memory_list_reminders",
    "List open reminders, optionally including completed reminders.",
    _object(
        {
            "include_completed": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
        }
    ),
    _memory_reminders_list,
)
register_tool(
    "memory_complete_reminder",
    "Mark a saved reminder complete by its ID.",
    _object({"reminder_id": {"type": "string", "maxLength": 100}}, ["reminder_id"]),
    _memory_reminder_complete,
)
register_tool(
    "memory_add_todo",
    "Add an item to the local to-do list.",
    _object({"content": {"type": "string", "maxLength": 20000}}, ["content"]),
    _memory_todo_add,
)
register_tool(
    "memory_list_todos",
    "List open or completed to-do items.",
    _object(
        {
            "status": {
                "type": "string",
                "enum": ["open", "completed", "all"],
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
        }
    ),
    _memory_todos_list,
)
register_tool(
    "memory_set_todo_status",
    "Mark a to-do item open or completed by its ID.",
    _object(
        {
            "todo_id": {"type": "string", "maxLength": 100},
            "status": {"type": "string", "enum": ["open", "completed"]},
        },
        ["todo_id", "status"],
    ),
    _memory_todo_status,
)
