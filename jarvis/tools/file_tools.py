"""Workspace-confined file operations."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from jarvis.config import get_settings

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_SEARCH_FILES = 1000


def workspace_root() -> Path:
    root = get_settings().workspace_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_workspace_path(relative_path: str) -> Path:
    """Resolve a relative path and reject traversal or symlink escapes."""
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("A non-empty workspace-relative path is required.")
    normalized_path = relative_path.replace("\\", "/")
    candidate = Path(normalized_path)
    if candidate.is_absolute() or candidate.drive:
        raise ValueError("Absolute paths are not allowed; use a workspace-relative path.")
    if ".." in candidate.parts:
        raise ValueError("Parent-directory traversal is not allowed.")

    root = workspace_root()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("The requested path resolves outside the workspace.") from exc
    return resolved


def list_files(directory: str = ".") -> dict[str, Any]:
    target = resolve_workspace_path(directory)
    if not target.exists():
        return {
            "ok": False,
            "error": {"code": "not_found", "message": "Directory does not exist."},
        }
    if not target.is_dir():
        return {
            "ok": False,
            "error": {"code": "not_a_directory", "message": "Path is not a directory."},
        }
    entries = []
    for entry in sorted(target.iterdir(), key=lambda item: item.name.casefold()):
        if entry.is_symlink():
            continue
        entries.append(
            {
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
                "size_bytes": entry.stat().st_size if entry.is_file() else None,
            }
        )
    return {
        "ok": True,
        "data": {
            "directory": str(target.relative_to(workspace_root())),
            "entries": entries[:500],
        },
    }


def read_file(path: str) -> dict[str, Any]:
    target = resolve_workspace_path(path)
    if not target.is_file():
        return {
            "ok": False,
            "error": {"code": "not_found", "message": "File does not exist."},
        }
    if target.stat().st_size > MAX_FILE_BYTES:
        return {
            "ok": False,
            "error": {
                "code": "file_too_large",
                "message": f"Files larger than {MAX_FILE_BYTES} bytes cannot be read.",
            },
        }
    try:
        content = target.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return {
            "ok": False,
            "error": {"code": "read_failed", "message": str(exc)},
        }
    return {
        "ok": True,
        "data": {
            "path": str(target.relative_to(workspace_root())),
            "content": content,
        },
    }


def write_file(path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
    target = resolve_workspace_path(path)
    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        return {
            "ok": False,
            "error": {
                "code": "content_too_large",
                "message": f"Content cannot exceed {MAX_FILE_BYTES} bytes.",
            },
        }
    if target.exists() and not overwrite:
        return {
            "ok": False,
            "error": {
                "code": "file_exists",
                "message": "The destination exists. Set overwrite=true and confirm the operation.",
            },
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    target = resolve_workspace_path(path)
    try:
        if overwrite:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=target.parent,
                prefix=".jarvis-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(content)
            try:
                os.replace(temporary_path, target)
            except OSError:
                temporary_path.unlink(missing_ok=True)
                raise
        else:
            with target.open("x", encoding="utf-8", newline="") as output:
                output.write(content)
    except FileExistsError:
        return {
            "ok": False,
            "error": {
                "code": "file_exists",
                "message": "The destination was created concurrently; no data was overwritten.",
            },
        }
    except OSError as exc:
        return {"ok": False, "error": {"code": "write_failed", "message": str(exc)}}

    return {
        "ok": True,
        "data": {"path": str(target.relative_to(workspace_root()))},
    }


def move_file(
    source: str, destination: str, overwrite: bool = False
) -> dict[str, Any]:
    source_path = resolve_workspace_path(source)
    destination_path = resolve_workspace_path(destination)
    if not source_path.is_file():
        return {
            "ok": False,
            "error": {"code": "not_found", "message": "Source file does not exist."},
        }
    if source_path == destination_path:
        return {
            "ok": False,
            "error": {
                "code": "same_path",
                "message": "Source and destination refer to the same file.",
            },
        }
    if destination_path.exists() and not overwrite:
        return {
            "ok": False,
            "error": {
                "code": "destination_exists",
                "message": "Destination exists. Set overwrite=true and confirm the move.",
            },
        }

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path = resolve_workspace_path(destination)
    try:
        if overwrite:
            os.replace(source_path, destination_path)
        else:
            source_path.rename(destination_path)
    except (OSError, shutil.Error) as exc:
        return {"ok": False, "error": {"code": "move_failed", "message": str(exc)}}
    return {
        "ok": True,
        "data": {
            "source": str(source_path.relative_to(workspace_root())),
            "destination": str(destination_path.relative_to(workspace_root())),
        },
    }


def search_files(
    query: str, directory: str = ".", max_results: int = 50
) -> dict[str, Any]:
    term = query.strip().casefold()
    if not term:
        raise ValueError("Search query cannot be empty.")
    if not 1 <= max_results <= 200:
        raise ValueError("max_results must be between 1 and 200.")

    base = resolve_workspace_path(directory)
    if not base.is_dir():
        return {
            "ok": False,
            "error": {"code": "not_a_directory", "message": "Search path is not a directory."},
        }

    matches: list[dict[str, Any]] = []
    scanned = 0
    skipped_files = 0
    walk_errors: list[str] = []
    for current, dirs, filenames in os.walk(
        base, followlinks=False, onerror=lambda error: walk_errors.append(str(error))
    ):
        dirs[:] = [
            name for name in dirs if not (Path(current) / name).is_symlink()
        ]
        for filename in filenames:
            file_path = Path(current) / filename
            if file_path.is_symlink():
                continue
            scanned += 1
            if scanned > MAX_SEARCH_FILES:
                return {
                    "ok": False,
                    "error": {
                        "code": "search_limit_reached",
                        "message": f"Search stopped after {MAX_SEARCH_FILES} files.",
                    },
                }
            try:
                if file_path.stat().st_size > MAX_FILE_BYTES:
                    continue
                with file_path.open("r", encoding="utf-8") as source_file:
                    for line_number, line in enumerate(source_file, start=1):
                        if term in line.casefold():
                            matches.append(
                                {
                                    "path": str(file_path.relative_to(workspace_root())),
                                    "line": line_number,
                                    "text": line.strip()[:500],
                                }
                            )
                            if len(matches) >= max_results:
                                return {
                                    "ok": True,
                                    "data": {
                                        "matches": matches,
                                        "skipped_files": skipped_files,
                                        "partial": True,
                                    },
                                }
            except (OSError, UnicodeError):
                skipped_files += 1
                continue
    data: dict[str, Any] = {
        "matches": matches,
        "skipped_files": skipped_files,
        "walk_errors": walk_errors,
    }
    if skipped_files or walk_errors:
        return {
            "ok": False,
            "error": {
                "code": "partial_search",
                "message": (
                    "Some files or folders could not be searched; see the partial "
                    "results for the skipped count and directory errors."
                ),
            },
            "data": data,
        }
    return {"ok": True, "data": data}
