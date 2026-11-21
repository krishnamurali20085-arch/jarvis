"""Bounded public downloads and automatic indexing for verified resources."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, unquote, urljoin, urlsplit, urlunsplit

import requests

from jarvis.config import get_settings
from jarvis.knowledge.ingest import DocumentSource
from jarvis.knowledge.registry import VerifiedResource, get_verified_resource
from jarvis.knowledge.store import KnowledgeBase, KnowledgeError
from jarvis.network import validate_public_url

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_REDIRECTS = 5
_USER_AGENT = "JarvisLocalAssistant/0.1 (verified educational resource indexer)"
_SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "key",
    "password",
    "secret",
    "signature",
    "token",
}
_MEDIA_SUFFIXES: dict[str, set[str]] = {
    "application/pdf": {".pdf"},
    "application/xhtml+xml": {".html", ".htm"},
    "text/html": {".html", ".htm"},
    "text/plain": {".txt", ".md"},
    "text/markdown": {".md"},
    "text/x-markdown": {".md"},
    "text/csv": {".csv"},
    "application/csv": {".csv"},
    "application/json": {".json"},
    "text/json": {".json"},
}

_manifest_lock = threading.RLock()


def _redact_url(url: str) -> str:
    parsed = urlsplit(url)
    safe_query = [
        (key, "[redacted]" if key.casefold() in _SENSITIVE_QUERY_KEYS else value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(safe_query), "")
    )


def _validate_download_url(url: str) -> str:
    safe_url = validate_public_url(url, require_https=True)
    query_keys = {key.casefold() for key, _ in parse_qsl(urlsplit(safe_url).query)}
    if query_keys & _SENSITIVE_QUERY_KEYS:
        raise ValueError(
            "URLs containing credential-like query parameters cannot be downloaded."
        )
    return safe_url


def download_confirmation_required(url: str) -> bool:
    """Return whether a download needs explicit app-level confirmation."""
    settings = get_settings()
    return not (
        settings.auto_index_approved_sources
        and get_verified_resource(url) is not None
    )


def _media_type_and_suffix(url: str, content_type: str) -> tuple[str, str]:
    media_type = content_type.split(";", 1)[0].strip().lower()
    suffixes = _MEDIA_SUFFIXES.get(media_type)
    if suffixes is None and media_type.endswith("+json"):
        suffixes = {".json"}
    if not suffixes:
        raise KnowledgeError(
            f"Unsupported download media type: {media_type or 'unknown'}."
        )

    url_suffix = Path(unquote(urlsplit(url).path)).suffix.casefold()
    if url_suffix:
        if url_suffix not in suffixes:
            raise KnowledgeError(
                "The downloaded media type does not match the URL file extension."
            )
        return media_type, url_suffix
    return media_type, sorted(suffixes)[0]


def _download_to_staging(
    url: str, resource: VerifiedResource | None
) -> tuple[Path, dict[str, Any]]:
    settings = get_settings()
    staging_dir = settings.download_staging_dir.resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    current_url = _validate_download_url(url)
    parsed_origin = urlsplit(current_url)
    original_origin = (parsed_origin.scheme, parsed_origin.netloc.casefold())
    temporary_path: Path | None = None

    try:
        for redirect_count in range(MAX_REDIRECTS + 1):
            with requests.get(
                current_url,
                headers={"User-Agent": _USER_AGENT},
                timeout=settings.request_timeout_seconds,
                stream=True,
                allow_redirects=False,
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("Location")
                    if not location:
                        raise KnowledgeError("Download redirected without a target URL.")
                    next_url = _validate_download_url(urljoin(current_url, location))
                    next_origin = (
                        urlsplit(next_url).scheme,
                        urlsplit(next_url).netloc.casefold(),
                    )
                    if resource is not None and next_url != resource.source_url:
                        raise KnowledgeError(
                            "Verified resource redirected away from its registered URL."
                        )
                    if resource is None and next_origin != original_origin:
                        raise KnowledgeError(
                            "Cross-origin downloads are blocked; use the final HTTPS URL."
                        )
                    if redirect_count == MAX_REDIRECTS:
                        raise KnowledgeError("Download redirected too many times.")
                    current_url = next_url
                    continue

                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                    except ValueError as exc:
                        raise KnowledgeError("Download returned an invalid Content-Length.") from exc
                    if declared_size < 0 or declared_size > MAX_DOWNLOAD_BYTES:
                        raise KnowledgeError(
                            f"Download exceeds the {MAX_DOWNLOAD_BYTES}-byte limit."
                        )

                media_type, suffix = _media_type_and_suffix(
                    current_url, response.headers.get("Content-Type", "")
                )
                if resource is not None and media_type not in {
                    resource.media_type,
                    "application/xhtml+xml",
                }:
                    raise KnowledgeError(
                        "Verified resource returned an unexpected media type."
                    )
                if resource is not None:
                    expected_suffix = Path(resource.relative_path).suffix.casefold()
                    if suffix != expected_suffix:
                        raise KnowledgeError(
                            "Verified resource returned an unexpected file extension."
                        )

                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=staging_dir,
                    prefix=".download-",
                    suffix=".part",
                    delete=False,
                ) as output:
                    temporary_path = Path(output.name)
                    digest = hashlib.sha256()
                    byte_count = 0
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        byte_count += len(chunk)
                        if byte_count > MAX_DOWNLOAD_BYTES:
                            raise KnowledgeError(
                                f"Download exceeds the {MAX_DOWNLOAD_BYTES}-byte limit."
                            )
                        digest.update(chunk)
                        output.write(chunk)
                if byte_count == 0:
                    raise KnowledgeError("Download returned an empty file.")

                identifier = (
                    resource.resource_id if resource is not None else uuid.uuid4().hex
                )
                destination = staging_dir / f"{identifier}{suffix}"
                os.replace(temporary_path, destination)
                temporary_path = None
                return destination, {
                    "download_id": identifier,
                    "source_url": _redact_url(url),
                    "final_url": _redact_url(current_url),
                    "content_sha256": digest.hexdigest(),
                    "byte_count": byte_count,
                    "media_type": media_type,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "registry_id": resource.resource_id if resource else None,
                    "license_id": resource.license_id if resource else None,
                    "attribution": resource.attribution if resource else None,
                    "authorization": (
                        "verified_registry"
                        if resource is not None
                        and settings.auto_index_approved_sources
                        else "user_confirmed"
                    ),
                    "status": "staged",
                    "staging_path": destination.name,
                    "indexed_path": None,
                }
    except requests.RequestException as exc:
        raise KnowledgeError("The public download failed due to a network error.") from exc
    except OSError as exc:
        raise KnowledgeError(f"Could not save downloaded file: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    raise KnowledgeError("Download did not produce a file.")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeError(f"Could not read download manifest: {exc}") from exc
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise KnowledgeError("Download manifest must contain a JSON list.")
    return records


def _write_manifest(staging_dir: Path, record: dict[str, Any]) -> None:
    manifest_path = staging_dir / "downloads.json"
    temporary_path: Path | None = None
    with _manifest_lock:
        records = _read_manifest(manifest_path)
        records = [
            existing
            for existing in records
            if existing.get("download_id") != record["download_id"]
        ]
        records.append(record)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=staging_dir,
                prefix=".downloads-",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                json.dump(records, output, ensure_ascii=False, indent=2)
                output.write("\n")
            os.replace(temporary_path, manifest_path)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise KnowledgeError(f"Could not update download manifest: {exc}") from exc


def _copy_for_indexing(
    staged_path: Path,
    resource: VerifiedResource,
    knowledge: KnowledgeBase,
) -> None:
    target_path = knowledge._resolve_document_path(resource.relative_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path = knowledge._resolve_document_path(resource.relative_path)
    backup_path: Path | None = None
    temporary_path: Path | None = None
    had_previous = target_path.exists()
    try:
        if had_previous:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target_path.parent,
                prefix=".previous-",
                suffix=".bak",
                delete=False,
            ) as backup:
                backup_path = Path(backup.name)
                with target_path.open("rb") as previous:
                    shutil.copyfileobj(previous, backup)

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_path.parent,
            prefix=".resource-",
            suffix=".tmp",
            delete=False,
        ) as staged_copy:
            temporary_path = Path(staged_copy.name)
            with staged_path.open("rb") as source:
                shutil.copyfileobj(source, staged_copy)
        os.replace(temporary_path, target_path)
        temporary_path = None
        try:
            knowledge.ingest_file(
                resource.relative_path,
                DocumentSource(
                    subject=resource.subject,
                    license_id=resource.license_id,
                    title=resource.title,
                    source_url=resource.source_url,
                    attribution=resource.attribution,
                ),
                allow_model_download=False,
            )
        except Exception:
            if backup_path is not None:
                os.replace(backup_path, target_path)
                backup_path = None
            elif not had_previous:
                target_path.unlink(missing_ok=True)
            raise
    finally:
        for path in (backup_path, temporary_path):
            if path is not None:
                path.unlink(missing_ok=True)


def download_resource(url: str) -> dict[str, Any]:
    """Download a public document; automatically index only exact verified sources."""
    safe_url = _validate_download_url(url)
    resource = get_verified_resource(safe_url)
    settings = get_settings()
    auto_index = resource is not None and settings.auto_index_approved_sources
    staged_path, record = _download_to_staging(safe_url, resource)
    _write_manifest(settings.download_staging_dir.resolve(), record)

    if auto_index and resource is not None:
        knowledge = KnowledgeBase()
        try:
            _copy_for_indexing(staged_path, resource, knowledge)
        except Exception as exc:
            logger.exception("Could not index verified educational download")
            record["status"] = "index_failed"
            _write_manifest(settings.download_staging_dir.resolve(), record)
            raise KnowledgeError(
                f"Download remains in staging because indexing failed: {exc}"
            ) from exc
        finally:
            knowledge.close()
        record["status"] = "indexed"
        record["indexed_path"] = resource.relative_path
        _write_manifest(settings.download_staging_dir.resolve(), record)

    return {
        "ok": True,
        "data": {
            "status": record["status"],
            "source_url": record["source_url"],
            "media_type": record["media_type"],
            "byte_count": record["byte_count"],
            "content_sha256": record["content_sha256"],
            "license_id": record["license_id"],
            "attribution": record["attribution"],
            "staging_path": record["staging_path"],
            "indexed_path": record["indexed_path"],
        },
    }
