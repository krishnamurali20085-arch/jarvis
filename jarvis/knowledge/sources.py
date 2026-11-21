"""Downloads for explicitly approved educational source material."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests

from jarvis.config import get_settings
from jarvis.knowledge.registry import (
    VERIFIED_RESOURCES,
    VerifiedResource,
    get_verified_resource,
)
from jarvis.knowledge.store import KnowledgeError

_MAX_PAGE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class DownloadedPage:
    relative_path: str
    source_url: str
    title: str


def _validate_source_url(url: str) -> str:
    if get_verified_resource(url) is None:
        raise KnowledgeError("Educational downloads are restricted to approved Python docs pages.")
    return url


def _download_html(url: str, timeout: float) -> bytes:
    current_url = _validate_source_url(url)
    for _ in range(4):
        try:
            with requests.get(
                current_url,
                headers={"User-Agent": "JarvisLocalAssistant/0.1 educational indexer"},
                timeout=timeout,
                stream=True,
                allow_redirects=False,
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("Location")
                    if not location:
                        raise KnowledgeError("Python docs returned a redirect without a target.")
                    current_url = _validate_source_url(urljoin(current_url, location))
                    continue
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                    raise KnowledgeError(
                        f"Expected an HTML source page, received {content_type or 'unknown'}."
                    )
                body = bytearray()
                for chunk in response.iter_content(chunk_size=16_384):
                    if not chunk:
                        continue
                    if len(body) + len(chunk) > _MAX_PAGE_BYTES:
                        raise KnowledgeError(
                            f"Python docs page exceeds the {_MAX_PAGE_BYTES}-byte limit."
                        )
                    body.extend(chunk)
                if not body:
                    raise KnowledgeError("Python docs returned an empty page.")
                return bytes(body)
        except requests.RequestException as exc:
            raise KnowledgeError(f"Could not download approved Python docs page: {exc}") from exc
    raise KnowledgeError("Python docs redirected too many times.")


def download_python_docs(
    knowledge_dir: str | Path | None = None,
) -> list[DownloadedPage]:
    """Download a fixed set of Python tutorial pages from the verified official host."""
    settings = get_settings()
    root = Path(knowledge_dir or settings.knowledge_dir).expanduser().resolve()
    documents_root = root / "documents"
    try:
        documents_root = documents_root.resolve()
        documents_root.relative_to(root)
    except ValueError as exc:
        raise KnowledgeError("Python docs destination resolves outside the knowledge store.") from exc
    timeout = settings.request_timeout_seconds
    downloaded: list[DownloadedPage] = []
    for resource in VERIFIED_RESOURCES:
        body = _download_html(resource.source_url, timeout)
        destination = _resource_path(documents_root, resource)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.parent.resolve().relative_to(documents_root.resolve())
        except ValueError as exc:
            raise KnowledgeError(
                "Python docs destination resolves outside the document store."
            ) from exc
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{resource.resource_id}-",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                output.write(body)
            os.replace(temporary_path, destination)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise KnowledgeError(
                f"Could not save Python docs page {resource.resource_id}: {exc}"
            ) from exc

        downloaded.append(
            DownloadedPage(
                relative_path=resource.relative_path,
                source_url=resource.source_url,
                title=resource.title,
            )
        )
    return downloaded


def _resource_path(documents_root: Path, resource: VerifiedResource) -> Path:
    candidate = Path(resource.relative_path)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise KnowledgeError("Verified source registry contains an unsafe document path.")
    return documents_root / candidate
