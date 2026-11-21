"""Tests for guarded downloads and automatic indexing policy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jarvis.knowledge import downloads
from jarvis.knowledge.registry import VERIFIED_RESOURCES
from jarvis.knowledge.store import KnowledgeError
from jarvis.tools import execute_tool


class _Response:
    ok = True
    status_code = 200
    is_redirect = False

    def __init__(
        self,
        body: bytes = b"<html><main>Lesson</main></html>",
        content_type: str = "text/html",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.body = body
        self.headers = headers or {"Content-Type": content_type}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self.body]


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.staging = Path(self.temporary.name) / "download_staging"
        self.settings = SimpleNamespace(
            download_staging_dir=self.staging,
            request_timeout_seconds=2.0,
            auto_index_approved_sources=True,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_unknown_download_requires_confirmation_and_only_stages(self) -> None:
        url = "https://example.org/lesson.pdf"
        staged_path = self.staging / "manual.pdf"
        staged_path.parent.mkdir(parents=True)
        staged_path.write_bytes(b"pdf")
        record = {
            "download_id": "manual",
            "source_url": url,
            "media_type": "application/pdf",
            "byte_count": 3,
            "content_sha256": "hash",
            "license_id": None,
            "attribution": None,
            "status": "staged",
            "staging_path": staged_path.name,
            "indexed_path": None,
        }
        with (
            patch.object(downloads, "get_settings", return_value=self.settings),
            patch.object(downloads, "get_verified_resource", return_value=None),
            patch.object(downloads, "validate_public_url", side_effect=lambda value, **_: value),
            patch.object(downloads, "_download_to_staging", return_value=(staged_path, record)),
            patch.object(downloads, "_write_manifest"),
        ):
            denied = execute_tool("download_resource", {"url": url})
            self.assertEqual(denied["error"]["code"], "confirmation_required")
            staged = execute_tool("download_resource", {"url": url}, confirmed=True)

        self.assertTrue(staged["ok"])
        self.assertEqual(staged["data"]["status"], "staged")
        self.assertIsNone(staged["data"]["indexed_path"])
        self.assertTrue(staged_path.exists())

    def test_verified_registry_download_is_auto_indexed_when_enabled(self) -> None:
        resource = VERIFIED_RESOURCES[0]
        staged_path = self.staging / "verified.html"
        staged_path.parent.mkdir(parents=True)
        staged_path.write_text("<html><main>Lesson</main></html>", encoding="utf-8")
        record = {
            "download_id": resource.resource_id,
            "source_url": resource.source_url,
            "media_type": "text/html",
            "byte_count": 30,
            "content_sha256": "hash",
            "license_id": resource.license_id,
            "attribution": resource.attribution,
            "status": "staged",
            "staging_path": staged_path.name,
            "indexed_path": None,
        }
        with (
            patch.object(downloads, "get_settings", return_value=self.settings),
            patch.object(downloads, "get_verified_resource", return_value=resource),
            patch.object(downloads, "validate_public_url", side_effect=lambda value, **_: value),
            patch.object(downloads, "_download_to_staging", return_value=(staged_path, record)),
            patch.object(downloads, "_write_manifest"),
            patch.object(downloads, "_copy_for_indexing") as index_resource,
            patch.object(downloads, "KnowledgeBase") as knowledge_base,
        ):
            result = execute_tool(
                "download_resource", {"url": resource.source_url}
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["status"], "indexed")
        index_resource.assert_called_once()
        knowledge_base.return_value.close.assert_called_once()

    def test_disabling_auto_index_keeps_verified_download_in_staging(self) -> None:
        resource = VERIFIED_RESOURCES[0]
        self.settings.auto_index_approved_sources = False
        staged_path = self.staging / "verified.html"
        staged_path.parent.mkdir(parents=True)
        staged_path.write_text("<html><main>Lesson</main></html>", encoding="utf-8")
        record = {
            "download_id": resource.resource_id,
            "source_url": resource.source_url,
            "media_type": "text/html",
            "byte_count": 30,
            "content_sha256": "hash",
            "license_id": resource.license_id,
            "attribution": resource.attribution,
            "status": "staged",
            "staging_path": staged_path.name,
            "indexed_path": None,
        }
        with (
            patch.object(downloads, "get_settings", return_value=self.settings),
            patch.object(downloads, "get_verified_resource", return_value=resource),
            patch.object(downloads, "validate_public_url", side_effect=lambda value, **_: value),
            patch.object(downloads, "_download_to_staging", return_value=(staged_path, record)),
            patch.object(downloads, "_write_manifest"),
            patch.object(downloads, "_copy_for_indexing") as index_resource,
        ):
            denied = execute_tool("download_resource", {"url": resource.source_url})
            self.assertEqual(denied["error"]["code"], "confirmation_required")
            result = execute_tool(
                "download_resource",
                {"url": resource.source_url},
                confirmed=True,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["status"], "staged")
        index_resource.assert_not_called()

    def test_streaming_download_enforces_size_and_media_type(self) -> None:
        url = "https://example.org/lesson.pdf"
        with (
            patch.object(downloads, "get_settings", return_value=self.settings),
            patch.object(downloads, "_validate_download_url", return_value=url),
            patch.object(downloads, "MAX_DOWNLOAD_BYTES", 2),
            patch.object(
                downloads.requests,
                "get",
                return_value=_Response(body=b"123", content_type="application/pdf"),
            ),
        ):
            with self.assertRaises(KnowledgeError):
                downloads._download_to_staging(url, None)

        self.assertEqual(list(self.staging.glob("*.part")), [])

    def test_download_url_rejects_non_https_and_private_addresses(self) -> None:
        with self.assertRaises(ValueError):
            downloads._validate_download_url("http://example.org/lesson.pdf")
        with self.assertRaises(ValueError):
            downloads._validate_download_url("https://127.0.0.1/lesson.pdf")

    def test_verified_registry_matches_canonical_url_without_fragment_only(self) -> None:
        resource = VERIFIED_RESOURCES[0]
        self.assertIs(
            downloads.get_verified_resource(resource.source_url + "#section"),
            resource,
        )
        self.assertIsNone(
            downloads.get_verified_resource(resource.source_url + "?download=1")
        )

    def test_download_tool_is_registered(self) -> None:
        from jarvis.tools import get_tool_schemas

        schema = next(
            item for item in get_tool_schemas() if item["name"] == "download_resource"
        )
        self.assertEqual(schema["parameters"]["required"], ["url"])


if __name__ == "__main__":
    unittest.main()
