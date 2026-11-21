"""Tests for local educational document ingestion and citation retrieval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis.knowledge import DocumentSource, KnowledgeBase
from jarvis.knowledge.ingest import chunk_document, parse_document
from jarvis.knowledge.sources import _validate_source_url
from jarvis.knowledge.store import KnowledgeError
from jarvis.tools import _knowledge_search, get_tool_schemas


class _FakeCollection:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, object]] = {}

    def get(self, where: dict[str, str], include: list[str]) -> dict[str, object]:
        items = [
            (identifier, item)
            for identifier, item in self.items.items()
            if item["metadata"]["document_id"] == where["document_id"]
        ]
        return {"ids": [identifier for identifier, _ in items]}

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]],
    ) -> None:
        for identifier, document, embedding, metadata in zip(
            ids, documents, embeddings, metadatas
        ):
            self.items[identifier] = {
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }

    def delete(self, ids: list[str]) -> None:
        for identifier in ids:
            self.items.pop(identifier, None)

    def count(self) -> int:
        return len(self.items)

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
        where: dict[str, str] | None = None,
    ) -> dict[str, list[list[object]]]:
        matching = [
            (identifier, item)
            for identifier, item in self.items.items()
            if where is None or item["metadata"]["subject"] == where["subject"]
        ][:n_results]
        return {
            "ids": [[identifier for identifier, _ in matching]],
            "documents": [[item["document"] for _, item in matching]],
            "metadatas": [[item["metadata"] for _, item in matching]],
            "distances": [[0.1 for _ in matching]],
        }


class KnowledgeTests(unittest.TestCase):
    def test_html_extraction_ignores_scripts_and_chunks_with_title(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lesson.html"
            path.write_text(
                "<html><head><title>Lesson</title></head>"
                "<body><nav>Navigation</nav>"
                "<div role='main'><h1>Functions</h1><p>Useful lesson text.</p>"
                "<script>ignore this</script><div><p>More lesson text.</p></div>"
                "</div></body></html>",
                encoding="utf-8",
            )
            document = parse_document(path)
            chunks = chunk_document(document, max_chars=200, overlap=20)

        text = " ".join(chunk["text"] for chunk in chunks)
        self.assertEqual(document.title, "Lesson")
        self.assertIn("Useful lesson text", text)
        self.assertIn("More lesson text", text)
        self.assertNotIn("ignore this", text)
        self.assertNotIn("Navigation", text)

    def test_ingestion_is_repeatable_and_search_returns_source_citation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = root / "documents"
            documents.mkdir()
            (documents / "lesson.md").write_text(
                "# Algebra\nLinear equations keep both sides balanced.",
                encoding="utf-8",
            )
            knowledge = KnowledgeBase(root_dir=root)
            collection = _FakeCollection()
            source = DocumentSource(
                subject="Algebra",
                license_id="CC-BY-4.0",
                title="Algebra lesson",
                source_url="https://example.org/algebra",
                attribution="Example Open Education",
            )
            with (
                patch.object(knowledge, "_get_collection", return_value=collection),
                patch.object(
                    knowledge,
                    "_embed",
                    side_effect=lambda texts, allow_download=False: [
                        [1.0, 0.0] for _ in texts
                    ],
                ),
            ):
                first = knowledge.ingest_file("lesson.md", source)
                second = knowledge.ingest_file("lesson.md", source)
                search = knowledge.search("balanced equation", subject="algebra")

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual(first["data"]["document_id"], second["data"]["document_id"])
        self.assertEqual(collection.count(), 1)
        match = search["data"]["matches"][0]
        self.assertIn("Algebra lesson", match["citation"])
        self.assertEqual(match["license_id"], "CC-BY-4.0")

    def test_ingestion_rejects_traversal_missing_license_and_unapproved_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            knowledge = KnowledgeBase(root_dir=root)
            with self.assertRaises(ValueError):
                knowledge._resolve_document_path("../outside.txt")
            with self.assertRaises(ValueError):
                knowledge._validate_source(DocumentSource(subject="Python", license_id=""))

        with self.assertRaises(KnowledgeError):
            _validate_source_url("https://example.org/lesson.html")

    def test_chunk_parameters_are_bounded(self) -> None:
        from jarvis.knowledge.ingest import ParsedDocument, ParsedPage

        document = ParsedDocument("test", "hash", (ParsedPage(None, "A" * 1000),))
        with self.assertRaises(ValueError):
            chunk_document(document, max_chars=100, overlap=10)

    def test_knowledge_search_is_exposed_as_a_read_only_tool(self) -> None:
        schemas = get_tool_schemas()
        knowledge_search = next(
            schema for schema in schemas if schema["name"] == "knowledge_search"
        )
        self.assertEqual(knowledge_search["parameters"]["required"], ["query"])
        self.assertEqual(
            set(knowledge_search["parameters"]["properties"]), {"query"}
        )
        self.assertFalse(knowledge_search["parameters"]["additionalProperties"])

    def test_knowledge_search_treats_empty_subject_as_unfiltered(self) -> None:
        with patch("jarvis.tools._knowledge_base") as knowledge_base:
            knowledge_base.return_value.search.return_value = {"ok": True}
            result = _knowledge_search("functions", top_k=3, subject="")

        self.assertTrue(result["ok"])
        knowledge_base.return_value.search.assert_called_once_with(
            "functions", 3, None
        )

    def test_persistent_index_releases_database_lock_when_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            knowledge = KnowledgeBase(root_dir=temporary)
            knowledge._get_collection()
            knowledge.close()


if __name__ == "__main__":
    unittest.main()
