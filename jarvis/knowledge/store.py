"""Local ChromaDB storage and semantic retrieval for educational documents."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jarvis.config import get_settings
from jarvis.knowledge.ingest import (
    DocumentError,
    DocumentSource,
    chunk_document,
    parse_document,
)

logger = logging.getLogger(__name__)


class KnowledgeError(RuntimeError):
    """Raised when the local knowledge base cannot complete an operation."""


class KnowledgeBase:
    """Manage a local document corpus, embedding model, and ChromaDB index."""

    def __init__(
        self,
        root_dir: str | Path | None = None,
        embedding_model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.root_dir = Path(root_dir or settings.knowledge_dir).expanduser().resolve()
        self.documents_dir = self.root_dir / "documents"
        self.index_dir = self.root_dir / "index"
        self.models_dir = self.root_dir / "models"
        self.metadata_dir = self.root_dir / "metadata"
        self.manifest_path = self.metadata_dir / "sources.json"
        self.embedding_model = embedding_model or settings.embedding_model
        self._client: Any | None = None
        self._collection: Any | None = None
        self._model: Any | None = None
        self._lock = threading.RLock()

    def close(self) -> None:
        """Close the persistent vector client and release its SQLite file lock."""
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
                self._collection = None

    def _resolve_internal_path(self, path: Path) -> Path:
        root = self.root_dir.resolve()
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise KnowledgeError(
                f"Knowledge-base path resolves outside the configured data directory: {path}"
            ) from exc
        return resolved

    def _resolve_document_path(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("A document path relative to the documents folder is required.")
        candidate = Path(relative_path.replace("\\", "/"))
        if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
            raise ValueError("Document paths must stay inside the documents folder.")
        root = self._resolve_internal_path(self.documents_dir)
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("Document path resolves outside the documents folder.") from exc
        return resolved

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            index_dir = self._resolve_internal_path(self.index_dir)
            index_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(index_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=(
                    "jarvis_educational_"
                    + hashlib.sha256(self.embedding_model.encode("utf-8")).hexdigest()[:12]
                ),
                metadata={"hnsw:space": "cosine"},
            )
        except KnowledgeError:
            raise
        except Exception as exc:
            logger.exception("Could not open the local educational index")
            raise KnowledgeError(f"Could not open local knowledge index: {exc}") from exc
        return self._collection

    def prepare_model(self, allow_download: bool = False) -> str:
        """Load the approved local embedding model, optionally downloading it once."""
        with self._lock:
            if self._model is not None:
                return str(getattr(self._model, "device", "loaded"))
            try:
                from huggingface_hub import snapshot_download
                from sentence_transformers import SentenceTransformer

                models_dir = self._resolve_internal_path(self.models_dir)
                models_dir.mkdir(parents=True, exist_ok=True)
                model_path = snapshot_download(
                    repo_id=self.embedding_model,
                    cache_dir=str(models_dir),
                    local_files_only=not allow_download,
                )
                self._model = SentenceTransformer(str(model_path), device="cpu")
            except Exception as exc:
                action = (
                    "Could not download the embedding model"
                    if allow_download
                    else "The embedding model is not cached locally; seed the knowledge base online first"
                )
                raise KnowledgeError(f"{action}: {exc}") from exc
            return str(getattr(self._model, "device", "cpu"))

    def _embed(self, texts: list[str], allow_download: bool = False) -> list[list[float]]:
        self.prepare_model(allow_download=allow_download)
        try:
            model = self._model
            if model is None:
                raise KnowledgeError("The local embedding model did not initialize.")
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vectors.tolist()
        except Exception as exc:
            raise KnowledgeError(f"Could not create local text embeddings: {exc}") from exc

    @staticmethod
    def _validate_source(source: DocumentSource) -> None:
        if not source.subject.strip() or len(source.subject) > 100:
            raise ValueError("A document subject is required.")
        if not source.license_id.strip() or len(source.license_id) > 200:
            raise ValueError("A license identifier is required for every source.")
        if (
            not source.attribution
            or not source.attribution.strip()
            or len(source.attribution) > 1000
        ):
            raise ValueError("Attribution or an explicit no-attribution note is required.")
        if source.source_url:
            source_url = source.source_url.strip()
            if len(source_url) > 2000:
                raise ValueError("Source URLs cannot exceed 2000 characters.")
            parsed = urlparse(source_url)
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError("Source URLs must be absolute HTTPS URLs.")
            if parsed.username or parsed.password:
                raise ValueError("Source URLs cannot contain credentials.")

    def _read_manifest(self) -> list[dict[str, Any]]:
        manifest_path = self._resolve_internal_path(self.manifest_path)
        if not manifest_path.exists():
            return []
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KnowledgeError(f"Could not read source manifest: {exc}") from exc
        if not isinstance(raw_manifest, list):
            raise KnowledgeError("Source manifest must contain a JSON list.")
        return raw_manifest

    def _write_manifest(self, records: list[dict[str, Any]]) -> None:
        metadata_dir = self._resolve_internal_path(self.metadata_dir)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._resolve_internal_path(self.manifest_path)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=metadata_dir,
                prefix=".sources-",
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
            raise KnowledgeError(f"Could not update source manifest: {exc}") from exc

    def ingest_file(
        self,
        relative_path: str,
        source: DocumentSource,
        *,
        allow_model_download: bool = False,
    ) -> dict[str, Any]:
        """Extract, embed, and index one approved source file."""
        self._validate_source(source)
        path = self._resolve_document_path(relative_path)
        try:
            path.relative_to(self.documents_dir.resolve())
        except ValueError as exc:
            raise ValueError("Document path resolves outside the documents folder.") from exc
        if not path.is_file():
            raise DocumentError(f"Document does not exist: {relative_path}.")

        parsed = parse_document(path)
        chunks = chunk_document(parsed)
        if not chunks:
            raise DocumentError("No searchable text chunks were produced.")

        normalized_path = path.relative_to(
            self._resolve_internal_path(self.documents_dir)
        ).as_posix()
        document_id = hashlib.sha256(normalized_path.casefold().encode("utf-8")).hexdigest()
        title = source.title.strip() if source.title and source.title.strip() else parsed.title
        if len(title) > 500:
            raise ValueError("Document titles cannot exceed 500 characters.")
        source_url = source.source_url.strip() if source.source_url else None
        attribution = source.attribution.strip() if source.attribution else None
        base_metadata: dict[str, Any] = {
            "document_id": document_id,
            "file_path": normalized_path,
            "title": title,
            "subject": source.subject.strip().casefold(),
            "license_id": source.license_id.strip(),
            "content_sha256": parsed.content_sha256,
        }
        if source_url:
            base_metadata["source_url"] = source_url
        if attribution:
            base_metadata["attribution"] = attribution

        documents = [str(chunk["text"]) for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = dict(base_metadata)
            if "page_number" in chunk:
                metadata["page_number"] = int(chunk["page_number"])
            metadatas.append(metadata)
        ids = [f"{document_id}-{index}" for index in range(len(chunks))]

        with self._lock:
            collection = self._get_collection()
            old_ids = collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            ).get("ids", [])
            embeddings = self._embed(documents, allow_download=allow_model_download)
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            stale_ids = [identifier for identifier in old_ids if identifier not in ids]
            if stale_ids:
                collection.delete(ids=stale_ids)

            record = {
                **base_metadata,
                "source_url": source_url,
                "attribution": attribution,
                "title": title,
                "chunk_count": len(chunks),
                "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            manifest = self._read_manifest()
            manifest = [
                existing
                for existing in manifest
                if existing.get("document_id") != document_id
            ]
            manifest.append(record)
            self._write_manifest(manifest)

        return {
            "ok": True,
            "data": {
                "document_id": document_id,
                "title": title,
                "subject": base_metadata["subject"],
                "chunks_indexed": len(chunks),
                "content_sha256": parsed.content_sha256,
            },
        }

    def search(
        self, query: str, top_k: int = 5, subject: str | None = None
    ) -> dict[str, Any]:
        """Retrieve local passages with source citations."""
        term = query.strip()
        if not term:
            raise ValueError("Knowledge query cannot be empty.")
        if len(term) > 2000:
            raise ValueError("Knowledge query cannot exceed 2000 characters.")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10.")
        if subject is not None and not subject.strip():
            raise ValueError("Subject cannot be empty when provided.")

        with self._lock:
            collection = self._get_collection()
            count = collection.count()
            if count == 0:
                return {
                    "ok": False,
                    "error": {
                        "code": "knowledge_empty",
                        "message": (
                            "No educational documents are indexed. Add licensed "
                            "documents and run the local indexing operation first."
                        ),
                    },
                }
            query_embedding = self._embed([term])[0]
            query_options: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": min(top_k, count),
                "include": ["documents", "metadatas", "distances"],
            }
            if subject:
                query_options["where"] = {"subject": subject.strip().casefold()}
            results = collection.query(**query_options)

        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        matches = []
        for identifier, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            metadata = metadata or {}
            title = str(metadata.get("title", "Untitled source"))
            page_number = metadata.get("page_number")
            citation = title
            if page_number is not None:
                citation += f", page {page_number}"
            source_url = metadata.get("source_url")
            if source_url:
                citation += f" — {source_url}"
            matches.append(
                {
                    "text": text,
                    "citation": citation,
                    "title": title,
                    "subject": metadata.get("subject"),
                    "license_id": metadata.get("license_id"),
                    "attribution": metadata.get("attribution"),
                    "source_url": source_url,
                    "file_path": metadata.get("file_path"),
                    "page_number": page_number,
                    "distance": float(distance),
                    "chunk_id": identifier,
                }
            )
        return {
            "ok": True,
            "data": {
                "query": term,
                "matches": matches,
                "embedding_model": self.embedding_model,
            },
        }
