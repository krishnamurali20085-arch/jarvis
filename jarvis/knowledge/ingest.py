"""Safe text extraction and chunking for local educational documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 2000
MAX_CHUNKS_PER_DOCUMENT = 10_000
DEFAULT_CHUNK_CHARS = 1200
DEFAULT_CHUNK_OVERLAP = 180
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".html", ".htm"}


class DocumentError(ValueError):
    """Raised when a source document is invalid or cannot be extracted."""


@dataclass(frozen=True)
class DocumentSource:
    subject: str
    license_id: str
    title: str | None = None
    source_url: str | None = None
    attribution: str | None = None


@dataclass(frozen=True)
class ParsedPage:
    page_number: int | None
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    content_sha256: str
    pages: tuple[ParsedPage, ...]


class _HTMLTextExtractor(HTMLParser):
    _IGNORED_TAGS = {"script", "style", "noscript", "svg", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._main_depth = 0
        self._in_title = False
        self._tag_stack: list[tuple[str, bool]] = []
        self.all_text: list[str] = []
        self.main_text: list[str] = []
        self.title_parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag_name = tag.lower()
        attributes = dict(attrs)
        if tag_name in self._IGNORED_TAGS:
            self._ignored_depth += 1
        is_main = tag_name == "main" or attributes.get("role") == "main"
        self._tag_stack.append((tag_name, is_main))
        if is_main:
            self._main_depth += 1
        if tag_name == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.lower()
        if tag_name in self._IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        if tag_name == "title":
            self._in_title = False
        for index in range(len(self._tag_stack) - 1, -1, -1):
            stack_tag, is_main = self._tag_stack[index]
            if stack_tag == tag_name:
                removed = self._tag_stack[index:]
                del self._tag_stack[index:]
                self._main_depth -= sum(marker for _, marker in removed)
                break

    def handle_data(self, data: str) -> None:
        if self._in_title:
            cleaned_title = " ".join(data.split())
            if cleaned_title:
                self.title_parts.append(cleaned_title)
        if self._ignored_depth:
            return
        cleaned = " ".join(data.split())
        if cleaned:
            self.all_text.append(cleaned)
            if self._main_depth:
                self.main_text.append(cleaned)


def parse_document(path: Path) -> ParsedDocument:
    """Extract text from a supported file and compute its content hash."""
    suffix = path.suffix.casefold()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise DocumentError(f"Unsupported file type {suffix!r}; supported: {supported}.")
    try:
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise DocumentError(
                f"{path.name} exceeds the {MAX_DOCUMENT_BYTES}-byte document limit."
            )
        raw = path.read_bytes()
    except OSError as exc:
        raise DocumentError(f"Could not read {path.name}: {exc}") from exc
    if not raw:
        raise DocumentError(f"{path.name} is empty.")
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise DocumentError(
            f"{path.name} exceeds the {MAX_DOCUMENT_BYTES}-byte document limit."
        )

    title = path.stem
    pages: list[ParsedPage] = []
    if suffix == ".pdf":
        from io import BytesIO

        from pypdf import PdfReader

        try:
            reader = PdfReader(BytesIO(raw), strict=False)
            if reader.is_encrypted:
                raise DocumentError(f"{path.name} is encrypted.")
            if len(reader.pages) > MAX_PDF_PAGES:
                raise DocumentError(
                    f"{path.name} exceeds the {MAX_PDF_PAGES}-page limit."
                )
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(ParsedPage(page_number, text))
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError(f"Could not extract PDF text from {path.name}: {exc}") from exc
    elif suffix in {".html", ".htm"}:
        extractor = _HTMLTextExtractor()
        try:
            extractor.feed(raw.decode("utf-8-sig", errors="replace"))
            extractor.close()
        except Exception as exc:
            raise DocumentError(f"Could not parse HTML in {path.name}: {exc}") from exc
        title = " ".join(extractor.title_parts).strip() or title
        text = " ".join(extractor.main_text or extractor.all_text)
        if text:
            pages.append(ParsedPage(None, text))
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentError(f"{path.name} is not valid UTF-8 text.") from exc
        if text.strip():
            pages.append(ParsedPage(None, text))

    if not pages:
        raise DocumentError(f"No readable text was found in {path.name}.")
    return ParsedDocument(
        title=title,
        content_sha256=hashlib.sha256(raw).hexdigest(),
        pages=tuple(pages),
    )


def chunk_document(
    document: ParsedDocument,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, str | int]]:
    """Split document text into bounded overlapping chunks with page metadata."""
    if max_chars < 200 or max_chars > 5000:
        raise ValueError("max_chars must be between 200 and 5000.")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars.")

    chunks: list[dict[str, str | int]] = []
    for page in document.pages:
        text = re.sub(r"[ \t]+", " ", page.text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                lower_bound = start + max_chars // 2
                boundaries = (
                    text.rfind("\n", lower_bound, end),
                    text.rfind(". ", lower_bound, end),
                    text.rfind(" ", lower_bound, end),
                )
                boundary = max(boundaries)
                if boundary > start:
                    end = boundary + (1 if text[boundary] == "." else 0)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk: dict[str, str | int] = {"text": chunk_text}
                if page.page_number is not None:
                    chunk["page_number"] = page.page_number
                chunks.append(chunk)
            if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
                raise DocumentError(
                    f"Document exceeds {MAX_CHUNKS_PER_DOCUMENT} text chunks."
                )
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
