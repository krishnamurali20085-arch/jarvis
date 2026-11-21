# JARVIS Educational Knowledge and Training Design

## Goal and terminology

Add educational knowledge to JARVIS in two distinct stages:

1. **Local retrieval-augmented generation (RAG):** index educational documents
   locally, retrieve relevant passages for a question, and return source
   citations to the assistant. This changes neither the embedding model nor the
   chat model's weights.
2. **Local fine-tuning (later):** optionally adapt a selected local chat model
   using curated, appropriately licensed question-and-answer examples. This is
   a separate project step that requires a model, hardware, and dataset decision.

The initial subject areas are Python programming, algebra, physics, biology, and
general study skills. Fine-tuning raw textbook pages is out of scope; textbooks
are source material for RAG, not automatically valid training examples.

## RAG architecture

Add a `jarvis/knowledge/` package with separate ingestion, indexing, and
retrieval responsibilities:

- **Seed command:** `python -m jarvis.knowledge.seed` downloads a fixed set of
  Python tutorial pages from `docs.python.org`, verifies the HTTPS host and
  redirect targets, prepares the approved embedding model, and indexes the
  pages. This initial seed covers Python only.
- **Ingestion:** support PDF, TXT, Markdown, and HTML files. Extract readable
  text, normalize it, split it into bounded overlapping chunks, and preserve
  document provenance.
- **Index:** use ChromaDB with local persistence and the
  `sentence-transformers/all-MiniLM-L6-v2` embedding model. Keep the index,
  embedding cache, and source files on the local machine. The model card lists
  the model under Apache-2.0.
- **Retrieval:** embed the user's question locally, return a bounded set of
  relevant chunks with source title, locator/page, source URL, and license
  metadata. The future brain may use those chunks to answer, but must cite
  retrieved sources and distinguish missing evidence from model knowledge.
- **Tool boundary:** expose read-only knowledge search and a guarded
  `download_resource` tool through the existing registry. Downloads matching
  an exact, license-verified source-registry entry may be indexed automatically
  when `JARVIS_AUTO_INDEX_APPROVED_SOURCES` is enabled. Other downloads require
  application confirmation and remain in staging; they are not automatically
  indexed or used for fine-tuning.

Proposed data layout:

```text
data/
  educational/
    documents/   # source files
    index/       # local ChromaDB data
    metadata/    # source/license manifest
```

Add the data directory and downloaded model artifacts to `.gitignore`. Store
metadata for each source: title, subject, source URL, license identifier,
required attribution, retrieval time, and content hash. Never infer a license
from a file extension or dataset name.

## Source and licensing policy

- Python's official documentation states that Python documentation is licensed
  under the Python Software Foundation License Version 2. The official license
  page is https://docs.python.org/3/license.html.
- OpenStax is a candidate for algebra, physics, biology, and study-skills
  materials. Its official licensing-information page was returned in web search,
  but direct retrieval was blocked by the current environment's URL safety
  filter. Do not download or ingest OpenStax material until its current
  book-specific license and attribution requirements can be verified. User-
  supplied files may be indexed only after their source and usage rights are
  recorded.
- The proposed embedding model's Hugging Face model card identifies its license
  as Apache-2.0:
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/main/README.md

Every downloaded corpus must come from an allowlisted, verified source and
retain attribution. No paywalled, restricted, or ambiguously licensed material
will be downloaded. RAG retrieval runs locally after document/model download;
questions and document text are not sent to a vector-service provider.

## Fine-tuning boundary

Fine-tuning is deferred until the local base model, available CPU/GPU memory,
operating environment, and dataset license are selected. Its data must be
curated question-and-answer examples rather than untransformed textbook copies.
The eventual implementation should use parameter-efficient tuning (LoRA or
QLoRA) where supported, keep training data and output adapters local, and never
change the configured API provider's remote model.

Fine-tuning dependencies and large model/dataset downloads are not part of the
RAG implementation. Keep this workflow in a separate training package or
environment so optional GPU dependencies do not destabilize the assistant's
Python 3.10+ runtime.

## Errors and safety

Reject unsupported formats, oversized inputs, malformed files, and paths outside
the configured educational-data directory with explicit errors. Network source
downloads must use HTTPS, enforce size and timeout limits, verify the expected
host, record source/license metadata, and never execute downloaded content.
Retrieved passages are untrusted data and must not override system or user
instructions.

## Verification

RAG tests should verify supported document extraction, chunk provenance,
repeat-ingestion behavior, local persistence across restarts, citation metadata,
path confinement, and that retrieval works without network access after
initialization. Test fine-tuning separately after selecting a model and
hardware; do not make GPU availability a requirement for using the RAG assistant.
