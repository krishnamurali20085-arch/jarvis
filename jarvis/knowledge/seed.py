"""Command-line entry point for seeding approved Python tutorial documents."""

from __future__ import annotations

import json
import sys

from jarvis.knowledge.ingest import DocumentSource
from jarvis.knowledge.registry import get_verified_resource
from jarvis.knowledge.sources import download_python_docs
from jarvis.knowledge.store import KnowledgeBase, KnowledgeError


def seed_python_knowledge() -> list[dict[str, object]]:
    knowledge = KnowledgeBase()
    knowledge.prepare_model(allow_download=True)
    pages = download_python_docs(knowledge.root_dir)
    results = []
    for page in pages:
        resource = get_verified_resource(page.source_url)
        if resource is None:
            raise KnowledgeError(
                f"Python tutorial page is missing from the verified source registry: "
                f"{page.source_url}"
            )
        results.append(
            knowledge.ingest_file(
                page.relative_path,
                DocumentSource(
                    subject=resource.subject,
                    license_id=resource.license_id,
                    title=resource.title,
                    source_url=resource.source_url,
                    attribution=resource.attribution,
                ),
                allow_model_download=False,
            )
        )
    return results


def main() -> int:
    try:
        results = seed_python_knowledge()
    except (KnowledgeError, OSError, ValueError) as exc:
        print(f"Knowledge seeding failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(results, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
