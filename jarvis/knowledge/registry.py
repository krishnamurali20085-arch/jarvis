"""Exact educational resources approved for automatic local indexing."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class VerifiedResource:
    resource_id: str
    source_url: str
    title: str
    relative_path: str
    subject: str
    license_id: str
    attribution: str
    media_type: str


_PYTHON_ATTRIBUTION = (
    "Copyright (c) Python Software Foundation. Python Documentation, "
    "licensed under the Python Software Foundation License Version 2."
)

VERIFIED_RESOURCES = (
    VerifiedResource(
        "python-tutorial-index",
        "https://docs.python.org/3/tutorial/index.html",
        "Python Tutorial",
        "python/index.html",
        "Python programming",
        "PSF-2.0",
        _PYTHON_ATTRIBUTION,
        "text/html",
    ),
    VerifiedResource(
        "python-tutorial-introduction",
        "https://docs.python.org/3/tutorial/introduction.html",
        "An Informal Introduction to Python",
        "python/introduction.html",
        "Python programming",
        "PSF-2.0",
        _PYTHON_ATTRIBUTION,
        "text/html",
    ),
    VerifiedResource(
        "python-tutorial-controlflow",
        "https://docs.python.org/3/tutorial/controlflow.html",
        "More Control Flow Tools",
        "python/controlflow.html",
        "Python programming",
        "PSF-2.0",
        _PYTHON_ATTRIBUTION,
        "text/html",
    ),
    VerifiedResource(
        "python-tutorial-datastructures",
        "https://docs.python.org/3/tutorial/datastructures.html",
        "Data Structures",
        "python/datastructures.html",
        "Python programming",
        "PSF-2.0",
        _PYTHON_ATTRIBUTION,
        "text/html",
    ),
    VerifiedResource(
        "python-tutorial-modules",
        "https://docs.python.org/3/tutorial/modules.html",
        "Modules",
        "python/modules.html",
        "Python programming",
        "PSF-2.0",
        _PYTHON_ATTRIBUTION,
        "text/html",
    ),
    VerifiedResource(
        "python-tutorial-inputoutput",
        "https://docs.python.org/3/tutorial/inputoutput.html",
        "Input and Output",
        "python/inputoutput.html",
        "Python programming",
        "PSF-2.0",
        _PYTHON_ATTRIBUTION,
        "text/html",
    ),
    VerifiedResource(
        "python-tutorial-errors",
        "https://docs.python.org/3/tutorial/errors.html",
        "Errors and Exceptions",
        "python/errors.html",
        "Python programming",
        "PSF-2.0",
        _PYTHON_ATTRIBUTION,
        "text/html",
    ),
    VerifiedResource(
        "python-tutorial-classes",
        "https://docs.python.org/3/tutorial/classes.html",
        "Classes",
        "python/classes.html",
        "Python programming",
        "PSF-2.0",
        _PYTHON_ATTRIBUTION,
        "text/html",
    ),
    VerifiedResource(
        "python-tutorial-stdlib",
        "https://docs.python.org/3/tutorial/stdlib.html",
        "Brief Tour of the Standard Library",
        "python/stdlib.html",
        "Python programming",
        "PSF-2.0",
        _PYTHON_ATTRIBUTION,
        "text/html",
    ),
    VerifiedResource(
        "python-tutorial-venv",
        "https://docs.python.org/3/tutorial/venv.html",
        "Virtual Environments and Packages",
        "python/venv.html",
        "Python programming",
        "PSF-2.0",
        _PYTHON_ATTRIBUTION,
        "text/html",
    ),
)

_VERIFIED_BY_URL = {resource.source_url: resource for resource in VERIFIED_RESOURCES}


def get_verified_resource(url: str) -> VerifiedResource | None:
    """Return a registry entry only for its canonical HTTPS URL."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold() != "docs.python.org"
        or parsed.username
        or parsed.password
        or parsed.query
        or port not in {None, 443}
    ):
        return None
    canonical_url = urlunsplit(
        ("https", "docs.python.org", parsed.path or "/", "", "")
    )
    return _VERIFIED_BY_URL.get(canonical_url)
