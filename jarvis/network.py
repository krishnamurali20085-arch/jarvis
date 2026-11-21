"""Network URL validation shared by public-web tools."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit, urlunsplit


def validate_public_url(url: str, *, require_https: bool = False) -> str:
    """Normalize an absolute public URL and reject local network destinations."""
    value = url.strip()
    parsed = urlsplit(value)
    if not parsed.hostname:
        raise ValueError("An absolute public HTTP(S) URL is required.")
    if require_https and parsed.scheme.lower() != "https":
        raise ValueError("HTTPS is required for fetched and downloaded URLs.")
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only absolute HTTP(S) URLs are allowed.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not allowed.")
    if parsed.port is not None and parsed.port not in {80, 443}:
        raise ValueError("Only standard HTTP and HTTPS ports are allowed.")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise ValueError("Local network hostnames are not allowed.")
    scheme = parsed.scheme.lower()
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(
                    hostname, parsed.port or (443 if scheme == "https" else 80)
                )
            }
        except OSError as dns_error:
            raise ValueError(f"Could not resolve public web host {hostname}.") from dns_error
        if not addresses or any(
            not ipaddress.ip_address(address_item).is_global for address_item in addresses
        ):
            raise ValueError("The URL resolves to a private or non-public address.")
    else:
        if not address.is_global:
            raise ValueError("Private and non-public IP addresses are not allowed.")

    return urlunsplit(
        (scheme, parsed.netloc, parsed.path or "/", parsed.query, "")
    )
