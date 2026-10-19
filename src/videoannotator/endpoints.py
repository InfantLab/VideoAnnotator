"""Whether a model server is on this computer or local network (spec 027 R9).

Decided on the server, where frames are sent from, by the addresses the host
resolves to. The networks are listed explicitly rather than taken from
`ipaddress.is_private`, whose meaning changed in Python 3.13 for some ranges.
A host that doesn't resolve is remote: warn rather than guess.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

EndpointClass = Literal["local", "remote"]

LOCAL_NETWORKS = [
    ipaddress.ip_network(n)
    for n in (
        "127.0.0.0/8",
        "::1/128",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "fc00::/7",
        "169.254.0.0/16",
        "fe80::/10",
    )
]


@dataclass(frozen=True)
class EndpointInfo:
    host: str
    cls: EndpointClass


def normalise_base_url(url: str) -> str:
    """Scheme, lower-case host and port, without a trailing slash or path."""
    parts = urlsplit(url.strip() if "://" in url else f"http://{url.strip()}")
    host = (parts.hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    port = f":{parts.port}" if parts.port else ""
    path = parts.path.rstrip("/")
    return f"{parts.scheme.lower()}://{host}{port}{path}"


def _is_local(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return any(ip in net for net in LOCAL_NETWORKS)


def classify_endpoint(base_url: str) -> EndpointInfo:
    parts = urlsplit(normalise_base_url(base_url))
    host = parts.hostname or ""
    if not host:
        return EndpointInfo(host, "remote")
    try:
        addresses = {
            info[4][0] for info in socket.getaddrinfo(host, parts.port or None)
        }
    except (OSError, UnicodeError):
        return EndpointInfo(host, "remote")
    try:
        local = bool(addresses) and all(_is_local(str(a)) for a in addresses)
    except ValueError:
        local = False
    return EndpointInfo(host, "local" if local else "remote")
