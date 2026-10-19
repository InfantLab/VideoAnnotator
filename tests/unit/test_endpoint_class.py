"""Local or remote, by the addresses a model server's host resolves to."""

import socket

import pytest

from videoannotator import endpoints


def _resolves(monkeypatch, table):
    def fake(host, port, *args, **kwargs):
        if host not in table:
            raise socket.gaierror("not found")
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0)) for a in table[host]
        ]

    monkeypatch.setattr(endpoints.socket, "getaddrinfo", fake)


@pytest.mark.parametrize(
    ("url", "addresses", "expected"),
    [
        ("http://localhost:11434", ["127.0.0.1", "::1"], "local"),
        ("http://lab-gpu:11434", ["192.168.1.20"], "local"),
        ("http://host.docker.internal:11434", ["192.168.65.254"], "local"),
        ("https://api.example.com", ["93.184.216.34"], "remote"),
        ("http://mixed:11434", ["10.0.0.5", "93.184.216.34"], "remote"),
        ("http://tailnet:11434", ["100.64.1.1"], "remote"),
        ("http://v6ula:11434", ["fd12:3456::1"], "local"),
    ],
)
def test_classify(monkeypatch, url, addresses, expected):
    host = url.split("://")[1].split(":")[0].split("/")[0]
    _resolves(monkeypatch, {host: addresses})
    info = endpoints.classify_endpoint(url)
    assert info.cls == expected
    assert info.host == host


def test_unresolvable_is_remote(monkeypatch):
    _resolves(monkeypatch, {})
    assert endpoints.classify_endpoint("http://nowhere.invalid:11434").cls == "remote"


def test_ip_literal(monkeypatch):
    _resolves(monkeypatch, {"10.1.2.3": ["10.1.2.3"], "8.8.8.8": ["8.8.8.8"]})
    assert endpoints.classify_endpoint("http://10.1.2.3:11434").cls == "local"
    assert endpoints.classify_endpoint("http://8.8.8.8").cls == "remote"


@pytest.mark.parametrize(
    ("url", "normalised"),
    [
        ("http://LocalHost:11434/", "http://localhost:11434"),
        ("HTTPS://Api.Example.com/v1/", "https://api.example.com/v1"),
        ("127.0.0.1:11434", "http://127.0.0.1:11434"),
    ],
)
def test_normalise_base_url(url, normalised):
    assert endpoints.normalise_base_url(url) == normalised
