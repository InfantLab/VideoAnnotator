"""The VLM provenance block says where the model ran (spec 027 R8)."""

from videoannotator.pipelines.vlm_annotation.vlm_pipeline import VLMAnnotationPipeline


def test_local_endpoint_recorded():
    vlm = VLMAnnotationPipeline({"base_url": "http://127.0.0.1:11434"}).provenance_vlm()
    assert vlm["endpoint_class"] == "local"
    assert vlm["endpoint_host"] == "127.0.0.1"


def test_remote_endpoint_recorded(monkeypatch):
    import socket

    from videoannotator import endpoints

    monkeypatch.setattr(
        endpoints.socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ],
    )
    vlm = VLMAnnotationPipeline(
        {"base_url": "https://api.example.com"}
    ).provenance_vlm()
    assert (vlm["endpoint_class"], vlm["endpoint_host"]) == (
        "remote",
        "api.example.com",
    )
