"""The documented Docker setup (spec 022, contracts/docker.md).

"Same machine" under Docker rests on the port being published on the host's
loopback only; these checks keep the compose file and that guarantee together.
"""

from pathlib import Path

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"


@pytest.fixture(scope="module")
def services():
    return yaml.safe_load(COMPOSE.read_text())["services"]


@pytest.mark.parametrize("name", ["videoannotator-prod", "videoannotator-gpu"])
def test_published_on_this_machine_only(services, name):
    assert services[name]["ports"] == ["127.0.0.1:18011:18011"]


@pytest.mark.parametrize("name", ["videoannotator-prod", "videoannotator-gpu"])
def test_videos_read_only_and_results_on_the_host(services, name):
    volumes = services[name]["volumes"]
    assert "${VIDEOS_DIR:-./videos}:/videos:ro" in volumes
    assert "${RESULTS_DIR:-~/VideoAnnotator}:/results" in volumes
    assert "./data:/app/data:ro" in volumes  # kept for existing setups


@pytest.mark.parametrize("name", ["videoannotator-prod", "videoannotator-gpu"])
def test_server_is_told_where_and_that_callers_are_local(services, name):
    env = dict(e.split("=", 1) for e in services[name]["environment"] if "=" in e)
    # /videos is shared only when VIDEOS_DIR is set (spec 024, R8).
    assert env["VIDEOANNOTATOR_INGEST_ROOTS"] == "${VIDEOS_DIR:+/videos}"
    assert env["VIDEOANNOTATOR_RESULTS_DIR"] == "/results"
    assert env["VIDEOANNOTATOR_PUBLISHED_LOCALLY"].split()[0] == "1"
    assert env["VIDEOANNOTATOR_HOST_PATHS"] == (
        "/videos=${VIDEOS_DIR:-./videos};/results=${RESULTS_DIR:-~/VideoAnnotator}"
    )


@pytest.mark.parametrize(
    "name", ["videoannotator-dev", "videoannotator-prod", "videoannotator-gpu"]
)
def test_download_cache_is_shared_with_the_launcher(services, name):
    assert "cache:/app/cache" in services[name]["volumes"]


def test_volumes_have_the_names_the_launcher_uses():
    volumes = yaml.safe_load(COMPOSE.read_text())["volumes"]
    names = {v["name"] for v in volumes.values() if v and "name" in v}
    assert {
        "videoannotator-models",
        "videoannotator-database",
        "videoannotator-storage",
        "videoannotator-cache",
    } <= names
