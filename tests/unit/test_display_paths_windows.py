"""Paths shown as a Windows host shows them (spec 024, R3)."""

from videoannotator.config_env import host_paths
from videoannotator.results_folder import display_path

STUDIES = "/c/Users/ada/Studies=C:\\Users\\ada\\Studies"


def test_a_windows_host_path_uses_backslashes(monkeypatch):
    monkeypatch.setenv("VIDEOANNOTATOR_HOST_PATHS", STUDIES)
    assert display_path("/c/Users/ada/Studies/Day 1/a.mp4") == (
        "C:\\Users\\ada\\Studies\\Day 1\\a.mp4"
    )
    assert display_path("/c/Users/ada/Studies") == "C:\\Users\\ada\\Studies"


def test_host_paths_keep_backslashes_and_drop_a_trailing_one(monkeypatch):
    monkeypatch.setenv(
        "VIDEOANNOTATOR_HOST_PATHS", "/c/Users/ada/Studies=C:\\Users\\ada\\Studies\\"
    )
    assert host_paths() == [("/c/Users/ada/Studies", "C:\\Users\\ada\\Studies")]


def test_a_drive_root(monkeypatch):
    monkeypatch.setenv("VIDEOANNOTATOR_HOST_PATHS", "/e=E:\\")
    assert display_path("/e") == "E:\\"
    assert display_path("/e/Data/a.mp4") == "E:\\Data\\a.mp4"


def test_linux_pairs_unchanged(monkeypatch):
    monkeypatch.setenv(
        "VIDEOANNOTATOR_HOST_PATHS", "/videos=/home/ada/Studies/;/results=/home/ada/VA"
    )
    assert display_path("/videos/Day 1/a.mp4") == "/home/ada/Studies/Day 1/a.mp4"
    assert display_path("/results") == "/home/ada/VA"
