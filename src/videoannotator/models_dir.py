"""One directory for every pipeline's model weights (spec 016).

`VIDEOANNOTATOR_MODELS_DIR`, defaulting to the platform's per-user data directory.
Imported by the package's `__init__` before any model library, so the libraries'
own cache variables can be pointed into it. Standard library only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODELS_DIR_ENV = "VIDEOANNOTATOR_MODELS_DIR"

# Subdirectory per weights source; the dev container's layout since 2026-09-28.
SOURCES = ("huggingface", "pyannote", "torch", "deepface", "whisper", "yolo")

# Library variable -> location under the models directory. HF_HOME is deliberately
# absent: `huggingface-cli login` keeps the token under it, and moving it would
# silently log users out. HF_HUB_CACHE moves only the downloaded models.
_LIBRARY_VARIABLES = {
    "HF_HUB_CACHE": ("huggingface", "hub"),
    "TORCH_HOME": ("torch",),
    "PYANNOTE_CACHE": ("pyannote",),
    "DEEPFACE_HOME": ("deepface",),
}


def _default_root() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "videoannotator" / "models"


def models_dir() -> Path:
    """The models directory, as an absolute path."""
    configured = os.environ.get(MODELS_DIR_ENV)
    root = Path(configured).expanduser() if configured else _default_root()
    return root.resolve()


def source_dir(source: str) -> Path:
    """Where weights from one source (e.g. 'whisper', 'yolo') live."""
    return models_dir() / source


def configure_model_caches() -> None:
    """Point the model libraries' cache variables into the models directory.

    Only where the user hasn't set them; never HF_HOME (see above). Also pins
    VIDEOANNOTATOR_MODELS_DIR to its resolved value, so a later change of working
    directory can't move it.
    """
    root = models_dir()
    os.environ[MODELS_DIR_ENV] = str(root)
    for variable, parts in _LIBRARY_VARIABLES.items():
        # A user-set HF_HOME already says where Hugging Face's cache goes.
        if variable == "HF_HUB_CACHE" and "HF_HOME" in os.environ:
            continue
        os.environ.setdefault(variable, str(root.joinpath(*parts)))


def resolve_yolo_model(model: str) -> str:
    """Resolve a YOLO model setting to a path in the models directory.

    An existing or absolute path is used as given. The old default
    (`models/yolo/<name>`, relative to the working directory) and a bare model
    name (`yolo11n-pose.pt`) resolve to `<models dir>/yolo/<name>`, where
    Ultralytics downloads it if missing.
    """
    path = Path(model)
    if path.is_absolute() or path.exists():
        return str(path)
    parts = path.parts
    if len(parts) >= 2 and parts[0] == "models":
        target = models_dir().joinpath(*parts[1:])
    else:
        target = source_dir("yolo") / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    return str(target)


def directory_size(path: Path) -> int:
    """Total size in bytes of the files under `path` (0 if absent)."""
    total = 0
    if not path.exists():
        return 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return total


def legacy_locations() -> list[Path]:
    """Pre-v1.6.0 default locations that hold weights, outside the models dir."""
    home = Path.home()
    cache = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
    candidates = [
        cache / "huggingface" / "hub",
        cache / "torch" / "pyannote",
        cache / "torch" / "hub",
        cache / "whisper",
        home / ".deepface",
        Path("models").resolve(),
    ]
    root = models_dir()
    found = []
    for path in candidates:
        try:
            inside = path.resolve().is_relative_to(root) or root.is_relative_to(
                path.resolve()
            )
        except OSError:
            inside = False
        if not inside and path.is_dir() and any(path.iterdir()):
            found.append(path)
    return found
