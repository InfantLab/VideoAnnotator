"""A Hugging Face token saved on this server (spec 031).

Kept next to the database, so it lives on the same persistent volume in a
container and never leaves the machine. Hugging Face's own token file isn't
used: in the container it isn't on a persistent volume. Readable only by the
server's user, like `huggingface-cli login`'s file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from videoannotator.database_location import database_path

LOGGER = logging.getLogger("videoannotator.secrets")

TOKEN_FILENAME = "huggingface_token"


def huggingface_token_path() -> Path:
    return database_path().parent / TOKEN_FILENAME


def read_saved_huggingface_token() -> str | None:
    path = huggingface_token_path()
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as e:
        LOGGER.warning("Couldn't read the saved Hugging Face token (%s): %s", path, e)
        return None
    return value or None


def save_huggingface_token(token: str) -> Path:
    """Write the token atomically, readable only by this user."""
    token = token.strip()
    if not token:
        raise ValueError("empty token")
    path = huggingface_token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


def remove_saved_huggingface_token() -> bool:
    try:
        huggingface_token_path().unlink()
    except FileNotFoundError:
        return False
    return True
