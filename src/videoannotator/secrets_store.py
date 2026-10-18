"""A Hugging Face token saved on this server (spec 031).

Kept next to the database, so it lives on the same persistent volume in a
container and never leaves the machine. Hugging Face's own token file isn't
used: in the container it isn't on a persistent volume. Readable only by the
server's user, like `huggingface-cli login`'s file.
"""

from __future__ import annotations

import logging
import os
import re
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


# Terminals wrap pasted text in these (bracketed paste); a hidden prompt keeps them.
_PASTE_MARKERS = re.compile(r"\x1b\[20[01]~")
_TOKEN_SHAPE = re.compile(r"hf_[A-Za-z0-9]{20,}")


def clean_huggingface_token(raw: str) -> str:
    """`raw` without paste markers, control characters or whitespace, none of
    which a token contains but all of which a paste can add."""
    text = _PASTE_MARKERS.sub("", raw)
    return "".join(c for c in text if c.isprintable() and not c.isspace())


def token_shape_problem(token: str) -> str | None:
    """Why `token` can't be a Hugging Face access token, or None. Says what
    was received without repeating it."""
    if not token:
        return "nothing was received"
    if _TOKEN_SHAPE.fullmatch(token):
        return None
    start = "hf_" if token.startswith("hf_") else f"{token[:3]!r}"
    return (
        f"got {len(token)} characters starting {start}; a token is hf_ followed "
        "by letters and digits"
    )


def mask_huggingface_token(token: str) -> str:
    """Enough to recognise a token, not enough to use it: hf_abc…wxyz."""
    return f"{token[:6]}…{token[-4:]}" if len(token) > 12 else "…"


def save_huggingface_token(token: str) -> Path:
    """Write the token atomically, readable only by this user."""
    token = clean_huggingface_token(token)
    problem = token_shape_problem(token)
    if problem:
        raise ValueError(f"not a Hugging Face token: {problem}")
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
