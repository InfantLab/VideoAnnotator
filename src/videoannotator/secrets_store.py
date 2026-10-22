"""A Hugging Face token (spec 031) and model server keys (spec 033) saved on
this server.

Kept next to the database, so it lives on the same persistent volume in a
container and never leaves the machine. Hugging Face's own token file isn't
used: in the container it isn't on a persistent volume. Readable only by the
server's user, like `huggingface-cli login`'s file.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from videoannotator.database_location import database_path
from videoannotator.endpoints import normalise_base_url

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


# ---------------------------------------------------------------------------
# Model server keys (spec 033 R4)
#
# One file beside the database, {normalised base URL: key}, like the token
# above: 0600, written atomically, read per call so a key changed in Settings
# applies to the next call. Never in the database, so a copy of the database
# file shared for support doesn't carry them.
# ---------------------------------------------------------------------------

MODEL_SERVER_KEYS_FILENAME = "model_server_keys.json"


def model_server_keys_path() -> Path:
    return database_path().parent / MODEL_SERVER_KEYS_FILENAME


def _read_model_server_keys() -> dict[str, str]:
    path = model_server_keys_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        LOGGER.warning("Couldn't read the model server keys (%s): %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str) and v}


def _write_model_server_keys(keys: dict[str, str]) -> Path:
    path = model_server_keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=1, sort_keys=True)
        f.write("\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


def read_model_server_key(base_url: str) -> str | None:
    return _read_model_server_keys().get(normalise_base_url(base_url))


def save_model_server_key(base_url: str, key: str) -> Path:
    key = clean_huggingface_token(key)
    if not key:
        raise ValueError("the key is empty")
    keys = _read_model_server_keys()
    keys[normalise_base_url(base_url)] = key
    return _write_model_server_keys(keys)


def remove_model_server_key(base_url: str) -> bool:
    keys = _read_model_server_keys()
    if keys.pop(normalise_base_url(base_url), None) is None:
        return False
    _write_model_server_keys(keys)
    return True


def known_model_server_keys() -> list[str]:
    return list(_read_model_server_keys().values())


# Names of environment variables servers take their key from; kept up to date
# by model_servers.py so the filter below needn't ask the database.
KEY_ENV_NAMES: set[str] = set()

# Shorter than this, a "key" would redact ordinary words.
_MIN_REDACTED_LENGTH = 8


def _known_keys() -> list[str]:
    keys = known_model_server_keys()
    keys += [os.environ.get(name, "") for name in KEY_ENV_NAMES]
    return sorted(
        {k for k in keys if len(k) >= _MIN_REDACTED_LENGTH}, key=len, reverse=True
    )


class KeyRedactingFilter(logging.Filter):
    """Replaces any known model server key in a log record with <redacted>.

    Belt and braces (FR-016): no code here logs a key, but a library or a
    server's echoed error message might.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        keys = _known_keys()
        if not keys:
            return True
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = message
        for key in keys:
            redacted = redacted.replace(key, "<redacted>")
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


_REDACTION = KeyRedactingFilter()
_REDACTED_LOGGERS = (
    "",
    "videoannotator",
    "httpx",
    "httpcore",
    "uvicorn",
    "uvicorn.error",
)


def install_key_redaction() -> None:
    """On the loggers that could carry a key and on their handlers: a
    logger's own filters don't see records from its children, a handler's do."""
    for name in _REDACTED_LOGGERS:
        logger = logging.getLogger(name)
        if _REDACTION not in logger.filters:
            logger.addFilter(_REDACTION)
        for handler in logger.handlers:
            if _REDACTION not in handler.filters:
                handler.addFilter(_REDACTION)
