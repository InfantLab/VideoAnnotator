"""
VideoAnnotator Storage Backend System

This module provides storage abstraction for batch processing,
enabling seamless migration from files to SQLite to PostgreSQL.
"""

from .base import StorageBackend
from .file_backend import FileStorageBackend

__all__ = [
    "FileStorageBackend",
    "StorageBackend",
]
