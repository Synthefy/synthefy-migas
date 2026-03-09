"""Migas-1.5 model and inference utilities (vendored for standalone eval)."""

from .migas15 import Migas15, build_model
from .util import (
    encode_texts,
    get_text_embedding_size,
    set_text_embedder,
)

__all__ = [
    "Migas15",
    "build_model",
    "encode_texts",
    "get_text_embedding_size",
    "set_text_embedder",
]
