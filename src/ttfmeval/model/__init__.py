"""TTFM model and inference utilities (vendored for standalone eval)."""

from .ttfm import TTFMLF, build_model
from .util import (
    encode_texts,
    get_text_embedding_size,
    set_text_embedder,
)

__all__ = [
    "TTFMLF",
    "build_model",
    "encode_texts",
    "get_text_embedding_size",
    "set_text_embedder",
]
