"""Compact, append-only collection contract for a future runner integration."""

from .compact_collection import (
    CompactCollectionWriter,
    CompactPublicReader,
    iter_public_one_step_examples,
)

__all__ = ["CompactCollectionWriter", "CompactPublicReader", "iter_public_one_step_examples"]
