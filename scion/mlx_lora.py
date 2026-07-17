"""Run MLX LM LoRA with bounded sequence-shape compilation.

MLX compiles a separate graph for each input shape. Single-example chat
training otherwise creates dozens of shapes and can exhaust Metal's resource
table during a long 27B run. Padding to a small set of buckets keeps every
token while bounding the number of compiled graphs.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

SEQUENCE_BUCKETS = (384, 768, 1024, 1280, 1536, 1600)


def bucket_for_length(length: int, *, maximum: int) -> int:
    """Return the smallest configured bucket that can contain ``length``."""
    if length <= 0:
        raise ValueError("sequence length must be positive")
    if maximum != SEQUENCE_BUCKETS[-1]:
        raise ValueError(f"max_seq_length must be {SEQUENCE_BUCKETS[-1]} for Scion training")
    return next((bucket for bucket in SEQUENCE_BUCKETS if length <= bucket), maximum)


def bucketed_iterate_batches(*args: Any, **kwargs: Any) -> Iterator[tuple[Any, Any]]:
    """Wrap MLX LM's iterator and pad batches to one of six stable shapes."""
    import mlx.core as mx
    from mlx_lm.tuner.trainer import iterate_batches

    maximum = int(kwargs.get("max_seq_length", args[2] if len(args) > 2 else 0))
    for batch, lengths in iterate_batches(*args, **kwargs):
        current = int(batch.shape[1])
        target = bucket_for_length(current, maximum=maximum)
        if target > current:
            batch = mx.pad(batch, [(0, 0), (0, target - current)])
        yield batch, lengths


def main() -> None:
    """Delegate argument parsing to pinned MLX LM after installing the iterator."""
    from mlx_lm import lora
    from mlx_lm.tuner import trainer

    def train_with_buckets(*args: Any, **kwargs: Any) -> Any:
        kwargs["iterate_batches"] = bucketed_iterate_batches
        return trainer.train(*args, **kwargs)

    lora.train = train_with_buckets
    lora.main()


if __name__ == "__main__":
    main()
