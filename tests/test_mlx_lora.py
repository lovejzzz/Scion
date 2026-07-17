from __future__ import annotations

import pytest

from scion.mlx_lora import bucket_for_length


@pytest.mark.parametrize(
    ("length", "expected"),
    [(1, 384), (384, 384), (385, 768), (1025, 1280), (1537, 1600), (1600, 1600)],
)
def test_sequence_bucket_boundaries(length: int, expected: int) -> None:
    assert bucket_for_length(length, maximum=1600) == expected


def test_sequence_bucket_requires_exact_maximum() -> None:
    with pytest.raises(ValueError, match="max_seq_length"):
        bucket_for_length(100, maximum=4096)
