from __future__ import annotations

from pathlib import Path

import pytest

from scion.packaging import file_record


def test_file_record_refuses_package_escape(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError):
        file_record(outside, package)
