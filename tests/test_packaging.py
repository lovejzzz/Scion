from __future__ import annotations

from inspect import signature
from pathlib import Path

import pytest

from scion.browser_package import build_browser_adapter
from scion.constants import LITE_BROWSER_INFERENCE_SCALE
from scion.packaging import file_record


def test_file_record_refuses_package_escape(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError):
        file_record(outside, package)


def test_browser_package_uses_qualified_inference_scale() -> None:
    assert signature(build_browser_adapter).parameters["inference_scale"].default == (
        LITE_BROWSER_INFERENCE_SCALE
    )
