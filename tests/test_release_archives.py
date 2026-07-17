from __future__ import annotations

import json
import tarfile
from pathlib import Path

from scion.packaging import sha256_file
from scion.release_archives import build_release_archive


def test_release_archive_is_deterministic_and_manifest_bounded(tmp_path: Path) -> None:
    package = tmp_path / "scion-test"
    package.mkdir()
    adapter = package / "adapter.bin"
    adapter.write_bytes(b"adapter")
    manifest = {
        "files": [
            {
                "path": adapter.name,
                "bytes": adapter.stat().st_size,
                "sha256": sha256_file(adapter),
            }
        ]
    }
    (package / "scion-adapter.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    build_release_archive(package_dir=package, output_path=first)
    build_release_archive(package_dir=package, output_path=second)

    assert sha256_file(first) == sha256_file(second)
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == [
            "scion-test/scion-adapter.json",
            "scion-test/adapter.bin",
        ]
