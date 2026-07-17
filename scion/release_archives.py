"""Build deterministic archives from checksum-covered Scion package files."""

from __future__ import annotations

import gzip
import json
import tarfile
from pathlib import Path
from typing import Any

from .packaging import sha256_file


def _manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _covered_paths(package_dir: Path) -> list[Path]:
    manifest_path = package_dir / "scion-adapter.json"
    manifest = _manifest(manifest_path)
    paths = [manifest_path]
    seen = {manifest_path.name}
    for record in manifest.get("files", []):
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen:
            raise RuntimeError(f"unsafe or duplicate package path: {relative}")
        path = package_dir / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"package file is not regular: {path}")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"package file does not match manifest: {path}")
        seen.add(relative.as_posix())
        paths.append(path)
    return paths


def build_release_archive(*, package_dir: Path, output_path: Path) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    output_path = output_path.resolve()
    paths = _covered_paths(package_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        output_path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for path in paths:
            relative = path.relative_to(package_dir)
            info = archive.gettarinfo(str(path), arcname=f"{package_dir.name}/{relative.as_posix()}")
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return {
        "name": output_path.name,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "package": package_dir.name,
        "files": len(paths),
        "format": "deterministic-tar-gzip-v1",
    }
