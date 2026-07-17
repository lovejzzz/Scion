"""Build the pinned PrismML llama.cpp runtime used for Bonsai serving."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .constants import PRISM_LLAMA_CPP_REPOSITORY, PRISM_LLAMA_CPP_REVISION


def build_runtime(output: Path, *, jobs: int = 8, dry_run: bool = False) -> dict[str, object]:
    output = output.resolve()
    binary = output / "build" / "bin" / "llama-server"
    commands: list[list[str]] = []
    if not (output / ".git").is_dir():
        commands.append(["git", "clone", "--filter=blob:none", PRISM_LLAMA_CPP_REPOSITORY, str(output)])
    commands.extend(
        [
            ["git", "-C", str(output), "fetch", "origin", PRISM_LLAMA_CPP_REVISION, "--depth", "1"],
            ["git", "-C", str(output), "checkout", "--detach", PRISM_LLAMA_CPP_REVISION],
            [
                "cmake",
                "-S",
                str(output),
                "-B",
                str(output / "build"),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DGGML_METAL=ON",
            ],
            ["cmake", "--build", str(output / "build"), "--target", "llama-server", "-j", str(max(1, jobs))],
        ]
    )
    if dry_run:
        return {"status": "dry-run", "commands": commands, "binary": str(binary)}
    if shutil.which("git") is None or shutil.which("cmake") is None:
        raise RuntimeError("building the runtime requires git and cmake")
    for command in commands:
        subprocess.run(command, check=True)
    revision = subprocess.check_output(["git", "-C", str(output), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(output), "status", "--porcelain"], text=True).strip()
    if revision != PRISM_LLAMA_CPP_REVISION or dirty:
        raise RuntimeError("runtime checkout is not the clean pinned PrismML revision")
    if not binary.is_file():
        raise RuntimeError(f"llama-server build did not produce {binary}")
    return {"status": "ready", "revision": revision, "binary": str(binary)}
