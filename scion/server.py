"""Launch Bonsai 27B plus a separately delivered Scion GGUF adapter."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

from .constants import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_SCION_ARTIFACT_BYTES,
    SCION_MODEL_ID,
    SERVE_BASE_BYTES,
    SERVE_BASE_FILE,
    SERVE_BASE_ID,
    SERVE_BASE_REVISION,
    SERVE_BASE_SHA256,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_base(*, cache_dir: Path | None = None, local_files_only: bool = False) -> Path:
    path = Path(
        hf_hub_download(
            repo_id=SERVE_BASE_ID,
            filename=SERVE_BASE_FILE,
            revision=SERVE_BASE_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    ).resolve()
    if path.stat().st_size != SERVE_BASE_BYTES or _sha256(path) != SERVE_BASE_SHA256:
        raise RuntimeError("downloaded Bonsai GGUF failed the pinned size/SHA-256 identity check")
    return path


def resolve_binary(explicit: Path | None = None) -> Path:
    candidates = [
        explicit,
        Path(os.environ["SCION_LLAMA_SERVER"]) if os.environ.get("SCION_LLAMA_SERVER") else None,
        Path(found) if (found := shutil.which("llama-server")) else None,
    ]
    for candidate in candidates:
        if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise RuntimeError("llama-server was not found; run `scion runtime build` or pass --llama-server")


def validate_adapter(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if not 1024 <= size < MAX_SCION_ARTIFACT_BYTES:
        raise RuntimeError(f"adapter must be a non-empty GGUF below 1 GB, got {size} bytes")
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise RuntimeError("adapter is not a GGUF file")
    return {"path": str(path.resolve()), "bytes": size, "sha256": _sha256(path)}


def server_command(
    *,
    binary: Path,
    base: Path,
    adapter: Path | None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    gpu_layers: int = 999,
) -> list[str]:
    command = [
        str(binary),
        "-m",
        str(base),
        "--host",
        host,
        "--port",
        str(port),
        "-ngl",
        str(gpu_layers),
        "-fa",
        "on",
        "-c",
        str(context_size),
        "--parallel",
        "1",
        "--jinja",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--reasoning-format",
        "deepseek",
        "--alias",
        SCION_MODEL_ID,
    ]
    if adapter is not None:
        command.extend(["--lora", str(adapter)])
    return command


def serve(
    *,
    adapter: Path | None,
    llama_server: Path | None = None,
    base: Path | None = None,
    cache_dir: Path | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    dry_run: bool = False,
) -> dict[str, object]:
    binary = resolve_binary(llama_server)
    model = base.resolve() if base else resolve_base(cache_dir=cache_dir)
    adapter_record = validate_adapter(adapter.resolve()) if adapter else None
    command = server_command(
        binary=binary,
        base=model,
        adapter=adapter.resolve() if adapter else None,
        host=host,
        port=port,
        context_size=context_size,
    )
    result: dict[str, object] = {
        "binary": str(binary),
        "base": str(model),
        "baseRevision": SERVE_BASE_REVISION,
        "adapter": adapter_record,
        "endpoint": f"http://{host}:{port}",
        "command": command,
    }
    if dry_run:
        result["status"] = "dry-run"
        return result
    subprocess.run(command, check=True)
    result["status"] = "stopped"
    return result
