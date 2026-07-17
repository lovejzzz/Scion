"""Deterministic-enough local MLX inference with immutable-model receipts."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

from .model_registry import ModelPin


@dataclass(frozen=True)
class GenerationSettings:
    max_tokens: int = 768
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int = 16031
    repetition_penalty: float = 1.05
    repetition_context_size: int = 256


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


class MlxGenerator:
    """One loaded MLX model used for a sequence of auditable generations."""

    def __init__(self, model_path: Path, pin: ModelPin, *, adapter_path: Path | None = None):
        resolved = model_path.resolve()
        if resolved.name != pin.revision:
            raise ValueError(f"model path is not pinned revision {pin.revision}: {resolved}")
        self.pin = pin
        self.model_path = resolved
        self.adapter_path = adapter_path.resolve() if adapter_path is not None else None
        self.adapter_identity = None
        if self.adapter_path is not None:
            records = []
            for name in ("adapter_config.json", "adapters.safetensors"):
                path = self.adapter_path / name
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"adapter is missing a regular {name}: {self.adapter_path}")
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                records.append({"path": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
            self.adapter_identity = {"path": str(self.adapter_path), "files": records}
        self.model, self.processor = load(
            str(resolved),
            adapter_path=str(self.adapter_path) if self.adapter_path is not None else None,
            strict=True,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        settings: GenerationSettings,
    ) -> dict[str, Any]:
        if not messages or any(
            message.get("role") not in {"system", "user", "assistant"} for message in messages
        ):
            raise ValueError("messages must use system, user, or assistant roles")
        mx.random.seed(settings.seed)
        prompt = apply_chat_template(
            self.processor,
            self.model.config,
            messages,
            add_generation_prompt=True,
            num_images=0,
        )
        started = time.monotonic()
        result = generate(
            self.model,
            self.processor,
            prompt,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            repetition_penalty=settings.repetition_penalty,
            repetition_context_size=settings.repetition_context_size,
            verbose=False,
        )
        return {
            "text": result.text.strip(),
            "metrics": {
                "promptTokens": result.prompt_tokens,
                "generationTokens": result.generation_tokens,
                "promptTokensPerSecond": result.prompt_tps,
                "generationTokensPerSecond": result.generation_tps,
                "peakMemoryGb": result.peak_memory,
                "elapsedSeconds": round(time.monotonic() - started, 3),
                "finishReason": result.finish_reason,
            },
            "receipt": {
                "protocol": "scion-local-mlx-generation-v1",
                "model": {
                    "id": self.pin.model_id,
                    "revision": self.pin.revision,
                    "role": self.pin.role,
                },
                "adapter": self.adapter_identity,
                "messagesSha256": canonical_sha256(messages),
                "settings": asdict(settings),
            },
        }


def snapshot_path(cache_dir: Path, pin: ModelPin) -> Path:
    expected = cache_dir / f"models--{pin.model_id.replace('/', '--')}" / "snapshots" / pin.revision
    if not expected.is_dir():
        raise FileNotFoundError(f"missing pinned model snapshot: {expected}")
    return expected
