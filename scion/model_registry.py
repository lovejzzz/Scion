"""Machine-readable registry for all pinned Scion models."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from . import constants as c


@dataclass(frozen=True)
class ModelPin:
    role: str
    model_id: str
    revision: str
    format: str
    license: str
    filename: str | None = None
    bytes: int | None = None
    sha256: str | None = None
    required: bool = True


MODEL_PINS = {
    "teacher": ModelPin(
        "primary-teacher",
        c.PRIMARY_TEACHER_ID,
        c.PRIMARY_TEACHER_REVISION,
        "mlx-8bit-safetensors",
        c.PRIMARY_TEACHER_LICENSE,
    ),
    "critic": ModelPin(
        "independent-critic",
        c.CRITIC_ID,
        c.CRITIC_REVISION,
        "mlx-affine-4bit-safetensors",
        c.CRITIC_LICENSE,
        bytes=c.CRITIC_BYTES,
    ),
    "optional-teacher": ModelPin(
        "optional-escalation-teacher",
        c.OPTIONAL_TEACHER_ID,
        c.OPTIONAL_TEACHER_REVISION,
        "mlx-optiq-2bit-safetensors",
        c.OPTIONAL_TEACHER_LICENSE,
        required=False,
    ),
    "lite-train": ModelPin(
        "student-lite-training-base",
        c.LITE_TRAIN_BASE_ID,
        c.LITE_TRAIN_BASE_REVISION,
        "gemma4-qat-unquantized-safetensors",
        c.BASE_LICENSE,
        "model.safetensors",
        sha256=c.LITE_TRAIN_WEIGHT_SHA256,
    ),
    "lite-runtime": ModelPin(
        "student-lite-runtime-base",
        c.LITE_RUNTIME_BASE_ID,
        c.LITE_RUNTIME_BASE_REVISION,
        "gguf-q4_0",
        c.BASE_LICENSE,
        c.LITE_RUNTIME_BASE_FILE,
        c.LITE_RUNTIME_BASE_BYTES,
        c.LITE_RUNTIME_BASE_SHA256,
    ),
    "pro-train": ModelPin(
        "student-pro-training-base",
        c.PRO_TRAIN_BASE_ID,
        c.PRO_TRAIN_BASE_REVISION,
        "gemma4-qat-unquantized-safetensors",
        c.BASE_LICENSE,
        "model.safetensors",
        sha256=c.PRO_TRAIN_WEIGHT_SHA256,
    ),
    "pro-runtime": ModelPin(
        "student-pro-runtime-base",
        c.PRO_RUNTIME_BASE_ID,
        c.PRO_RUNTIME_BASE_REVISION,
        "gguf-q4_0",
        c.BASE_LICENSE,
        c.PRO_RUNTIME_BASE_FILE,
        c.PRO_RUNTIME_BASE_BYTES,
        c.PRO_RUNTIME_BASE_SHA256,
    ),
}


def registry_json() -> dict[str, dict[str, object]]:
    return {name: asdict(pin) for name, pin in MODEL_PINS.items()}


def student_pin(tier: str, *, runtime: bool = False) -> ModelPin:
    normalized = tier.lower()
    if normalized not in {"lite", "pro"}:
        raise ValueError("tier must be 'lite' or 'pro'")
    return MODEL_PINS[f"{normalized}-{'runtime' if runtime else 'train'}"]
