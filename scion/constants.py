"""Immutable model, runtime, and delivery identities."""

from __future__ import annotations

SCION_VERSION = "2.0.0"
SCION_MODEL_ID = "scion-1"
SCION_MODEL_NAME = "Scion Bonsai 27B"

TRAIN_BASE_ID = "prism-ml/Bonsai-27B-unpacked"
TRAIN_BASE_REVISION = "d619b27283ac02b4199ced97a89419529dc0bfac"
TRAIN_BASE_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"

SERVE_BASE_ID = "prism-ml/Bonsai-27B-gguf"
SERVE_BASE_REVISION = "0cf7e3d21581b169b4df1de8bf01316000e2fbb7"
SERVE_BASE_FILE = "Bonsai-27B-Q1_0.gguf"
SERVE_BASE_BYTES = 3_803_452_480
SERVE_BASE_SHA256 = "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0"
SERVE_BASE_ARCHITECTURE = "qwen35"

PRISM_LLAMA_CPP_REPOSITORY = "https://github.com/PrismML-Eng/llama.cpp.git"
PRISM_LLAMA_CPP_REVISION = "38c66ad0241da4f9fcce541cda8edc219086cec5"
PRISM_LORA_CONVERTER_SHA256 = "96dac0708611dfb9e245b7b5dcebac5258853166cc954e526d53a701b3771aa1"
MLX_LM_VERSION = "0.31.2"
MLX_LM_REVISION = "dcbf6e33d135a1b7c6767ca0fe7ebbd23df814a7"

# The base is independently downloaded. This cap covers only the Scion-specific
# adapter package, manifest, and receipts shipped by this repository.
MAX_SCION_ARTIFACT_BYTES = 1_000_000_000

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8799
DEFAULT_CONTEXT_SIZE = 32_768

LEGACY_SCION_SOURCE_REVISION = "7f97e9b7f995bb7bf74eedd0c07fa8ca291f1d06"
COURSEMAPPER_SOURCE_REVISION = "4f5bed3833f72494917e67c1a0c878af8c2b9a70"
