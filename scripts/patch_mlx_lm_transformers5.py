#!/usr/bin/env python3
"""Apply the audited MLX-LM tokenizer registration shim for Transformers 5.

MLX-LM 0.31.3 registers ``NewlineTokenizer`` with the Transformers 4 API.
Transformers 5.13.0 records the class and then raises while treating the old
string argument as a config class.  Replacing the call with a direct registry
assignment is equivalent to the successful first half of that registration.

The patch is intentionally exact and idempotent.  It refuses to modify an
unknown upstream source instead of silently patching a future release.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

EXPECTED_MLX_LM_VERSION = "0.31.3"
EXPECTED_TRANSFORMERS_VERSION = "5.13.0"
OLD = 'AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=NewlineTokenizer)'
NEW = """# Scion compatibility shim: Transformers 5 no longer accepts a tokenizer
# class name as the first AutoTokenizer.register argument.  This is the only
# registry used by tokenizer_class_from_name for custom tokenizer names.
from transformers.models.auto.tokenization_auto import REGISTERED_TOKENIZER_CLASSES

REGISTERED_TOKENIZER_CLASSES[\"NewlineTokenizer\"] = NewlineTokenizer"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    versions = {
        "mlx-lm": importlib.metadata.version("mlx-lm"),
        "transformers": importlib.metadata.version("transformers"),
    }
    expected = {
        "mlx-lm": EXPECTED_MLX_LM_VERSION,
        "transformers": EXPECTED_TRANSFORMERS_VERSION,
    }
    if versions != expected:
        raise SystemExit(f"REFUSING: expected {expected}, found {versions}")

    spec = importlib.util.find_spec("mlx_lm")
    if spec is None or spec.origin is None:
        raise SystemExit("REFUSING: mlx_lm is not importable")
    target = Path(spec.origin).parent / "tokenizer_utils.py"
    source = target.read_text(encoding="utf-8")
    if NEW in source and OLD not in source:
        print(f"already patched: {target} sha256={sha256(target)}")
        return
    if source.count(OLD) != 1 or NEW in source:
        raise SystemExit(f"REFUSING: unexpected upstream source: {target}")
    target.write_text(source.replace(OLD, NEW), encoding="utf-8")
    print(f"patched: {target} sha256={sha256(target)}")


if __name__ == "__main__":
    main()
