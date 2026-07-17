#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scion.token_audit import audit_orpo_lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/orpo"))
    parser.add_argument("--output", type=Path, default=Path("runs/audits/token-lengths.json"))
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    args = parser.parse_args()
    result = audit_orpo_lengths(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_path=args.output,
        max_sequence_length=args.max_sequence_length,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
