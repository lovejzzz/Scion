#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scion.local_evaluation import rescore_coursemapper_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=Path("data/seeds/heldout.jsonl"))
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            rescore_coursemapper_runtime(
                fixture_path=args.fixtures,
                source_report_path=args.source_report,
                source_results_path=args.source_results,
                output_dir=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
