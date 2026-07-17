#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scion.seed_tasks import write_seed_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/seeds"))
    args = parser.parse_args()
    print(json.dumps(write_seed_tasks(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
