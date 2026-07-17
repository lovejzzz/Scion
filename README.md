# Scion Bonsai 27B

Scion is the CourseMapper education adapter for PrismML Bonsai 27B. It trains a small LoRA on
CourseMapper's lesson, assessment, key-term, and source-grounding contracts, converts it to GGUF,
and serves it through the OpenAI-compatible endpoint CourseMapper already expects:
`http://127.0.0.1:8799`, model `scion-1`.

## Size contract

The separately delivered **Scion adapter must be below 1,000,000,000 bytes**. PrismML's pinned
`Bonsai-27B-Q1_0.gguf` base is 3,803,452,480 bytes and is downloaded independently. A complete
27B model below 1 GB is not technically possible with this checkpoint; the sub-1 GB promise is
therefore enforced on the Scion-specific artifact and its receipts, not on the shared base.

## Verified release

The promoted Scion v2.0.0 adapter is 29,897,824 bytes; the adapter, manifest, and all portable
receipts total 30,179,319 bytes. Adapter SHA-256:
`8c12c7fa2fb32c88117ca5d479d9d1bd9b93a3bb38a1d054b14344aa83282a83`.

| Frozen 48-case test-only evaluation | Bonsai base | Scion adapter |
|---|---:|---:|
| Contract pass | 45/48 (93.75%) | 45/48 (93.75%) |
| Mean structural/pedagogical quality | 0.9653 | 0.9705 |
| Mean reference-token F1 | 0.3449 | 0.3506 |
| p95 latency on M2 Max | 54.02 s | 54.25 s |

The adapter passes every promotion gate and the CourseMapper browser/SSE smoke test. These are
automated integration and content-retention measures, not evidence of student learning outcomes or
universal factual correctness. Exact row-level outputs are published in `artifacts/receipts`.

## What is pinned

| Component | Immutable identity |
|---|---|
| Training checkpoint | `prism-ml/Bonsai-27B-unpacked@d619b27283ac02b4199ced97a89419529dc0bfac` |
| Serving checkpoint | `prism-ml/Bonsai-27B-gguf@0cf7e3d21581b169b4df1de8bf01316000e2fbb7` |
| Serving file | `Bonsai-27B-Q1_0.gguf`, SHA-256 `17ef842e…f819aa0` |
| MLX LM | `0.31.2`, source revision `dcbf6e33…814a7` |
| PrismML llama.cpp | revision `38c66ad…cec5` |

Mutable model aliases are never accepted in a training or release receipt.

## Quick start

Training is designed for an Apple Silicon Mac with 64 GB unified memory and roughly 80 GB of free
disk space. Serving needs substantially less memory. Install Python 3.11 or 3.12, `uv`, CMake, and
the Xcode command-line tools.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[train,convert,dev]'

scion prepare
scion train
scion runtime build
scion convert
```

Preparation verifies the exact unpacked Bonsai revision before producing a local MLX 4-bit QLoRA
base. Training writes an MLX adapter plus immutable run receipts. It pads examples into six
bounded sequence buckets (maximum 1,600 tokens, above the longest checked-in example) so long
runs do not accumulate a separate Metal graph for every observed length. Conversion creates
`artifacts/scion-bonsai-27b.gguf` and rejects it if it reaches 1 GB.

Serve the adapter:

```bash
scion serve \
  --adapter artifacts/scion-bonsai-27b.gguf \
  --llama-server .cache/PrismML-llama.cpp/build/bin/llama-server
```

Then open CourseMapper, choose the local Scion provider, and connect. No API key or CourseMapper
source change is required. See [CourseMapper integration](docs/COURSEMAPPER.md).

## Dataset

The checked-in corpus contains 711 training, 297 validation, and 457 test examples. Course groups,
not individual rows, define the splits. The 48 promotion fixtures are sourced only from test groups
and contain 12 lessons, 12 MC items, 12 key terms, and 12 source bundles.
Every response is admitted by the same structural education contracts used in evaluation. See
[the dataset card](DATASET_CARD.md).

To rebuild it from immutable sibling checkouts:

```bash
git show 7f97e9b7f995bb7bf74eedd0c07fa8ca291f1d06:data/preference-pairs-full.jsonl \
  > /tmp/scion-preferences.jsonl

scion dataset \
  --legacy-jsonl /tmp/scion-preferences.jsonl \
  --approved-jsonl ../CourseMapper/evaluation/scion-adapters/evidence/codex-approved-preferences-v0.16.42.jsonl \
  --source-capture-dir ../CourseMapper/evaluation/scion-source-capture-evidence \
  --source-capture-dir ../CourseMapper/evaluation/scion-source-capture-expansion-evidence
```

## Promotion gates

A trained checkpoint is explicitly `unpromoted` until all of these pass:

1. The GGUF adapter is valid and below 1 GB.
2. Base and adapter run the same SHA-bound, balanced set of 48 test-only fixtures.
3. Adapter schema compliance is at least 90% and does not regress.
4. Structural quality and reference-content F1 do not regress from the base.
5. The real CourseMapper `/v1/models` and `/v1/chat/completions` contract passes.

Run `scion evaluate` once against the base and once against the adapter, `scion compare`, then
`scion smoke`. Only `scion release` can write a promoted manifest. This prevents an unevaluated
checkpoint from being described as production-ready.

## Development

```bash
ruff check .
pytest
```

Scion code is Apache-2.0 licensed. The dataset retains source-specific terms. Bonsai and its runtime
remain separate third-party components; see
[third-party notices](THIRD_PARTY_NOTICES.md) and [the model card](MODEL_CARD.md).
