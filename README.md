# Scion Education

Scion is a local, reproducible education specialization for
[CourseMapper](https://github.com/lovejzzz/CourseMapper). It produces two parameter-efficient Gemma 4 adapters:

| Tier | Exact student base | Intended runtime |
|---|---|---|
| Scion Lite | Gemma 4 E2B QAT | CourseMapper browser runtime via a separate GGUF LoRA |
| Scion Pro | Gemma 4 12B QAT | Apple Silicon through MLX-VLM |

The adapter is the Scion-specific artifact; the immutable base model is downloaded and cached separately. Every
adapter must remain below 1,000,000,000 bytes. Lite also obeys CourseMapper's tighter 64 MiB browser-package and
two-percent-of-base limits.

## Qualified research artifacts

| Artifact | Adapter bytes | Locked result | Status |
|---|---:|---|---|
| Lite MLX | 52,808,169 | 26/32 vs base 2/32, strict raw JSON | research |
| Lite browser GGUF | 26,370,912 (26,383,787-byte package) | 21/32 vs base 20/32, exact GGUF + JSON schema | research |
| Pro MLX | 332,730,430 | 30/32 vs base 28/32, CourseMapper transport normalization | research |

All three are below the one-gigabyte requirement. The browser package is 0.788% of its pinned 3.35 GB base and
uses the validation-selected inference scale `10`. These numbers are paired results on the same locked 32 tasks,
not general education benchmarks or evidence of improved student outcomes.

## What Scion learns

Scion is trained for grounded course planning and educational behavior rather than memorizing a university
catalog. Its task contracts cover prerequisite graphs, schedule constraints, degree audits, source-bounded
uncertainty, supportive tutoring, safe academic boundaries, tool calls, and CourseMapper lesson kernels. Live
course facts must continue to come from CourseMapper retrieval or tools.

The local distillation chain is:

```text
license-clean synthetic tasks
  -> Qwen3.6 27B 8-bit teacher
  -> deterministic task oracle
  -> blind Gemma 4 31B Q4 critic
  -> Gemma 4 Lite and Pro ORPO LoRA students
  -> locked base-versus-adapter evaluation
  -> MLX and browser GGUF release packages
```

No closed-model output, private student record, or real institutional catalog is used. The optional Qwen3.5
122B 2-bit escalation teacher is pinned but disabled unless the primary teacher and independent critic fail a
measured quality gate.

## Immutable model registry

Run `scion models` for the machine-readable registry. The important revisions are:

| Role | Model | Revision |
|---|---|---|
| Teacher | `mlx-community/Qwen3.6-27B-8bit` | `c5a593c1475a746e43a543b0a02bd2b357e5745f` |
| Critic | `mlx-community/gemma-4-31b-it-4bit` | `696d436c404745a59f30e4939a658162b0a9e57f` |
| Lite training base | `google/gemma-4-E2B-it-qat-q4_0-unquantized` | `1ca4dd94b623b6e0dd9da00c2239ab84b4f3e5ce` |
| Pro training base | `google/gemma-4-12B-it-qat-q4_0-unquantized` | `b8dea52d5ea56a20e8872f0ee5d25ada7501327e` |

All required models report Apache-2.0 licensing. Mutable aliases are never accepted as run identities.

## Reproduce the pipeline

The supported training host is an Apple Silicon Mac. This project was developed on an M2 Max with 64 GB unified
memory.

```bash
./scripts/bootstrap_macos.sh
source .venv-gemma/bin/activate

python scripts/build_seed_tasks.py
python scripts/generate_teacher_corpus.py --split train
python scripts/generate_teacher_corpus.py --split validation
python scripts/generate_teacher_corpus.py --split preference-test

python scripts/filter_with_critic.py --split train
python scripts/filter_with_critic.py --split validation
python scripts/filter_with_critic.py --split preference-test
python scripts/build_dataset_manifest.py
```

Teacher generation resumes by exact task hash. If a prompt changes, stale output is removed and only affected
tasks are regenerated. The critic randomizes candidate labels, records its blind judgment, and admits a pair only
when the deterministic oracle and independent critic agree.

Before training, commit the complete corpus and code. Training refuses a dirty repository, audits every formatted
sequence against the 2,048-token limit, and binds the clean Git commit, dataset identity, toolchain, seed, and
hyperparameters into its receipt.

```bash
# Ten-step hardware and loss canaries first
scion train --tier lite --smoke --local-files-only --output artifacts/scion-lite-smoke
scion train --tier pro  --smoke --local-files-only --output artifacts/scion-pro-smoke

# Qualified research schedules
scion train --tier lite --iterations 100 --local-files-only --output artifacts/scion-lite-mlx
scion train --tier pro --smoke --local-files-only --output artifacts/smoke-pro-mlx
```

The exact selected runs are bound to source commit `553d53571ed3dd339643bae17b2b6b300a216e08` in their training
receipts. Lite uses 100 updates. Pro deliberately uses the qualified 10-update canary: longer and retuned Pro
runs were evaluated and rejected for worse held-out stability. Use that historical commit when reproducing the
released artifacts exactly; later source keeps the conservative experimental Pro schedule rather than silently
claiming a replacement artifact.

Package the trained adapters:

```bash
python scripts/package_mlx_adapter.py \
  --tier lite \
  --adapter-dir artifacts/scion-lite-mlx

python scripts/package_browser_adapter.py \
  --source-manifest artifacts/scion-lite-mlx/scion-adapter.json \
  --output-dir artifacts/scion-lite-browser \
  --inference-scale 10 \
  --base-dir .cache/huggingface/models--google--gemma-4-E2B-it-qat-q4_0-unquantized/snapshots/1ca4dd94b623b6e0dd9da00c2239ab84b4f3e5ce
```

The browser conversion uses llama.cpp revision `5ec717d1256e34558a44dc09adf1e6e16f2e2682`, audits all GGUF metadata and
LoRA tensor pairs, and binds the source MLX manifest through the conversion chain.

## Evaluation and release boundary

The 32 locked evaluation tasks are never passed to either teacher or critic. Run the exact same fixtures against
the base and adapter:

```bash
scion evaluate --tier lite --variant base
scion evaluate --tier lite --variant adapter --adapter artifacts/scion-lite-mlx
scion compare \
  --base runs/evaluation/lite/base/report.json \
  --adapter runs/evaluation/lite/adapter/report.json \
  --output runs/evaluation/lite/comparison.json
```

A passing comparison requires a real overall improvement, fewer deterministic issues, no category regression,
and no citation-hallucination regression. Three separately labeled comparisons are shipped:

| Evaluation path | Base | Adapter | Issue delta | Important boundary |
|---|---:|---:|---:|---|
| Lite MLX strict | 2/32 | 26/32 | -21 | Native unquantized training base; lesson kernels remain 0/4 |
| Pro MLX CourseMapper | 28/32 | 30/32 | -2 | Only lossless Markdown-fence removal; no malformed or semantic repair |
| Lite GGUF schema | 20/32 | 21/32 | -1 | Exact 3,349,514,112-byte runtime base and scale 10 in pinned llama.cpp |

The GGUF run verifies the converted adapter and CourseMapper prompt/schema contract through pinned llama.cpp; it
is not a WebGPU device-activation test. The native Lite result therefore must not be presented as browser quality.
Full paired reports are included under `release/scion-3.0.0-research/evidence`.

The dataset and resulting adapters are deliberately labeled **research**, not production or promoted:
CourseMapper's production policy requires at least 3,000 verified preference pairs, at least five domains, and
broader instructor and real-world evidence.

See [MODEL_CARD.md](MODEL_CARD.md), [DATASET_CARD.md](DATASET_CARD.md), and
[docs/COURSEMAPPER.md](docs/COURSEMAPPER.md) for limitations and integration details.

Release archives contain only each package's `scion-adapter.json` and its checksum-covered files. They can be
rebuilt byte-for-byte with:

```bash
python scripts/build_release_archives.py \
  --package artifacts/scion-lite-mlx \
  --package artifacts/scion-lite-browser \
  --package artifacts/smoke-pro-mlx
```

## Development

```bash
.venv-gemma/bin/ruff check scion scripts tests
.venv-gemma/bin/python -m pytest
```

Repository-authored code and synthetic data are Apache-2.0. External spend for the local build is $0.
