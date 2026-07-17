# CourseMapper integration

Scion Lite targets CourseMapper's existing adapter architecture:

```text
exact Gemma 4 E2B QAT base
  + verified Scion GGUF LoRA
  + scion-wllama-webgpu-jspi-v1
  + CourseMapper schemas, tools, compiler, and rollback
```

The browser package contains only the adapter delta and receipts. CourseMapper downloads and caches the pinned
3.35 GB base separately. Its schema-v3 `scion-adapter.json` declares:

- exact training base model and revision;
- `gguf-lora` format and inference scale;
- `scion-wllama-webgpu-jspi-v1` runtime compatibility;
- dataset, training-plan, training-result, source-manifest, conversion, and file hashes;
- research status with `promotable: false`;
- a total package below 64 MiB and below two percent of the browser base.

The MLX source package declares `mlx-lora-safetensors` and runtime `mlx-vlm`. Scion Pro is an MLX package only; it
does not target the current E2B browser base.

## Build and validate

After the Lite adapter passes the locked comparison:

```bash
python scripts/package_mlx_adapter.py \
  --tier lite \
  --adapter-dir artifacts/scion-lite-mlx

python scripts/package_browser_adapter.py \
  --source-manifest artifacts/scion-lite-mlx/scion-adapter.json \
  --dataset-manifest data/orpo/dataset-manifest.json \
  --output-dir artifacts/scion-lite-browser \
  --base-dir .cache/huggingface/models--google--gemma-4-E2B-it-qat-q4_0-unquantized/snapshots/1ca4dd94b623b6e0dd9da00c2239ab84b4f3e5ce
```

Validate with CourseMapper's actual current code, not a duplicated approximation:

```bash
node scripts/validate_coursemapper_package.mjs \
  --coursemapper ../CourseMapper \
  --repo-root . \
  --tier lite \
  --manifest artifacts/scion-lite-mlx/scion-adapter.json \
  --dataset-manifest data/orpo/dataset-manifest.json \
  --verify-training-run

node scripts/validate_coursemapper_package.mjs \
  --coursemapper ../CourseMapper \
  --repo-root . \
  --tier lite \
  --manifest artifacts/scion-lite-browser/scion-adapter.json \
  --dataset-manifest data/orpo/dataset-manifest.json
```

The first command verifies the dataset firewall, clean training run, manifest, and all package bytes. The second
verifies the converted browser manifest and every inherited file. Browser/device activation remains a separate
CourseMapper qualification step: a research package must not be presented as promoted merely because conversion
worked.

## Product responsibilities

The learned adapter is only one layer of the system. CourseMapper must continue to own current course retrieval,
source presentation, schema validation, tool authorization, answer-key checks, bounded repair, visible user edits,
export integrity, and exact base-only rollback. If the adapter is unavailable or fails validation, CourseMapper
should remain usable with its pinned base-only path.
