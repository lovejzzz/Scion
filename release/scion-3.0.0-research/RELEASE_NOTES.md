# Scion Education 3.0.0 research 1

This prerelease replaces the earlier Bonsai experiment with locally distilled Gemma 4 education adapters for
CourseMapper. It contains no foundation-model weights, closed-API output, real student records, or real catalog
data. External API/cloud spend was $0.

## Artifacts

- `scion-lite-mlx-v3.0.0-research.1.tar.gz`: 52.8 MB native MLX LoRA.
- `scion-lite-browser-v3.0.0-research.1.tar.gz`: 26.4 MB CourseMapper GGUF LoRA package at scale 10.
- `smoke-pro-mlx-v3.0.0-research.1.tar.gz`: 332.7 MB native MLX LoRA selected from the qualified 10-update Pro
  canary. The `smoke` filename is retained to prevent the canary provenance from being obscured.

The release manifest records archive and package checksums, exact bases and revisions, dataset identity, and full
paired evidence paths.

## Locked 32-task results

| Protocol | Base | Adapter | Issues before / after |
|---|---:|---:|---:|
| Lite MLX strict JSON | 2/32 | 26/32 | 30 / 9 |
| Pro MLX with lossless CourseMapper fence removal | 28/32 | 30/32 | 5 / 3 |
| Lite exact GGUF with JSON schema | 20/32 | 21/32 | 30 / 29 |

The browser-runtime improvement is modest and must not be replaced by the stronger native MLX number in product
claims. The GGUF evaluation used pinned llama.cpp rather than a WebGPU device, so browser/device activation remains
a separate CourseMapper qualification step.

## Status

Research only; `promotable` is false. The 224-pair, four-domain synthetic corpus passes the research gate but not
CourseMapper's production requirement of at least 3,000 verified pairs and five domains. Human instructor review,
real-world catalog evaluation, and WebGPU device-matrix evidence are still required before production promotion.
