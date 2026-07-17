# Third-party notices

Scion does not bundle its base model or runtime binaries.

- **PrismML Bonsai 27B** — training and serving checkpoints published by PrismML under the terms
  stated in their Hugging Face repositories; the checked model metadata identifies Apache-2.0.
- **PrismML llama.cpp** — pinned fork of llama.cpp, built locally and subject to its upstream
  license and notices.
- **MLX and MLX LM** — Apple machine-learning libraries used for local QLoRA preparation and
  training, subject to their upstream licenses.
- **Hugging Face Hub, safetensors, PyTorch, NumPy** — tooling used for immutable checkpoint
  retrieval and adapter conversion, subject to their upstream licenses.

Educational source attributions and per-source license labels are retained in `eval/fixtures.jsonl`
and summarized in `DATASET_CARD.md`.
