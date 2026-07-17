# Third-party notices

Scion does not commit or redistribute foundation-model weights.

- **Qwen3.6 27B and optional Qwen3.5 122B** — local teacher checkpoints published by the Qwen/MLX community and
  identified by their model cards as Apache-2.0.
- **Gemma 4 E2B, 12B, and 31B** — student and critic checkpoints published by Google or MLX community conversions
  and identified by their model cards as Apache-2.0. Users remain responsible for reviewing the upstream Gemma
  terms and acceptable-use documentation.
- **MLX, MLX-LM, and MLX-VLM** — Apple Silicon inference and training libraries under their upstream licenses.
- **llama.cpp** — the pinned MLX-LoRA-to-GGUF conversion tool and CourseMapper browser runtime foundation under its
  upstream MIT license and notices.
- **Hugging Face Hub, Transformers, PyTorch, Datasets, safetensors, tokenizers, NumPy, and PyArrow** — model
  retrieval, processor compatibility, data, and serialization libraries under their respective upstream licenses.

The narrow MLX-to-PEFT bridge in `scripts/convert_mlx_lora_to_peft.py` is derived from the Apache-2.0 CourseMapper
implementation at revision `4f5bed3833f72494917e67c1a0c878af8c2b9a70` and remains Apache-2.0.

Exact package versions, revisions, hashes, and model licenses are recorded in the model registry and run receipts.
