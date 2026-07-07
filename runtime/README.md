# runtime/

- **serve_scion.py** — the Scion inference server: Gemma 4 E2B under
  llguidance grammar-constrained decoding, JSONL stdin/stdout protocol.
  STANDALONE (only needs the MLX venv). Reads `G4_MODEL` and `G4_ADAPTERS`
  env vars — set `G4_ADAPTERS=adapters-scion/...` to serve a trained adapter.
  This is what `verify_adapter.py` and CourseMapper's local provider drive.

- **scion_openai_server.mjs** — an OpenAI-compatible HTTP facade over
  serve_scion.py (answers /v1/chat/completions, /v1/models, /v1/flywheel;
  grammar contract enforcement + the D3 quality passes). It is the app-facing
  surface. NOTE: its import paths (`sModel.mjs`, the venv/serve locations)
  assume the CourseMapper repo layout — it is included here as the reference
  copy; it runs in CourseMapper, not standalone. Training does not need it.

- **sModel.mjs** — the Node↔Python bridge scion_openai_server.mjs uses. Same
  caveat: CourseMapper-relative paths.
