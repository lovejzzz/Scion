# CourseMapper integration

CourseMapper already defines the required local provider contract in `src/lib/localProvider.js`:

- endpoint: `http://127.0.0.1:8799`
- model: `scion-1`
- liveness: `GET /v1/models`
- generation: `POST /v1/chat/completions`
- structured output: OpenAI `response_format` is forwarded
- streaming: supported by the runtime, although the Scion evaluator uses non-streaming calls

Start the server from this repository:

```bash
scion serve \
  --adapter artifacts/scion-bonsai-27b.gguf \
  --llama-server .cache/PrismML-llama.cpp/build/bin/llama-server
```

Then start CourseMapper normally and select **Local / Scion** in its model configuration. The
provider is keyless. If another process owns port 8799, stop it before starting Scion; changing the
port requires changing CourseMapper's local endpoint setting as well.

The serving command disables the Qwen thinking channel and uses deterministic decoding. This is
intentional: CourseMapper expects the response body itself to be valid JSON, without reasoning text
before or after it.

Before promoting an adapter, run:

```bash
scion smoke --output runs/bonsai-27b/coursemapper-smoke.json
```

The smoke gate checks browser CORS preflight, calls model discovery, then uses CourseMapper's
schema-constrained SSE request shape with model `scion-1`. It requires a complete `[DONE]` stream
and admits the returned assessment item through Scion's CourseMapper contract validator.
