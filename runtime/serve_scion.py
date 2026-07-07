# Gemma 4 E2B local inference server (the items/authoring route).
# Same JSONL protocol as serve_s.py; mlx-vlm backend (E2B is
# multimodal-native and mlx-lm cannot load it). Zero-shot BY DESIGN —
# two measured fine-tune collapses retired SFT for this model
# (TENDRIL_ROADMAP_V0.2.md §3).
#
# V2 (E2B-MAX V2 Workstream A): grammar-constrained JSON decoding via
# llguidance (mlx_vlm.structured). A request may carry:
#   schema:   a JSON Schema dict — decoding is constrained to it
#   jsonMode: true — constrained to a permissive {"type":"object"}
# Fallback ladder, DISCLOSED per response as "constrained":
#   "schema" -> "object" (schema failed to compile) -> "none".
# The long-JSON failure class (near-miss commas/brackets, the doubled
# closing brace of ladder L2) becomes impossible at the decode layer —
# the compiler-seat autopsy proved content was never the problem.

import json
import os
import sys

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = os.environ.get("G4_MODEL", "google/gemma-4-e2b-it")
# V2.1 A4: house adapters ride an env var — base weights stay untouched and
# rollback is unsetting G4_ADAPTERS (checkpoint gates decide adoption).
ADAPTERS = os.environ.get("G4_ADAPTERS", "")
model, processor = load(MODEL, **({"adapter_path": ADAPTERS} if ADAPTERS else {}))
config = model.config

# llguidance needs a FAST tokenizer; mlx-vlm's processor is the slow
# backend, so a fast twin of the same vocab loads once for grammar use.
# Grammar is built here (NOT via mlx_vlm.structured's builder). Whitespace is
# the measured knife-edge: whitespace_pattern="" (the mlx_vlm default) bans
# the model's preferred tokens and greedy decoding cascades into
# earliest-legal-exit (770-char CourseIR); whitespace_flexible lets greedy
# pretty-print THOUSANDS of indentation tokens and the budget dies at
# finish_reason=length (17.6K chars / 5K tokens, object unfinished). The
# bounded pattern [ \n]{0,2} measured finish=stop, VALID, ~600 tokens on the
# same kernel-lesson probe — natural spacing without the flood.
_llg_tokenizer = None
_json_compiler = None
_processor_cls = None
try:
    import llguidance as _llg
    import llguidance.hf as _llg_hf
    from mlx_vlm.structured import LLGuidanceLogitsProcessor as _LLGP
    from transformers import AutoTokenizer

    _fast = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if _fast.is_fast:
        _llg_tokenizer = _llg_hf.from_tokenizer(_fast)
        _json_compiler = _llg.JsonCompiler(whitespace_pattern="[ \n]{0,2}")
        _processor_cls = _LLGP
except Exception:  # noqa: BLE001 — constrained mode is optional, never fatal
    _llg_tokenizer = None

PERMISSIVE_OBJECT = {"type": "object"}
_grammar_cache = {}


def _grammar_for(schema):
    key = json.dumps(schema, sort_keys=True)
    if key not in _grammar_cache:
        _grammar_cache[key] = _json_compiler.compile(key if isinstance(schema, str) else json.dumps(schema))
    return _grammar_cache[key]


def constrained_processor(req):
    """Return (logits_processors, tier). Fresh processor per call —
    LLGuidance FSM state advances with the sequence and must not be reused."""
    if _llg_tokenizer is None:
        return None, "none"
    schema = req.get("schema")
    if schema:
        try:
            return [_processor_cls(_grammar_for(schema), _llg_tokenizer)], "schema"
        except Exception:  # noqa: BLE001 — schema too rich for llguidance
            pass
    if schema or req.get("jsonMode"):
        try:
            return [_processor_cls(_grammar_for(PERMISSIVE_OBJECT), _llg_tokenizer)], "object"
        except Exception:  # noqa: BLE001
            pass
    return None, "none"


print(json.dumps({"ready": True, "constrained": _llg_tokenizer is not None}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        prompt = apply_chat_template(processor, config, f"{req['system']}\n\n{req['user']}", num_images=0)
        processors, tier = constrained_processor(req)
        # temperature>0 enables the best-of-N sampling harness (E2B-MAX);
        # default stays greedy for deterministic single-shot authoring.
        out = generate(
            model,
            processor,
            prompt,
            max_tokens=int(req.get("maxTokens", 1200)),
            temperature=float(req.get("temperature", 0.0)),
            verbose=False,
            **({"logits_processors": processors} if processors else {}),
        )
        text = (out.text if hasattr(out, "text") else str(out)).strip()
        print(json.dumps({"id": req.get("id"), "text": text, "constrained": tier}), flush=True)
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"id": req.get("id") if isinstance(req, dict) else None, "error": str(error)[:200]}), flush=True)
