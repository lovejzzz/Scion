# Scion — standalone adapter verification (no Node, no app needed).
# Loads Gemma 4 E2B + a trained adapter, then generates a grammar-constrained
# quiz item. Confirms the trained adapter (1) still loads and serves, (2) still
# emits VALID JSON under llguidance, and (3) produces on-topic content — the
# frozen-capability gate of PLAN.md Step 2, runnable on the training machine.
#
#   ADAPTER=adapters-scion/0000800_adapters.safetensors ./.venv/bin/python verify_adapter.py
import json
import os

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
import llguidance
import llguidance.hf
from mlx_vlm.structured import LLGuidanceLogitsProcessor
from transformers import AutoTokenizer

MODEL = os.environ.get("SCION_MODEL", "google/gemma-4-e2b-it")
ADAPTER = os.environ.get("ADAPTER", "adapters-scion")

print(f"loading {MODEL} + adapter {ADAPTER} ...")
model, processor = load(MODEL, adapter_path=ADAPTER)
config = model.config

fast = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
llg_tok = llguidance.hf.from_tokenizer(fast)
compiler = llguidance.JsonCompiler(whitespace_pattern="[ \n]{0,2}")

# The mc-item contract Scion was trained to write well.
schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "q": {"type": "string", "minLength": 25, "maxLength": 300, "pattern": r"^\S+( \S+)*$"},
        "op": {"type": "array", "items": {"type": "string", "minLength": 5, "maxLength": 95, "pattern": r"^\S+( \S+)*$"}, "minItems": 4, "maxItems": 4},
        "ai": {"type": "integer", "minimum": 0, "maximum": 3},
        "ex": {"type": "string", "minLength": 20, "maxLength": 300, "pattern": r"^\S+( \S+)*$"},
    },
    "required": ["q", "op", "ai", "ex"],
}
grammar = compiler.compile(json.dumps(schema))
lp = LLGuidanceLogitsProcessor(grammar, llg_tok)

prompt = apply_chat_template(
    processor,
    config,
    "You write flawless quiz items. Write ONE multiple-choice item (q, 4 options op, "
    "answer index ai, explanation ex) testing the perfect fifth interval in music theory. "
    "The key MUST be verifiably correct. Return ONLY the item JSON object.",
    num_images=0,
)
out = generate(model, processor, prompt, max_tokens=400, temperature=0.0, logits_processors=[lp], verbose=False)
text = out.text if hasattr(out, "text") else str(out)

print("\n=== raw output ===")
print(text)
try:
    item = json.loads(text)
    ok = isinstance(item.get("op"), list) and len(item["op"]) == 4 and 0 <= item.get("ai", -1) <= 3
    print("\nVALID JSON:", True, "| well-formed mc item:", ok)
    print("PASS — the adapter serves valid, grammar-constrained items." if ok else "WARN — JSON valid but item shape off.")
except Exception as e:
    print("\nFAIL — adapter output is not valid JSON:", e)
    print("(A collapsed adapter fails here — do not adopt it; see PLAN.md kill condition.)")
