# E2B-MAX V2.1 — Close the Gap, Ship the Surface

_Authored 2026-07-07, immediately after the V2 compiler-seat campaign
(BAKEOFF addendum 4: 3.33 → 5.83-best / 5.2-pooled at $0; paid mini pooled
6.08; best draw TIED same-day). V2 proved the harness ceiling and located
the remaining deficit to ONE artifact class. V2.1 exists to (1) close that
deficit with the now-unlocked weights campaign, (2) ship the local model as
a real product surface, and (3) give the house model its name._

## 0. The V2 verdict that shapes V2.1

- **Where E2B wins**: study guides 6.0–6.8 vs paid 5.0–5.3 (the self-refine
  polish pass); lesson plans at parity (5.7–6.2 vs 5.33).
- **Where it loses**: quiz items (4.7–5.5 vs 6–6.3) — stem fluency and key
  reliability. That artifact decides the overall mean.
- **Instrument facts**: single judge panels are ±0.9 noise on identical
  configs; the deepseek cross-family seats confirm the gap is real. Claims
  in V2.1 use pooled ≥12-seat panels, nothing less.
- **The unlock**: Phase-0 spike GREEN — ORPO preference training runs on
  Gemma-4-E2B via mlx_vlm.lora (adapter trains, saves, loads, serves).
  Preference-not-imitation, the standing rule since the SFT collapses.

## 1. The name

The harness config keeps its lab name (E2B-MAX), but the customized house
model deserves its own. Proposal, in the repo's own garden vocabulary
(Trellis, Tendril, the graft of a wild base onto house cultivation):

> **SCION** — in horticulture, the cultivated cutting grafted onto wild
> rootstock; in lineage, the heir of the house. Gemma is the rootstock;
> the Trellis harness and the house adapters are the scion. The first
> trained cut ships as **Scion-1**.

Alternates considered: _Espalier_ (a tree trained on a trellis — thematically
perfect, phonetically unkind), _Cordon_, _Graft_.

**ADOPTED by the owner, 2026-07-07.** UI ships Provider "Local" · API key
"Free" · Model "Scion-1" (src/lib/localProvider.js is the single source).

## 2. Workstream A — the weights campaign (teacher-preference ORPO)

The highest-signal corpus is one V2 accidentally designed: **mini's kernel
outputs judge ~1 point higher, so (mini output = chosen, E2B output =
rejected) pairs exist by construction.**

- **A1. Corpus machinery** (`trellis/tendril/distill/buildTeacherPairs.mjs`):
  for each catalog lesson (12 disciplines, ~10 reference courses, the
  crucible course pool), author the SAME per-lesson kernel contract twice —
  gpt-5.4-mini (chosen, ~$0.01/lesson) and E2B greedy (rejected, $0) — into
  `data-g4-orpo/` as {prompt, chosen, rejected}. Atom-level pairs ride
  along free: every mc item whose key fails the two-solve blind check, every
  lint-rejected atom, pairs against its accepted sibling.
- **A2. Poison filters, build-time REJECTS** (the SFT-collapse law):
  chosen/rejected similarity ceiling, margin floor (drop both-bad pairs),
  per-discipline caps, dedupe at the standing ε. Held-out audit before any
  training run.
- **A3. ORPO items-v2 rounds** (`run_orpo_g4.sh`): LoRA rank 8–16,
  checkpoints every 100 steps, EVERY checkpoint gated on the frozen rulers
  (long-JSON bench, showdown, scoreboard, battery) — a seat win that drifts
  any other ruler is rejected. Kill condition stands: two ruler-rejected
  rounds re-retires the seat's weights for this version.
- **A4. Serving**: `serve_g4.py` gains `G4_ADAPTERS` (adapter_path load);
  sModel's items route passes it; the shim/local server inherits it. Base
  weights never touched — rollback is unsetting an env var.
- **A5. The gauntlet**: compiler-seat rerun with the adapter; pooled
  ≥12-seat panels vs paid mini's pooled 6.08. The quiz artifact moving from
  ~4.8 to ~6 crosses the overall mean — that is the beat, measured honestly.

Budget: corpus ~$10–20 paid (mini authoring + spot labels), training $0 +
electricity. Multi-session by design; the machinery ships in session one.

## 3. Workstream B — the Local provider (the product surface)

The V2 shim graduates from test harness to the app's first local provider:

- **B1. Server**: the OpenAI-compatible local server (`npm run local-model`)
  gains CORS, `GET /v1/models`, and real SSE with keep-alive heartbeats —
  long on-device generations must not trip the app's 120s stream-inactivity
  timeout (the wall the crucible cache had to engineer around).
- **B2. Provider**: `local` joins the provider registry mirroring the
  deepseek pattern (OpenAI-shaped, different base URL) with three
  differences: **keyless** (webllm precedent), **$0 cost rows**, and a
  static capability profile (chat-completions, jsonMode + jsonSchema via
  llguidance, no tool calling). Landing's dropdown shows
  **Local (this device)** with the house model; "Connected" = the server
  answering `/v1/models`.
- **B3. Honesty in UI**: the model row is labeled with its honest quality
  band (the V2 pooled numbers) until A5's gauntlet upgrades it.

## 4. Workstream C — instrument and carry fixes

- Pooled-panel mode in advisoryJudge (seats × runs; never read one panel).
- The courseLevel attempt-1 deterministic failure (autopsy: first call
  returns unconstrained output; the 0.7 retry rescues it — find why the
  first grammar pass disengages).
- The "Major scale" citation false-positive in the grader's vocabulary
  heuristic (an on-topic reading flagged for zero lexical overlap).
- Upstream candidates (flag-gated, all-provider wins): per-lesson kernel
  calls (A3 of V2), the self-refine polish pass, the mc self-verify.

## 4b. Workstream D — the Scion-native compiler (adopted 2026-07-07)

The compiler currently treats Scion as a stranger; the server reverse-
engineers the app's own prompts to know what to enforce. We own both sides
of the wire — dissolve the disguise:

- **D1. Contract handoff**: when provider is Scion, every structure call
  ships its REAL contract as `response_format: json_schema` (the kernel
  lesson contract, the skeleton shape, CourseIR's outputContract) — the
  server enforces what the app declares instead of sniffing prompts. Pass B
  becomes per-lesson calls as a first-class plan (progress UI gets honest
  "lesson 3 of 7"); per-call latency drops inside every timeout.
- **D2. Time-planner, not cost-planner**: the scarce resource is minutes.
  Skip the CourseIR direct-authoring call (never once passed acceptance on
  Scion OR paid — 60-90s of deterministic fallback); never retry greedy
  identically (temperature ladder on content-rejection retries only);
  long per-call timeout profile for local.
- **D3. Quality passes promoted into the compiler** (flag-gated, default on
  for Scion, available to all providers): blind-solve quiz-key verification
  with regeneration, the lexical topic gate, and the self-refine polish
  pass — the three passes that moved the judge, currently invisible inside
  the server.
- **D4. The flywheel hook (house-model exclusive)**: local generations bank
  training signal ON-DEVICE — verified pairs and verdicts POST to the local
  server's /v1/flywheel and append to the ORPO corpus; nothing leaves the
  machine. The app reads the served identity (Scion-1 vs Scion-1+adapter)
  from /v1/models so adapter adoption is an env var the UI understands.

Compatibility: the server keeps its prompt-sniffing tier for the legacy
crucible reroute path (paid-model prompts through playwright); app-direct
Scion traffic uses declared contracts only.

## 5. Exit bars (pre-registered)

| Bar            | Instrument                               | V2 today        | **V2.1 bar**                                           |
| -------------- | ---------------------------------------- | --------------- | ------------------------------------------------------ |
| Compiler seat  | pooled ≥12-seat panels, same course      | 5.2 (paid 6.08) | **> paid pooled**                                      |
| Quiz artifact  | per-artifact panel means                 | 4.7–5.5         | **≥ 6.0**                                              |
| Frozen rulers  | long-JSON, showdown, scoreboard, battery | green           | **no drift at any adopted checkpoint**                 |
| Local provider | app end-to-end, keyless                  | n/a             | **generate a full course from the Landing page at $0** |
| Honesty        | disclosure discipline                    | —               | quality band shown in-app until the gauntlet passes    |

## 6. The vision, one paragraph

A course engine whose default brain lives on the machine that runs it.
mini charges $0.07 every compile; Scion at parity is $0 forever — unlimited
regeneration, sync-on-every-edit, offline, private, and improving with every
verified pair the flywheel banks. V2 built the runtime (grammar-enforced
contracts, chunked generation, self-verification, self-polish); V2.1 trains
the resident and hands users the switch.
