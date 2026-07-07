// Tendril-S from Node (zero-API mode) — wraps the persistent Python
// inference server (distill/serve_s.py). $0 per call, local only. The
// deployment prompts here are the EXACT single-entry prompts S was
// trained on (distill/prep_data.py) — serving a model off its training
// distribution is a self-inflicted wound.

import { spawn } from 'node:child_process';

export const SKIN_SYSTEM =
  "You are the course's own instructor unifying a lesson plan assembled from proven parts. Rewrite the segment MINIMALLY so it reads as one instructor: fix week/lesson references, add one-clause transitions where segments collide, unify register. NEVER change technical content, examples, numbers, or code; never add new claims; keep the rewrite within ±40% of the original length. Return only the rewritten segment text.";
export const BLEND_SYSTEM =
  "You polish quiz explanations. The text contains corrective sentences that were pasted in mechanically, so it reads as two voices. Rewrite it as ONE natural explanation (2-3 sentences) that makes every corrective's content its own point — keep the key technical terms (a lexical gate checks this), never paste a corrective as a standalone sentence. Return only the rewritten explanation text.";

// TASK-ROUTED local serving (v0.2, 'the better model is a pair'): the
// held-out gate bench measured complementary strengths — Qwen2.5-0.5B
// (s3b, 800-iter checkpoint) wins SKIN 71.7% vs 61.7%, while the
// SmolLM2 round-2 tune keeps BLEND 83.3% vs 61.7%. Routing by task
// scores 77.5% combined vs 72.5% single-model. Servers start lazily.
const ROUTES = {
  skin: {
    base: 'Qwen/Qwen2.5-0.5B-Instruct',
    adapters: 'trellis/tendril/distill/adapters-s3-800',
  },
  blend: {
    base: 'HuggingFaceTB/SmolLM2-135M-Instruct',
    adapters: 'trellis/tendril/distill/adapters',
  },
  // items: Gemma 4 E2B zero-shot via mlx-vlm (plan v0.2 A1) — beat the
  // paid author 26/30 vs 22/30 on its own gates at $0.
  items: {
    python: 'trellis/tendril/.venv-g4/bin/python',
    script: 'trellis/tendril/distill/serve_g4.py',
  },
};

// Scion (the V2.1 house model name) IS the g4 server — the alias resolves
// to the same route entry so skin/polish/fill seats share one loaded 4B
// process with the items author instead of spawning a twin.
export function resolveRoute(task) {
  if (task === 'scion') return 'items';
  return ROUTES[task] ? task : 'skin';
}

let nextId = 1;
const servers = new Map(); // route -> { proc, pending }

async function startRoute(route, { timeoutMs = 120_000 } = {}) {
  if (servers.has(route)) return servers.get(route);
  const cfg = ROUTES[route];
  const proc = spawn(
    cfg.python ?? 'trellis/tendril/.venv/bin/python',
    [cfg.script ?? 'trellis/tendril/distill/serve_s.py'],
    {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        ...(cfg.base ? { S_BASE: cfg.base } : {}),
        ...(cfg.adapters ? { S_ADAPTERS: cfg.adapters } : {}),
      },
    },
  );
  const pending = new Map();
  const entry = { proc, pending };
  servers.set(route, entry);
  let buffer = '';
  proc.stdout.on('data', (chunk) => {
    buffer += chunk.toString();
    let nl;
    while ((nl = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line) continue;
      try {
        const msg = JSON.parse(line);
        if (msg.ready) {
          pending.get('ready')?.resolve();
          pending.delete('ready');
        } else if (pending.has(msg.id)) {
          const p = pending.get(msg.id);
          pending.delete(msg.id);
          if (msg.error) p.reject(new Error(msg.error));
          else p.resolve(msg.text);
        }
      } catch {
        /* non-JSON stdout chatter ignored */
      }
    }
  });
  proc.on('exit', () => {
    for (const p of pending.values()) p.reject?.(new Error(`tendril-s [${route}] server exited`));
    pending.clear();
    servers.delete(route);
  });
  await new Promise((resolve, reject) => {
    pending.set('ready', { resolve, reject });
    setTimeout(() => reject(new Error(`tendril-s [${route}] did not start (venv built?)`)), timeoutMs);
  });
  return entry;
}

export async function startS(options = {}) {
  await startRoute('skin', options); // blend starts lazily on first use
}

export async function sGenerate(
  // V2: schema (JSON Schema dict) / jsonMode (bool) engage llguidance
  // grammar-constrained decoding on the g4 route (serve_g4.py) — parse
  // validity by construction. Ignored by serve_s routes.
  { system, user, source = '', task = 'skin', maxTokens, temperature, schema, jsonMode },
  { timeoutMs = 180_000 } = {},
) {
  const route = resolveRoute(task);
  const entry = await startRoute(route);
  const id = String(nextId++);
  const promise = new Promise((resolve, reject) => {
    entry.pending.set(id, { resolve, reject });
    setTimeout(() => {
      if (entry.pending.has(id)) {
        entry.pending.delete(id);
        reject(new Error('tendril-s timeout'));
      }
    }, timeoutMs);
  });
  entry.proc.stdin.write(
    `${JSON.stringify({
      id,
      system,
      user,
      source,
      ...(maxTokens ? { maxTokens } : {}),
      ...(temperature ? { temperature } : {}),
      ...(schema ? { schema } : {}),
      ...(jsonMode ? { jsonMode: true } : {}),
    })}\n`,
  );
  return promise;
}

export function stopS() {
  for (const { proc } of servers.values()) proc.kill();
  servers.clear();
}
