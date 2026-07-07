// Shared kernel-lesson JSON Schema (V2.1) — the per-lesson contract the local
// server enforces at decode time, extracted so the corpus builder and the
// serving shim describe the SAME distribution. Derived from the app's own
// kernel contract + lint floor (buildLessonKernelPrompt /
// lintKernelFact ≥25ch / lintEnrichedKeyTerm df ≥45ch / exactly-4 options)
// plus the measured anti-degeneration string pattern (round-17: greedy
// space-runs inside values ate whole token budgets).
export const NO_SPACE_RUNS = '^\\S+( \\S+)*$';
export const str = (minLength, maxLength) => ({ type: 'string', minLength, maxLength, pattern: NO_SPACE_RUNS });
export const arr = (items, minItems, maxItems) => ({ type: 'array', items, minItems, maxItems });

export function lockObjects(node) {
  if (Array.isArray(node)) return node.forEach(lockObjects);
  if (!node || typeof node !== 'object') return;
  if (node.type === 'object' && node.properties && node.additionalProperties === undefined) {
    node.additionalProperties = false;
  }
  for (const value of Object.values(node)) lockObjects(value);
}

export function kernelLessonSchema({ mcCount = 4, keyTermCount = 4 } = {}) {
  return {
    type: 'object',
    properties: {
      lessonId: { type: 'string' },
      goal: str(8, 120),
      outcomes: arr(str(12, 160), 3, 5),
      async: arr(str(8, 160), 2, 3),
      sync: arr(str(8, 160), 2, 3),
      facts: arr(str(25, 140), 5, 8),
      keyTerms: arr(
        {
          type: 'object',
          properties: { tr: str(3, 60), df: str(45, 380), eg: str(12, 300), mi: str(12, 300), cx: str(12, 300) },
          required: ['tr', 'df', 'eg', 'mi', 'cx'],
        },
        Math.max(3, keyTermCount - 1),
        6,
      ),
      scenario: { type: 'object', properties: { su: str(45, 500), ma: str(10, 300) }, required: ['su', 'ma'] },
      discussionPrompt: {
        type: 'object',
        properties: { pr: str(20, 300), tn: str(12, 300), po: arr(str(8, 200), 2, 3) },
        required: ['pr', 'tn', 'po'],
      },
      assignmentCore: {
        type: 'object',
        properties: { td: str(45, 500), pa: arr(str(8, 160), 2, 4) },
        required: ['td', 'pa'],
      },
      mc: arr(
        {
          type: 'object',
          properties: {
            q: str(25, 300),
            op: arr(str(5, 95), 4, 4),
            ai: { type: 'integer', minimum: 0, maximum: 3 },
            ex: str(20, 300),
          },
          required: ['q', 'op', 'ai', 'ex'],
        },
        mcCount,
        mcCount,
      ),
      studyGuide: { type: 'object', properties: { sm: str(70, 550), rs: str(35, 380) }, required: ['sm', 'rs'] },
    },
    required: [
      'lessonId',
      'goal',
      'outcomes',
      'async',
      'sync',
      'facts',
      'keyTerms',
      'scenario',
      'discussionPrompt',
      'assignmentCore',
      'mc',
      'studyGuide',
    ],
  };
}

export function singleLessonEnvelope(lessonId, { mcCount, keyTermCount } = {}) {
  const lessonSchema = kernelLessonSchema({ mcCount, keyTermCount });
  lessonSchema.properties.lessonId = { type: 'string', enum: [lessonId] };
  const schema = { type: 'object', properties: { lessons: arr(lessonSchema, 1, 1) }, required: ['lessons'] };
  lockObjects(schema);
  return schema;
}
