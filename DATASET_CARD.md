# Dataset card: Scion CourseMapper education v2

## Purpose and composition

This supervised fine-tuning corpus teaches Bonsai 27B the JSON contracts used by CourseMapper. It
contains 1,465 admitted examples:

| Split | Total | Course groups |
|---|---:|---:|
| Train | 711 | 10 |
| Validation | 297 | 6 |
| Test | 457 | 7 |

Response types are complete lesson kernels, multiple-choice items, key terms, and source bundles.
Domains include astronomy, computer science, economics, geology, language learning, music theory,
nutrition, nursing, psychology, statistics, UX design, and world literature.

## Sources

1. Teacher-style chosen outputs from the prior Scion repository at exact revision
   `7f97e9b7f995bb7bf74eedd0c07fa8ca291f1d06`, normalized to the current CourseMapper contract.
2. Stable, order-swapped preference winners from CourseMapper at exact revision
   `4f5bed3833f72494917e67c1a0c878af8c2b9a70`.
3. CourseMapper compiler-admitted source captures derived from attributed OpenStax, W3C,
   Digital.gov, Open Geology, and music-theory source packets.

`data/manifest.json` records source hashes, exact revisions, per-file hashes, counts, and split
membership. Per-example provenance is stored separately in `data/metadata`; training JSONL contains
only chat messages.

## Processing and leakage controls

- Outputs are normalized to the current abbreviated CourseMapper schemas.
- Every assistant response passes deterministic contract admission before writing.
- Exact response duplicates are removed.
- Entire course groups are assigned to only one split; group intersections are rejected.
- Held-out evaluation fixtures come from test groups and are never copied into training messages.
- Local absolute paths are removed from published provenance.

## Known limitations

The corpus does not contain student personal data or classroom outcome measurements. Most examples
are machine-authored, and stable automated preference selection is not independent instructor
review. Source-grounded coverage is deeper in four domains than elsewhere. Users should treat the
dataset as task-format specialization, not a comprehensive or authoritative curriculum.

## Licensing

Repository-authored transformations are distributed under Apache-2.0. Source-capture records retain
their own attribution and license labels in fixture provenance, including CC-BY-4.0,
CC-BY-NC-SA-4.0, the W3C Document License, and U.S. Government Work. Downstream users are
responsible for preserving applicable attribution and non-commercial restrictions when reusing
source-derived material beyond model evaluation.
