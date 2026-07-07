# The beat-mini gauntlet

Measuring trained-Scion vs `gpt-5.4-mini` needs the CourseMapper app (its
compiler + judge panel), so the gauntlet runs THERE, not in this repo. Once
`train.sh` here produces a `adapters-scion/` you're happy with (valid JSON per
`verify_adapter.py`), do this in the CourseMapper checkout:

```bash
# 1. drop the trained adapter into CourseMapper
cp -r adapters-scion  <CourseMapper>/trellis/tendril/distill/adapters-g4-orpo

# 2. serve Scion + adapter and run the compiler seat + pooled judge vs mini
cd <CourseMapper>
ADAPTER=trellis/tendril/distill/adapters-g4-orpo \
  bash trellis/tendril/distill/scion_gauntlet.sh
```

`scion_gauntlet.sh` serves the adapter, smoke-checks JSON validity, compiles
`music-theory` through the Local provider, runs a 5-seat judge panel, and
prints a verdict against mini's pooled **6.08**. Run it twice (10 pooled seats)
before trusting the number — single panels carry ±0.9 noise.

**Adopt** the adapter (make it the `serve_g4.py` default via `G4_ADAPTERS`)
only if the pooled mean clears 6.08 AND the long-JSON / scoreboard benches
stay green. Otherwise: discard the file, grow the corpus (Step 4 in PLAN.md),
retrain.
