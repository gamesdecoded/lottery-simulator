# Lottery Strategy Simulator — "The Best Lottery Numbers to Play, Tested on 100 Million Tickets"

Monte Carlo simulation backing the humandecoded video **"The Best Lottery
Numbers to Play, Tested on 100 Million Tickets"**. Five bots play one 6/49 lottery ticket per draw
for 20 million draws (100M tickets total) under identical rules. Every ticket
has the same chance of winning — but popular numbers get *shared* with more
co-winners, so the strategies keep different fractions of the jackpot.

**This is a math experiment, not gambling advice.** All strategies lose
money; the finding is which loses least. See METHODS.md.

## Results

See `results.json` (written by `run_production.py`). Per strategy:
`expected_share` (fraction of a jackpot kept, ±SE), `roi_precision` (net ROI
per $1 with 95% CI — the ranking basis), `roi_direct` (realized ROI over the
100M-ticket run — noisy by construction, ~1.4 expected jackpot wins per
strategy), plus the `ranking` and adjacent-CI-overlap check.

## Files

- `engine.py` — numba engine: 6/49 draws, 3-type crowd popularity model,
  Poisson crowd winners, jackpot rollover/split accounting, 5 strategy bots,
  and the three production phases.
- `validate.py` — 9 validation gates (must all pass before production).
- `run_production.py` — Phase A (crowd jackpot process) → Phase B
  (expected-share estimation) → Phase C (direct 100M-ticket sim, checkpointed
  per 1M-draw chunk, resumable). Writes `results.json`.
- `results.json` — production results.
- `METHODS.md` — full methodology, framing, limitations.

## Run it

```bash
/home/hatch/workspace/venvs/sim/bin/python validate.py         # all gates must pass
/home/hatch/workspace/venvs/sim/bin/python run_production.py   # ~2-4 min, checkpointed
```

## Key design note

Direct ROI simulation cannot rank lottery strategies at any feasible scale:
with 20M tickets per strategy only ~1.4 jackpot wins are expected, while a
single win swings ROI by ~17 points. The ranking therefore uses a
Rao-Blackwellized estimator — E[jackpot] × E[share] / 13,983,816 — which has
the same expectation with tiny variance. The 100M-ticket direct run is real
and validates the pipeline end-to-end.
