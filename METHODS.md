# Methods — "I Simulated 100 Million Lottery Players"

## The question
Which number-picking strategy loses the least money playing the lottery?

## Framing (read first)
This is a **math experiment, not gambling advice**. Lotteries are
negative-expectation games: every strategy loses money on average, and no
number-picking method can change that. The honest finding of this study is
*which strategy loses the least*, not how to win. If you play, expect to
lose; never spend money you can't afford to lose.

## The modeled lottery
A generic 6/49 draw (6 distinct balls from 49; C(49,6) = 13,983,816
combinations), modeled on typical national-lottery rules:

| Parameter | Value |
|---|---|
| Ticket price | $3 |
| Jackpot seed | $5,000,000 |
| Jackpot funding | +$0.50 per ticket sold, each draw |
| Rollover | full jackpot rolls over when nobody wins |
| Jackpot win | split equally among ALL jackpot winners that draw |
| Match 5 (fixed) | $2,500 per winner |
| Match 4 (fixed) | $100 per winner |
| Match 3 (fixed) | $10 per winner |

Lower-tier prizes are fixed per winner (not pari-mutuel), so only the
jackpot involves sharing.

## The crowd (the splitting mechanism)
8,000,000 background tickets per draw — realistic national-lottery scale.
Crowd tickets are never materialized one by one. For a drawn combination W,
crowd jackpot winners are sampled as Poisson(8,000,000 × p_pop(W)), where
p_pop(W) is the crowd's probability of playing exactly W under a 3-type
mixture calibrated to real lottery behavior:

- **35% quick-pick** — uniform random combinations.
- **50% birthday-biased** — ball weights 6 for 1–12 (months), 4 for 13–31
  (days), 1 for 32–49. Real players overplay calendar numbers.
- **15% pattern players** — uniform over balls 1–31 (all-low combinations,
  sequences, "lucky" low numbers).

p_pop(W) = Σ_types q_k × (∏ ball weights) / Z_k, with Z_k the 6th elementary
symmetric sum of the type's weights (exact normalizer). A popular
all-birthday combination is ~24× more likely to be played by the crowd than
an all-high combination, so it splits ~24× more often.

## The five strategies (1 ticket per draw each, like a real player)
1. **quickpick** — uniform random combination every draw.
2. **birthday** — birthday-biased balls every draw (same distribution as the
   crowd's birthday type).
3. **samenumbers** — one fixed popular combination every draw
   {3, 7, 12, 19, 26, 31}.
4. **hotnumbers** — the 6 most frequent balls of the last 50 draws
   (ties → smaller ball).
5. **unpopular** — uniform over balls 32–49, avoiding the birthday/pattern
   zone entirely.

## Why the ranking needs a precision estimator
Every strategy has *identical* win probabilities: P(jackpot) = 1/13,983,816
per ticket and identical match-3/4/5 probabilities (any 6 distinct balls).
The ONLY thing that differs is jackpot split exposure.

20M tickets per strategy ⇒ ~1.4 expected jackpot wins each — far too few for
a direct ROI comparison (a single $10M win moves ROI by ~17 points; the
real gaps between strategies are a few points). A direct simulation literally
cannot resolve the ranking at any feasible scale.

So the reported ROI uses a Rao-Blackwellized (conditional-expectation)
estimator with the same expectation but tiny variance:

> per-ticket jackpot EV = E[jackpot at win] × E[share] / 13,983,816

- **E[jackpot at win]** from a 300,000-draw crowd-only rollover simulation
  (~120k observed wins; SE negligible).
- **E[share]** = E[f(8M × p_pop(W))] per strategy, Monte-Carlo estimated with
  2,000,000 combination samples each (f(λ) = E[1/(1+Poisson(λ))], the expected
  fraction of a jackpot kept). 95% CIs from 20 batch means.
- Lower tiers are exact combinatorics (identical for all strategies).

A direct 100,000,000-ticket simulation (5 strategies × 1 ticket × 20M draws)
is ALSO run — it is the video's title number and validates the pipeline
end-to-end (jackpot hit rates, lower-tier frequencies, realized splits).
Its realized per-strategy jackpot ROI is reported in `results.json` but is
explicitly NOT the ranking basis (too noisy by construction).

## Validation
`validate.py` (9 gates, all passing before production): draw uniformity
(χ²), match-k distribution vs exact combinatorics (χ²), jackpot probability
≈ 1/13,983,816 (140M draws), crowd split-frequency ratio (popular vs
unpopular), alias-sampler distribution (χ²), hand-checked mini-lottery
normalizer, hand-checked jackpot ledger (rollover/split/reset), expected-share
sanity, and determinism (same seed → identical output, required for
checkpoint resume).

## Limitations
- The crowd mixture (35/50/15) is a calibrated model, not measured data; the
  *direction* of the effect (popular numbers split more) is robust, the exact
  magnitudes depend on the real crowd mix.
- One lottery structure (6/49, $3, fixed lower tiers); Powerball-style
  multi-state games have different split dynamics.
- No taxes, no annuity-vs-cash discounting, no time value of 20M draws.
- Bots play exactly 1 ticket/draw; real syndicates and multi-ticket players
  exist but don't change the per-ticket ranking.
- The hot-numbers bot uses a fixed 50-draw window; real players vary.
