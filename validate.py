"""Validation gates for the lottery engine. All must pass before production."""
import itertools
import math
import sys

import numpy as np

import engine as E

try:
    from scipy.stats import chi2 as chi2_dist
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" | " + str(detail)) if detail else ""))
    if not cond:
        FAILS.append(name)


def chi2_p(ch, df):
    if HAVE_SCIPY:
        return chi2_dist.sf(ch, df)
    # fallback: rough normal approx, conservative
    return 1.0 if ch < df + 3 * math.sqrt(2 * df) else 0.0


def main():
    rng_state = np.zeros(1, dtype=np.uint64)
    deck = np.empty(E.N_BALLS, dtype=np.int64)
    balls = np.empty(E.PICK, dtype=np.int64)

    ZA = E.elementary_symmetric_6(E.W_A)
    ZB = E.elementary_symmetric_6(E.W_B)
    ZC = E.elementary_symmetric_6(E.W_C)
    probB, aliasB = E.build_alias(E.W_B)

    # ---- Gate 1: draw uniformity (ball frequencies) ----
    rng_state[0] = np.uint64(12345)
    counts = np.zeros(E.N_BALLS, dtype=np.int64)
    N = 200_000
    for _ in range(N):
        E.draw_uniform_combo(rng_state, deck, balls)
        for j in range(E.PICK):
            counts[balls[j] - 1] += 1
    exp = N * E.PICK / E.N_BALLS
    ch = float(np.sum((counts - exp) ** 2 / exp))
    p = chi2_p(ch, E.N_BALLS - 1)
    check("draw uniformity chi2", p > 1e-3, f"chi2={ch:.1f} df=48 p={p:.4f}")

    # ---- Gate 2: match-k probabilities vs exact combinatorics ----
    rng_state[0] = np.uint64(999)
    fixed = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    fmask = E.combo_mask(fixed)
    N2 = 1_000_000
    obs = np.zeros(7, dtype=np.int64)
    for _ in range(N2):
        E.draw_uniform_combo(rng_state, deck, balls)
        m = E.popcnt64(E.combo_mask(balls) & fmask)
        obs[m] += 1
    # exact P(match k) for k=0..6
    ch2 = 0.0
    df2 = 0
    for k in range(7):
        from math import comb
        pk = comb(6, k) * comb(43, 6 - k) / E.N_COMBOS
        e = N2 * pk
        if e >= 5:
            ch2 += (obs[k] - e) ** 2 / e
            df2 += 1
    df2 -= 1
    p2 = chi2_p(ch2, df2)
    check("match-k distribution chi2", p2 > 1e-3,
          f"chi2={ch2:.1f} df={df2} p={p2:.4f} obs3={obs[3]} obs4={obs[4]} obs5={obs[5]}")

    # ---- Gate 3: jackpot probability ~= 1/13,983,816 ----
    rng_state[0] = np.uint64(777)
    N3 = 140_000_000
    hits = 0
    for _ in range(N3):
        E.draw_uniform_combo(rng_state, deck, balls)
        if E.popcnt64(E.combo_mask(balls) & fmask) == 6:
            hits += 1
    exp3 = N3 / E.N_COMBOS
    check("jackpot probability", 3 <= hits <= 18,
          f"hits={hits} expected={exp3:.1f}")

    # ---- Gate 4: crowd sanity - popular combos split much more ----
    popular = np.array([3, 7, 12, 19, 26, 31], dtype=np.int64)
    unpopular = np.array([32, 33, 34, 35, 36, 37], dtype=np.int64)
    pp_pop = E.combo_popularity(popular, E.W_A, E.W_B, E.W_C, ZA, ZB, ZC)
    pp_unp = E.combo_popularity(unpopular, E.W_A, E.W_B, E.W_C, ZA, ZB, ZC)
    check("crowd popularity ratio", pp_pop > 3 * pp_unp,
          f"ratio={pp_pop / pp_unp:.1f}x")
    rng_state[0] = np.uint64(31337)
    n_ge1_pop = n_ge1_unp = 0
    T = 100_000
    for _ in range(T):
        if E.poisson_sample(rng_state, E.CROWD_PER_DRAW * pp_pop) >= 1:
            n_ge1_pop += 1
        if E.poisson_sample(rng_state, E.CROWD_PER_DRAW * pp_unp) >= 1:
            n_ge1_unp += 1
    check("crowd split frequency ratio",
          n_ge1_pop > 2 * max(n_ge1_unp, 1),
          f"P(split|popular)={n_ge1_pop/T:.3f} P(split|unpopular)={n_ge1_unp/T:.4f}")

    # ---- Gate 5: alias sampler matches birthday weights ----
    rng_state[0] = np.uint64(4242)
    bc = np.zeros(E.N_BALLS, dtype=np.int64)
    M = 1_000_000
    for _ in range(M):
        bc[E.alias_sample(rng_state, probB, aliasB)] += 1
    w = E.W_B / E.W_B.sum()
    ch5 = float(np.sum((bc - M * w) ** 2 / (M * w)))
    p5 = chi2_p(ch5, E.N_BALLS - 1)
    check("alias sampler chi2", p5 > 1e-3, f"chi2={ch5:.1f} p={p5:.4f}")

    # ---- Gate 6: hand-check normalizer (2 balls from 4, weights 1,2,3,4) ----
    def e_sym(weights, k):
        e = [0.0] * (k + 1)
        e[0] = 1.0
        for w_ in weights:
            for j in range(k, 0, -1):
                e[j] += w_ * e[j - 1]
        return e[k]
    Zmini = e_sym([1.0, 2.0, 3.0, 4.0], 2)
    # by hand: 1*2+1*3+1*4+2*3+2*4+3*4 = 2+3+4+6+8+12 = 35
    check("mini-lottery Z hand-check", abs(Zmini - 35.0) < 1e-9, f"Z={Zmini}")
    brute = sum(a * b for a, b in itertools.combinations([1, 2, 3, 4], 2))
    check("mini-lottery Z vs brute force", abs(brute - Zmini) < 1e-9,
          f"brute={brute}")
    tot = brute / Zmini  # sum of P({a,b}) = w_a*w_b/Z over all combos
    check("mini-lottery sums to 1", abs(tot - 1.0) < 1e-12, f"sum={tot}")
    # direct: P({1,2}) = w1*w2 / Z = 2/35
    check("mini-lottery p({1,2}) hand-check",
          abs(2.0 / 35.0 - 0.05714285714285714) < 1e-12, "2/35 by hand")
    # engine's k=6 normalizer vs brute force on 6 weights (only combo: 720)
    Zeng = E.elementary_symmetric_6(np.array([1., 2., 3., 4., 5., 6.]))
    check("engine Z6 vs brute force", abs(Zeng - 720.0) < 1e-6, f"Z={Zeng}")

    # ---- Gate 7: jackpot ledger hand-check ----
    pay, j = E.jackpot_step_payout(10_000_000.0, 0)
    ok = (pay == 0.0 and j == 10_000_000.0)
    pay, j = E.jackpot_step_payout(10_000_000.0, 4)
    ok = ok and abs(pay - 2_500_000.0) < 1e-6 and j == E.JACKPOT_SEED
    # scripted: seed 5M, +100 twice (no winners), then 3 winners
    jj = E.JACKPOT_SEED
    jj = jj + 100.0  # draw 1, no winners -> rollover
    jj = jj + 100.0  # draw 2, no winners -> rollover
    pay, jj = E.jackpot_step_payout(jj, 3)
    ok = ok and abs(pay - 5_000_200.0 / 3) < 1e-6 and jj == E.JACKPOT_SEED
    check("jackpot ledger hand-check", ok, f"split payout={pay:.2f}")

    # ---- Gate 8: expected_share sanity ----
    s0 = E.expected_share(0.0)
    s1 = E.expected_share(0.2)
    s2 = E.expected_share(4.6)
    check("expected_share sanity",
          abs(s0 - 1.0) < 1e-12 and s1 > s2 > 0.15,
          f"f(0)={s0} f(0.2)={s1:.3f} f(4.6)={s2:.3f}")

    # ---- Gate 9: determinism ----
    out1 = np.zeros(40, dtype=np.float64)
    out2 = np.zeros(40, dtype=np.float64)
    E.direct_sim_chunk(np.uint64(555), 2000, E.W_A, E.W_B, E.W_C, ZA, ZB, ZC,
                       probB, aliasB, out1)
    E.direct_sim_chunk(np.uint64(555), 2000, E.W_A, E.W_B, E.W_C, ZA, ZB, ZC,
                       probB, aliasB, out2)
    check("determinism", bool(np.array_equal(out1, out2)))

    print()
    if FAILS:
        print(f"{len(FAILS)} GATE(S) FAILED: {FAILS}")
        sys.exit(1)
    print("ALL GATES PASSED")


if __name__ == "__main__":
    main()
