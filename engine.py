"""Lottery 6/49 simulation engine — humandecoded video #4.

Question: which number-picking strategy loses the least money?
Mechanism: jackpot SPLITTING. Every ticket has identical win probability,
but popular numbers (birthdays!) are shared with more co-winners, so the
payout per winning ticket is smaller.

Model
-----
* 6/49 draw: 6 distinct balls from 49. C(49,6) = 13,983,816 combinations.
* Ticket price $3. Fixed lower-tier prizes: match 5 = $2,500, match 4 = $100,
  match 3 = $10.
* Jackpot: seeds at $5M, grows by $0.50 per ticket sold each draw, rolls over
  when nobody wins, splits equally among ALL jackpot winners (crowd +
  strategies) when won, then resets to seed.
* Crowd: 8,000,000 background tickets per draw (realistic national-lottery
  scale). Crowd tickets are never materialized: for a drawn combination W,
  the number of crowd jackpot winners is sampled directly as
  Poisson(N_crowd * p_pop(W)), where p_pop(W) is the crowd's probability of
  playing exactly W under a 3-type mixture model:
    - 35% quick-pick (uniform),
    - 50% birthday-biased (ball weights 6 for 1-12, 4 for 13-31, 1 for 32-49),
    - 15% pattern players (uniform over 1-31).
  Lower-tier prizes are fixed per winner, so the crowd only matters for
  jackpot splits and rollover dynamics.
* Strategies (1 ticket per draw each, like a real player):
    1. quickpick  - uniform random combination every draw
    2. birthday   - birthday-biased balls every draw (same as crowd type B)
    3. samenumbers- one fixed popular combination every draw {3,7,12,19,26,31}
    4. hotnumbers - the 6 most frequent balls of the last 50 draws
    5. unpopular  - uniform over balls 32-49 (avoids birthday/pattern zone)

Key statistical fact
-------------------
P(match exactly k) and P(jackpot hit) are IDENTICAL for every strategy
(any 6 distinct balls). Only the jackpot SPLIT exposure differs. So the
ranking is estimated with a Rao-Blackwellized (conditional-expectation)
estimator: per-ticket jackpot EV = E[jackpot] * E[share] / 13,983,816,
where E[jackpot] comes from a long crowd-only rollover simulation and
E[share] = E[f(N_crowd * p_pop(W))] is Monte-Carlo estimated per strategy
with millions of combo samples (tiny variance). A direct 100M-ticket
simulation is ALSO run (the title number) to validate the pipeline
end-to-end; its realized jackpot ROI is reported but is far too noisy
(~1.4 expected jackpot wins per strategy) to rank on.
"""

import math
import numpy as np
import numba
from numba import njit

# ---------------------------------------------------------------- constants
N_BALLS = 49
PICK = 6
N_COMBOS = 13983816          # C(49,6)
TICKET_PRICE = 3.0
JACKPOT_SEED = 5_000_000.0
JACKPOT_PER_TICKET = 0.50    # jackpot pool contribution per ticket sold
PRIZE_MATCH5 = 2500.0
PRIZE_MATCH4 = 100.0
PRIZE_MATCH3 = 10.0
CROWD_PER_DRAW = 8_000_000
HOT_WINDOW = 50              # draws of history the hot-numbers bot uses

# crowd mixture: (fraction, ball weights)
Q_A, Q_B, Q_C = 0.35, 0.50, 0.15
W_A = np.ones(N_BALLS)                                   # quick-pick: uniform
W_B = np.array([6.0 if i < 12 else 4.0 if i < 31 else 1.0
                for i in range(N_BALLS)])                # birthday bias
W_C = np.array([1.0 if i < 31 else 0.0
                for i in range(N_BALLS)])                # pattern: 1-31 only

SAME_NUMBERS = np.array([3, 7, 12, 19, 26, 31], dtype=np.int64)

STRATEGIES = ["quickpick", "birthday", "samenumbers", "hotnumbers", "unpopular"]

# exact lower-tier probabilities (combinatorics, identical for all strategies)
P_MATCH3 = 246820 / N_COMBOS    # C(6,3)*C(43,3)
P_MATCH4 = 13545 / N_COMBOS     # C(6,4)*C(43,2)
P_MATCH5 = 258 / N_COMBOS       # C(6,5)*C(43,1)
LOWER_TIER_PER_TICKET = (P_MATCH3 * PRIZE_MATCH3 +
                         P_MATCH4 * PRIZE_MATCH4 +
                         P_MATCH5 * PRIZE_MATCH5)

# ---------------------------------------------------------------- RNG
@njit
def _splitmix64(state):
    state[0] += np.uint64(0x9E3779B97F4A7C15)
    z = state[0]
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))

@njit
def rand_double(state):
    """Uniform [0,1)."""
    return (_splitmix64(state) >> np.uint64(11)) * (1.0 / 9007199254740992.0)

@njit
def randn(state):
    """Standard normal via Box-Muller."""
    u1 = rand_double(state)
    u2 = rand_double(state)
    if u1 < 1e-12:
        u1 = 1e-12
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

@njit
def poisson_sample(state, lam):
    if lam <= 0.0:
        return 0
    if lam < 30.0:
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= rand_double(state)
            if p <= L:
                break
        return k - 1
    # normal approximation for large lambda (rare in our use)
    return int(max(0.0, lam + math.sqrt(lam) * randn(state) + 0.5))

# ---------------------------------------------------------------- alias method (python build, numba sample)
def build_alias(weights):
    n = len(weights)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum() * n
    prob = np.empty(n, dtype=np.float64)
    alias = np.empty(n, dtype=np.int64)
    small = [i for i in range(n) if w[i] < 1.0]
    large = [i for i in range(n) if w[i] >= 1.0]
    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = w[s]
        alias[s] = l
        w[l] -= (1.0 - w[s])
        (small if w[l] < 1.0 else large).append(l)
    for i in small + large:
        prob[i] = 1.0
        alias[i] = i
    return prob, alias

@njit
def alias_sample(state, prob, alias):
    n = prob.shape[0]
    i = int(rand_double(state) * n)
    if i >= n:
        i = n - 1
    if rand_double(state) > prob[i]:
        return alias[i]
    return i

# ---------------------------------------------------------------- combo sampling
@njit
def draw_uniform_combo(state, deck, out):
    """6 distinct balls uniform from 1..49 (partial Fisher-Yates)."""
    for i in range(N_BALLS):
        deck[i] = i + 1
    for i in range(PICK):
        j = i + int(rand_double(state) * (N_BALLS - i))
        tmp = deck[i]
        deck[i] = deck[j]
        deck[j] = tmp
        out[i] = deck[i]

@njit
def draw_weighted_combo(state, prob, alias, out):
    """6 distinct balls from an alias distribution (rejection of dups)."""
    n = 0
    while n < PICK:
        b = alias_sample(state, prob, alias) + 1
        dup = False
        for j in range(n):
            if out[j] == b:
                dup = True
                break
        if not dup:
            out[n] = b
            n += 1

@njit
def draw_uniform_high_combo(state, deck, out):
    """6 distinct balls uniform from 32..49."""
    for i in range(18):
        deck[i] = 32 + i
    for i in range(PICK):
        j = i + int(rand_double(state) * (18 - i))
        tmp = deck[i]
        deck[i] = deck[j]
        deck[j] = tmp
        out[i] = deck[i]

@njit
def combo_mask(balls):
    m = np.uint64(0)
    for j in range(PICK):
        m |= np.uint64(1) << np.uint64(balls[j] - 1)
    return m

@njit
def popcnt64(x):
    c = 0
    while x != np.uint64(0):
        c += 1
        x &= x - np.uint64(1)
    return c

# ---------------------------------------------------------------- crowd popularity
def elementary_symmetric_6(weights):
    """6th elementary symmetric sum of weights (normalizer Z)."""
    e = np.zeros(7, dtype=np.float64)
    e[0] = 1.0
    for w in weights:
        for k in range(6, 0, -1):
            e[k] += w * e[k - 1]
    return e[6]

@njit
def combo_popularity(balls, wA, wB, wC, ZA, ZB, ZC):
    """P(crowd plays exactly this combination) under the 3-type mixture."""
    pa = 1.0
    pb = 1.0
    pc = 1.0
    for j in range(PICK):
        b = balls[j] - 1
        pa *= wA[b]
        pb *= wB[b]
        pc *= wC[b]
    return Q_A * pa / ZA + Q_B * pb / ZB + Q_C * pc / ZC

@njit
def expected_share(lam):
    """E[1/(1+X)] for X ~ Poisson(lam): expected jackpot fraction kept."""
    if lam <= 0.0:
        return 1.0
    total = 0.0
    term = math.exp(-lam)  # k = 0 term of Poisson pmf
    k = 0
    while True:
        total += term / (1.0 + k)
        k += 1
        term *= lam / k
        if term < 1e-14 and k > lam + 10:
            break
        if k > 1000:
            break
    return total

@njit
def jackpot_step_payout(jackpot, n_winners):
    """Pure jackpot accounting. Returns (payout_per_winner, new_jackpot)."""
    if n_winners > 0:
        return jackpot / n_winners, JACKPOT_SEED
    return 0.0, jackpot

# ---------------------------------------------------------------- hot-numbers helpers
@njit
def hottest6(counts, out):
    """6 balls with highest counts; ties -> smaller ball number."""
    used = np.zeros(N_BALLS, dtype=np.bool_)
    for r in range(PICK):
        best = -1
        bestc = -1
        for b in range(N_BALLS):
            if not used[b] and counts[b] > bestc:
                bestc = counts[b]
                best = b
        used[best] = True
        out[r] = best + 1

# ---------------------------------------------------------------- phase A: crowd jackpot process
@njit
def crowd_jackpot_process(seed, n_draws, wA, wB, wC, ZA, ZB, ZC,
                          aliasB_prob, aliasB_alias, win_jackpots):
    """Simulate the jackpot rollover driven by the crowd.

    Returns (n_wins, mean_jackpot_at_win, max_jackpot). win_jackpots is a
    preallocated array filled with the jackpot value at each win.
    """
    state = np.zeros(1, dtype=np.uint64)
    state[0] = np.uint64(seed)
    deck = np.empty(N_BALLS, dtype=np.int64)
    balls = np.empty(PICK, dtype=np.int64)
    jackpot = JACKPOT_SEED
    tickets_per_draw = float(CROWD_PER_DRAW + 5)  # crowd + 5 bots
    inc = JACKPOT_PER_TICKET * tickets_per_draw
    strat_win_rate = 5 / N_COMBOS  # bot jackpot hits reset too
    n_wins = 0
    max_j = 0.0
    for _ in range(n_draws):
        draw_uniform_combo(state, deck, balls)
        pp = combo_popularity(balls, wA, wB, wC, ZA, ZB, ZC)
        lam_c = CROWD_PER_DRAW * pp
        c_crowd = poisson_sample(state, lam_c)
        c_strat = poisson_sample(state, strat_win_rate)
        n_w = c_crowd + c_strat
        if n_w > 0:
            win_jackpots[n_wins] = jackpot
            n_wins += 1
            if jackpot > max_j:
                max_j = jackpot
            jackpot = JACKPOT_SEED
        else:
            jackpot += inc
    mean_j = 0.0
    for i in range(n_wins):
        mean_j += win_jackpots[i]
    if n_wins > 0:
        mean_j /= n_wins
    return n_wins, mean_j, max_j

# ---------------------------------------------------------------- phase B: expected-share estimation
@njit
def estimate_shares(seed, n_samples, wA, wB, wC, ZA, ZB, ZC,
                    aliasB_prob, aliasB_alias, means, ses):
    """Monte-Carlo E[share] per strategy. means/ses: arrays len 5 (output).

    Strategies: 0 quickpick, 1 birthday, 2 samenumber, 3 hotnumbers, 4 unpopular.
    Standard errors via 20 batch means.
    """
    n_batches = 20
    per = n_samples // n_batches
    for s in range(5):
        state = np.zeros(1, dtype=np.uint64)
        state[0] = np.uint64(seed + 7919 * (s + 1))
        deck = np.empty(N_BALLS, dtype=np.int64)
        balls = np.empty(PICK, dtype=np.int64)
        batch = np.empty(n_batches, dtype=np.float64)
        if s == 2:
            # degenerate: fixed combination, exact value
            pp = combo_popularity(SAME_NUMBERS, wA, wB, wC, ZA, ZB, ZC)
            v = expected_share(CROWD_PER_DRAW * pp)
            for b in range(n_batches):
                batch[b] = v
        elif s == 3:
            # hot-numbers: run the chain (50-draw warmup, then sample)
            counts = np.zeros(N_BALLS, dtype=np.int64)
            hist = np.empty((HOT_WINDOW, PICK), dtype=np.int64)
            for t in range(HOT_WINDOW):
                draw_uniform_combo(state, deck, balls)
                for j in range(PICK):
                    hist[t, j] = balls[j]
                    counts[balls[j] - 1] += 1
            pos = 0
            hot = np.empty(PICK, dtype=np.int64)
            for b in range(n_batches):
                acc = 0.0
                for _ in range(per):
                    draw_uniform_combo(state, deck, balls)
                    for j in range(PICK):  # slide window
                        counts[hist[pos, j] - 1] -= 1
                        hist[pos, j] = balls[j]
                        counts[balls[j] - 1] += 1
                    pos += 1
                    if pos == HOT_WINDOW:
                        pos = 0
                    hottest6(counts, hot)
                    pp = combo_popularity(hot, wA, wB, wC, ZA, ZB, ZC)
                    acc += expected_share(CROWD_PER_DRAW * pp)
                batch[b] = acc / per
        else:
            for b in range(n_batches):
                acc = 0.0
                for _ in range(per):
                    if s == 0:
                        draw_uniform_combo(state, deck, balls)
                    elif s == 1:
                        draw_weighted_combo(state, aliasB_prob, aliasB_alias,
                                            balls)
                    else:  # s == 4, unpopular
                        draw_uniform_high_combo(state, deck, balls)
                    pp = combo_popularity(balls, wA, wB, wC, ZA, ZB, ZC)
                    acc += expected_share(CROWD_PER_DRAW * pp)
                batch[b] = acc / per
        m = 0.0
        for b in range(n_batches):
            m += batch[b]
        m /= n_batches
        v = 0.0
        for b in range(n_batches):
            d = batch[b] - m
            v += d * d
        v /= (n_batches - 1)
        means[s] = m
        ses[s] = math.sqrt(v / n_batches)

# ---------------------------------------------------------------- phase C: direct 100M-ticket simulation (chunked)
@njit
def direct_sim_chunk(chunk_seed, n_draws, wA, wB, wC, ZA, ZB, ZC,
                     aliasB_prob, aliasB_alias, out):
    """Simulate n_draws; 1 ticket per strategy per draw.

    out (float64, len>=30) accumulators, all starting at 0 on entry:
      per strategy s (0..4): out[6*s+0]=tickets, [1]=cost, [2]=m3, [3]=m4,
        [4]=m5, [5]=jackpot_hits, [6]=jackpot_won  -> stride 7, base s*7
      then out[35]=crowd_jackpot_wins, out[36]=total_jackpot_events,
      out[37]=max_co_winners, out[38]=max_jackpot_paid
    """
    state = np.zeros(1, dtype=np.uint64)
    state[0] = np.uint64(chunk_seed)
    deck = np.empty(N_BALLS, dtype=np.int64)
    balls = np.empty(PICK, dtype=np.int64)
    tb = np.empty(PICK, dtype=np.int64)
    counts = np.zeros(N_BALLS, dtype=np.int64)
    hist = np.empty((HOT_WINDOW, PICK), dtype=np.int64)
    # warmup for hot-numbers history
    for t in range(HOT_WINDOW):
        draw_uniform_combo(state, deck, balls)
        for j in range(PICK):
            hist[t, j] = balls[j]
            counts[balls[j] - 1] += 1
    pos = 0
    hot = np.empty(PICK, dtype=np.int64)
    jackpot = JACKPOT_SEED
    tickets_per_draw = float(CROWD_PER_DRAW + 5)
    inc = JACKPOT_PER_TICKET * tickets_per_draw
    for _ in range(n_draws):
        # hot-numbers bot plays the hottest 6 of PAST draws only --
        # computing it after the draw would let it peek at today's balls
        hottest6(counts, hot)
        draw_uniform_combo(state, deck, balls)
        wmask = combo_mask(balls)
        # crowd jackpot winners via Poisson
        pp = combo_popularity(balls, wA, wB, wC, ZA, ZB, ZC)
        c_crowd = poisson_sample(state, CROWD_PER_DRAW * pp)
        # strategy tickets
        strat_hits = 0
        hit_idx = np.empty(5, dtype=np.int64)
        m3 = np.zeros(5, dtype=np.int64)
        m4 = np.zeros(5, dtype=np.int64)
        m5 = np.zeros(5, dtype=np.int64)
        for s in range(5):
            if s == 0:
                draw_uniform_combo(state, deck, tb)
            elif s == 1:
                draw_weighted_combo(state, aliasB_prob, aliasB_alias, tb)
            elif s == 2:
                for j in range(PICK):
                    tb[j] = SAME_NUMBERS[j]
            elif s == 3:
                for j in range(PICK):
                    tb[j] = hot[j]
            else:
                draw_uniform_high_combo(state, deck, tb)
            mc = popcnt64(combo_mask(tb) & wmask)
            if mc == 6:
                hit_idx[strat_hits] = s
                strat_hits += 1
            elif mc == 5:
                m5[s] = 1
            elif mc == 4:
                m4[s] = 1
            elif mc == 3:
                m3[s] = 1
        n_w = c_crowd + strat_hits
        if n_w > 0:
            payout, jackpot = jackpot_step_payout(jackpot, n_w)
            for i in range(strat_hits):
                s = hit_idx[i]
                out[s * 7 + 5] += 1
                out[s * 7 + 6] += payout
            out[35] += c_crowd
            out[36] += 1
            if c_crowd > out[37]:
                out[37] = c_crowd
            if payout > out[38]:
                out[38] = payout
        else:
            jackpot += inc
        # slide hot history with this draw AFTER evaluation (no peeking)
        for j in range(PICK):
            counts[hist[pos, j] - 1] -= 1
            hist[pos, j] = balls[j]
            counts[balls[j] - 1] += 1
        pos += 1
        if pos == HOT_WINDOW:
            pos = 0
        for s in range(5):
            out[s * 7 + 0] += 1
            out[s * 7 + 1] += TICKET_PRICE
            out[s * 7 + 2] += m3[s]
            out[s * 7 + 3] += m4[s]
            out[s * 7 + 4] += m5[s]
    # lower-tier winnings added by caller (exact prizes)
