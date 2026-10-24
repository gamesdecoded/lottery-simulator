"""Production run: lottery 6/49 strategy comparison.

Phase A: crowd jackpot rollover process -> E[jackpot at win].
Phase B: Rao-Blackwellized expected jackpot share per strategy (tight CIs).
Phase C: direct 100M-ticket simulation (the title number), chunked with
         per-chunk seeds -> resumable after VM restarts; re-running only
         executes missing chunks (deterministic).

Writes results.json on completion.
"""
import json
import math
import os
import time

import numpy as np

import engine as E

OUT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(OUT, "results.json.checkpoint.npz")
RESULTS = os.path.join(OUT, "results.json")

BASE_SEED = 20261023
N_DRAWS_DIRECT = 20_000_000      # 1 ticket/strategy/draw -> 100M tickets
CHUNK_DRAWS = 1_000_000
N_CHUNKS = N_DRAWS_DIRECT // CHUNK_DRAWS
N_DRAWS_JACKPOT = 300_000        # phase A
N_SHARE_SAMPLES = 2_000_000      # phase B per strategy


def main():
    t0 = time.time()
    ZA = E.elementary_symmetric_6(E.W_A)
    ZB = E.elementary_symmetric_6(E.W_B)
    ZC = E.elementary_symmetric_6(E.W_C)
    probB, aliasB = E.build_alias(E.W_B)
    W = (E.W_A, E.W_B, E.W_C)

    # ---- Phase A: crowd jackpot process ----
    print("Phase A: crowd jackpot process...", flush=True)
    win_j = np.empty(N_DRAWS_JACKPOT, dtype=np.float64)
    n_wins, mean_j, max_j = E.crowd_jackpot_process(
        np.uint64(BASE_SEED + 1), N_DRAWS_JACKPOT,
        E.W_A, E.W_B, E.W_C, ZA, ZB, ZC, probB, aliasB, win_j)
    win_j = win_j[:n_wins]
    se_j = float(win_j.std(ddof=1) / math.sqrt(n_wins))
    print(f"  wins={n_wins} win_rate={n_wins/N_DRAWS_JACKPOT:.3f} "
          f"E[jackpot]=${mean_j:,.0f} +- ${se_j:,.0f} max=${max_j:,.0f}",
          flush=True)

    # ---- Phase B: expected shares ----
    print("Phase B: expected jackpot share per strategy...", flush=True)
    means = np.empty(5, dtype=np.float64)
    ses = np.empty(5, dtype=np.float64)
    E.estimate_shares(np.uint64(BASE_SEED + 2), N_SHARE_SAMPLES,
                      E.W_A, E.W_B, E.W_C, ZA, ZB, ZC,
                      probB, aliasB, means, ses)
    for s, name in enumerate(E.STRATEGIES):
        print(f"  {name:12s} E[share]={means[s]:.4f} +- {1.96*ses[s]:.5f}",
              flush=True)

    # ---- Phase C: direct 100M-ticket sim, chunked ----
    print("Phase C: direct simulation (100M tickets)...", flush=True)
    acc = np.zeros(40, dtype=np.float64)
    done = np.zeros(N_CHUNKS, dtype=np.bool_)
    if os.path.exists(CKPT):
        z = np.load(CKPT)
        acc = z["acc"]
        done = z["done"]
        print(f"  resumed: {int(done.sum())}/{N_CHUNKS} chunks already done",
              flush=True)
    for c in range(N_CHUNKS):
        if done[c]:
            continue
        chunk = np.zeros(40, dtype=np.float64)
        E.direct_sim_chunk(np.uint64(BASE_SEED + 3 + c * 1000003), CHUNK_DRAWS,
                           E.W_A, E.W_B, E.W_C, ZA, ZB, ZC,
                           probB, aliasB, chunk)
        acc += chunk
        done[c] = True
        np.savez(CKPT, acc=acc, done=done)
        el = time.time() - t0
        print(f"  chunk {c+1}/{N_CHUNKS} done ({el:.0f}s elapsed)", flush=True)

    # ---- assemble results ----
    # NOTE: acc layout stride 7 per strategy: 0 tickets,1 cost,2 m3,3 m4,4 m5,5 hits,6 won
    tickets = acc[0::7][:5]
    cost = acc[1::7][:5]
    m3 = acc[2::7][:5]
    m4 = acc[3::7][:5]
    m5 = acc[4::7][:5]
    hits = acc[5::7][:5]
    won = acc[6::7][:5]
    lower = m3 * E.PRIZE_MATCH3 + m4 * E.PRIZE_MATCH4 + m5 * E.PRIZE_MATCH5
    roi_direct = (lower + won - cost) / cost

    # precision ROI: L + E[J]*share/N_COMBOS per ticket, vs $3 cost
    roi_p, lo, hi = [], [], []
    for s in range(5):
        ev = E.LOWER_TIER_PER_TICKET + mean_j * means[s] / E.N_COMBOS
        var = ((mean_j * ses[s]) ** 2 + (means[s] * se_j) ** 2) / E.N_COMBOS ** 2
        se = math.sqrt(var) / E.TICKET_PRICE
        r = ev / E.TICKET_PRICE - 1.0
        roi_p.append(r)
        lo.append(r - 1.96 * se)
        hi.append(r + 1.96 * se)

    order = sorted(range(5), key=lambda s: -roi_p[s])
    overlap = {}
    for a, b in zip(order, order[1:]):
        overlap[f"{E.STRATEGIES[a]}_vs_{E.STRATEGIES[b]}"] = bool(lo[a] < hi[b])

    total_tickets = int(tickets.sum())
    res = {
        "meta": {
            "title_uses_total_tickets": total_tickets,
            "draws": N_DRAWS_DIRECT,
            "tickets_per_strategy_per_draw": 1,
            "crowd_tickets_per_draw": E.CROWD_PER_DRAW,
            "ticket_price": E.TICKET_PRICE,
            "jackpot_seed": E.JACKPOT_SEED,
            "jackpot_per_ticket": E.JACKPOT_PER_TICKET,
            "prizes": {"match5": E.PRIZE_MATCH5, "match4": E.PRIZE_MATCH4,
                       "match3": E.PRIZE_MATCH3},
            "seed": BASE_SEED,
        },
        "crowd_jackpot": {
            "draws": N_DRAWS_JACKPOT,
            "wins": int(n_wins),
            "win_rate": n_wins / N_DRAWS_JACKPOT,
            "mean_jackpot_at_win": float(mean_j),
            "se_jackpot_at_win": float(se_j),
            "max_jackpot": float(max_j),
        },
        "strategies": {},
        "ranking": [E.STRATEGIES[s] for s in order],
        "ci_overlap_adjacent": overlap,
        "validation": "validate.py: 9 gates passed before production",
    }
    for s, name in enumerate(E.STRATEGIES):
        res["strategies"][name] = {
            "tickets": int(tickets[s]),
            "cost": float(cost[s]),
            "match3": int(m3[s]),
            "match4": int(m4[s]),
            "match5": int(m5[s]),
            "jackpot_hits": int(hits[s]),
            "jackpot_won": float(won[s]),
            "expected_share": float(means[s]),
            "expected_share_se": float(ses[s]),
            "roi_direct": float(roi_direct[s]),
            "roi_precision": float(roi_p[s]),
            "roi_precision_ci95": [float(lo[s]), float(hi[s])],
        }
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=1)

    # ---- consistency checks (informational) ----
    print("\nConsistency checks:", flush=True)
    for s, name in enumerate(E.STRATEGIES):
        exp_hits = tickets[s] / E.N_COMBOS
        z = (hits[s] - exp_hits) / math.sqrt(exp_hits) if exp_hits else 0
        print(f"  {name:12s} jackpot hits={int(hits[s])} "
              f"(expected {exp_hits:.2f}, z={z:+.1f}) "
              f"roi_direct={roi_direct[s]:+.1%} roi_precision={roi_p[s]:+.2%} "
              f"[{lo[s]:+.2%}, {hi[s]:+.2%}]", flush=True)
    print(f"\nTotal strategy tickets: {total_tickets:,}")
    print(f"Ranking: {' > '.join(res['ranking'])}")
    print(f"Adjacent CI overlaps: {overlap}")
    print(f"Wrote {RESULTS} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
