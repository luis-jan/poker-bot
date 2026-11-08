# The Larpinator — how it works

Files added/changed (everything else is the untouched starter kit):

| File | Purpose |
|---|---|
| `agent.py` | Crash-proof wrappers that call `strategy.py` (any exception → safe legal action). Set `PARTICIPANT` to your team name. |
| `strategy.py` | The decision logic and all tunable parameters (`DEFAULT_PARAMS`). |
| `fasteval.py` | Vectorised numpy Omaha evaluator: scores 100 combos × hundreds of runouts in <1 ms. Verified identical to `omaha_score`. |
| `preflop_equity.npy` | Equity of all 376,992 hole hands vs a random hand (1.5 MB). Rebuild with `python build_preflop.py`. |
| `tools/sim.py` | Fast in-process duplicate match (no HTTP): `python tools/sim.py agent.py baseline-bots/baseline_random.py` |
| `tools/opponents/` | Tougher sparring bots (equity+pot-odds "MC", tight "nit", loose-aggressive "lag"). Run with `PYTHONPATH=.` |
| `tools/tune.py` | Parameter sweeps: `python tools/tune.py '{"pf_open":0.55}'` |

## Strategy (v2: expected-value model)
1. **Weighted Monte Carlo.** For each decision, ~420 opponent hands + runouts are sampled with the vectorised evaluator. Each sampled hand is weighted by how likely *this* opponent would have taken this hand's actions with it (bets = strongest part of their range, calls = the band below, checks = everything below their betting hands, with some slowplays).
2. **Opponent model.** Action frequencies are learned per situation (first bet, facing a bet, facing a re-raise, preflop opens) and per street, shrunk towards priors. An "honesty" score (alpha) is learned from showdowns: how strong their betting hands were vs. what an honest bettor would show. Bluffy opponents get called down; honest ones get respected.
3. **EV decision.** Fold = 0; call = equity × final pot − cost (with an out-of-position realisation discount); each raise = fold equity × pot + (1 − fold equity) × (equity vs. continuing range × final pot − cost). Tiny noise keeps it unpredictable.
4. **Defense floor.** EV-based bluffers profit when we fold to bets more often than their bet size allows (to_call / pot). The bot tracks its own fold rate, and when it is too high it calls hands that are only slightly -EV (`mdf_tol`). This was the biggest single gain against the strongest opponents.
5. **Adapting to honest and trapping opponents.** The defense floor only switches on against opponents whose shown-down bets are often weak (bluffers); against honest bettors it stays off. "They checked, then faced our bet" is tracked separately (check-raise rate), and thin bets into a check need an extra edge against opponents that often raise our bets. When we do bet, the size is sometimes swapped to the other +EV size so it doesn't reveal strength.
6. **Cautious calling on flop and turn.** Calls of a bet count only part of our equity (70-78% out of position, 78-86% in position), because future bets from value-heavy opponents make weak calls lose. This fixed the outliers matchup and also improved results vs Royal Flush, the teammate bot and Blueprint.
7. **Time bank.** Targets ≈4.5 s per half and automatically uses fewer samples if it falls behind (verified with a 6× heavier workload: still within budget, no errors).

## Local results (1,000-deal duplicate matches)
| Opponent | bb/deal | Verdict |
|---|---|---|
| baseline_random | +11.4 | WIN |
| baseline_calling_station | +12.9 | WIN |
| baseline_always_pot | +8.1 | WIN |
| baseline_always_half | +12.1 | WIN |
| MC equity + pot odds bot | +5.3 | WIN |
| tight "nit" | +2.0 | WIN |
| loose-aggressive bluffer | +5.7 | WIN |
| teammate's RangeReader (8 fresh seeds) | +1.19 avg | 2 wins, 6 chops, 0 losses |
| Royal Flush (public competitor, 8 fresh seeds) | +0.63 avg | 1 win, 7 chops, 0 losses |
| TAMU-Datathon-CYH (16 fresh seeds) | +0.05 avg (was -0.78) | 1 win, 15 chops, 0 losses |
| outliers v6 (6 fresh seeds) | +0.08 avg (was -0.64) | even |

## Tools for head-to-head work
- `tools/h2h.py OTHER_REPO --seeds 1 2 3 --stats` plays our bot vs a bot in another folder, in-process, even if its files share names with ours.
- `tools/analyze.py OTHER_REPO 1 2` shows where chips are won/lost (who folds on which street, showdowns).
- `tools/scout.py URL...` clones public bot repos and plays them.
- `tools/batch.sh` / `tools/batch2.sh` / `tools/batch3.sh` run parameter sweeps (`--params '{"alpha_scale":1.25}'`).
