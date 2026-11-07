# RangeEquityBot — how it works

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

## Strategy
1. **Opponent range.** Each opponent raise narrows their range to the top fraction of hands matching how often *this* opponent raises (learned during the half); calls of our bets narrow it more against opponents who fold often. A maniac's raises barely narrow anything, so we call them down wide; a nit's raises are respected.
2. **Equity.** Monte Carlo vs that range (sampled opponent hands ranked by current made hand + preflop strength), 320 samples per decision. Preflop vs an unraised opponent uses the lookup table directly.
3. **Action.** Facing a bet: raise above a strength threshold, call if equity beats pot odds plus a small margin (bigger out of position / earlier streets), else fold. Checked to: value-bet above a threshold (pot-size when very strong), and bluff only if the opponent has shown they fold to bets often.
4. **Time bank.** Thinking time is tracked; samples drop automatically if the bank runs low, with a table-only emergency mode. Typical use ≈1.8 s of 8 s per half.

## Local results (1,000-deal duplicate matches)
| Opponent | bb/deal | Verdict |
|---|---|---|
| baseline_random | +4.5 | WIN |
| baseline_calling_station | +9.0 | WIN |
| baseline_always_pot | +8.8 | WIN |
| baseline_always_half | +12.9 | WIN |
| MC equity + pot odds bot | +2.7 | WIN |
| tight "nit" | +1.2 | WIN |
| loose-aggressive bluffer | +3.8 | WIN |
