"""Head-to-head duplicate matches vs a bot in another folder, in-process.

    python tools/h2h.py /path/to/other/repo --seeds 1 2 3 --params '{"bet_thresh": {...}}'

The other bot's modules (which may share names with ours, e.g. strategy.py)
are loaded in isolation so both bots run in one process.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game import AgentConfig, Game, button_for_deal, make_deal, default_action, Action  # noqa


def load_isolated(folder, name):
    """Import folder/agent.py with folder's own module namespace."""
    names = {f[:-3] for f in os.listdir(folder) if f.endswith(".py")}
    # packages (sub-folders with __init__.py) are isolated too, so no module
    # state leaks between halves (the tournament uses a fresh process per half)
    names |= {f for f in os.listdir(folder)
              if os.path.isfile(os.path.join(folder, f, "__init__.py"))}

    def ours(n):
        return n.split(".")[0] in names

    saved = {n: sys.modules.pop(n) for n in list(sys.modules) if ours(n)}
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location(name, os.path.join(folder, "agent.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        sys.path.remove(folder)
        for n in list(sys.modules):
            if ours(n):
                sys.modules.pop(n)
        sys.modules.update(saved)
    return m


def load_mine(params):
    m = load_isolated(ROOT, f"mine{time.time_ns()}")
    if params:
        strat = m.strategy
        for k, v in params.items():
            strat.DEFAULT_PARAMS[k] = {int(a): b for a, b in v.items()} if isinstance(v, dict) else v
    return m


STAT_KEYS = ("faced", "fold", "raise_vs", "bets_unopp", "chances_unopp", "sd", "sd_won")


def play_half(bots, seed, deals, seat_of_a, stats):
    mems = [{}, {}]
    cfg = AgentConfig(num_hands=deals)
    for b, m in zip(bots, mems):
        b.new_half(cfg, m)
    seat_bot = {seat_of_a: 0, 1 - seat_of_a: 1}
    nets, times, errs = [], [0.0, 0.0], [0, 0]
    for i in range(deals):
        g = Game(make_deal(seed, i), button_for_deal(i), i)
        while not g.is_terminal:
            seat = g.to_act
            bi = seat_bot[seat]
            obs = g.observation(seat)
            t = time.perf_counter()
            try:
                a = Action(int(bots[bi].choose_action(obs, mems[bi])))
                if a not in obs.legal_actions:
                    raise ValueError("illegal")
            except Exception:
                errs[bi] += 1
                a = default_action(obs.to_call)
            times[bi] += time.perf_counter() - t
            if obs.street > 0:
                s = stats[bi]
                if obs.to_call > 0:
                    s["faced"] += 1
                    s["fold"] += a == Action.FOLD
                    s["raise_vs"] += a in (Action.RAISE_HALF, Action.RAISE_POT)
                else:
                    s["chances_unopp"] += 1
                    s["bets_unopp"] += a in (Action.RAISE_HALF, Action.RAISE_POT)
            g.apply(a)
        res = g.result()
        for seat in (0, 1):
            bi = seat_bot[seat]
            bots[bi].on_hand_end(g.hand_result(seat), mems[bi])
            if res.went_to_showdown:
                stats[bi]["sd"] += 1
                stats[bi]["sd_won"] += res.nets[seat] > 0
        nets.append(res.nets[seat_of_a])
    return nets, times, errs


def match(other, seed, deals, params):
    stats = [{k: 0 for k in STAT_KEYS} for _ in range(2)]
    A, B = load_mine(params), load_isolated(other, f"other{time.time_ns()}")
    n1, t1, e1 = play_half((A, B), seed, deals, 0, stats)
    A, B = load_mine(params), load_isolated(other, f"other{time.time_ns()}")
    n2, t2, e2 = play_half((A, B), seed, deals, 1, stats)
    per = [(x + y) / 2 for x, y in zip(n1, n2)]
    mean = sum(per) / len(per)
    se = math.sqrt(sum((v - mean) ** 2 for v in per) / (len(per) - 1) / len(per))
    return mean, se, stats, (t1, t2), (e1, e2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("other")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    params = json.loads(args.params)
    ms = []
    for s in args.seeds:
        mean, se, st, tm, er = match(args.other, s, args.deals, params)
        ms.append(mean)
        v = "WIN" if mean - 1.96 * se > 0 else ("LOSS" if mean + 1.96 * se < 0 else "CHOP")
        line = f"seed {s}: {mean:+.3f} ± {1.96*se:.3f}  {v}  time mine {tm[0][0]:.1f}/{tm[1][0]:.1f}s  err {er}"
        if args.stats:
            for nm, x in zip(("mine", "theirs"), st):
                line += (f"\n   {nm}: foldVsBet {x['fold']/max(1,x['faced']):.2f} raiseVsBet {x['raise_vs']/max(1,x['faced']):.2f}"
                         f" betWhenChecked {x['bets_unopp']/max(1,x['chances_unopp']):.2f} sdWin {x['sd_won']/max(1,x['sd']):.2f} sd {x['sd']}")
        print(line, flush=True)
    print(f"{args.params} AVG {sum(ms)/len(ms):+.3f}", flush=True)
