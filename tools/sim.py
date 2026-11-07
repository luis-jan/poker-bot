"""Fast in-process duplicate match between two bot files (no HTTP).

    python tools/sim.py agent.py baseline-bots/baseline_random.py --deals 1000 --seed 1
"""
import argparse
import importlib.util
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game import AgentConfig, Game, button_for_deal, make_deal, default_action  # noqa

_n = [0]


def load(path):
    _n[0] += 1
    spec = importlib.util.spec_from_file_location(f"bot{_n[0]}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def play_half(bots, seed, deals, seat_of_a):
    """bots = (A, B). Returns list of A's chip net per deal, plus A's think time."""
    mems = [{}, {}]
    cfg = AgentConfig(num_hands=deals)
    for b, m in zip(bots, mems):
        b.new_half(cfg, m)
    seat_bot = {seat_of_a: 0, 1 - seat_of_a: 1}
    nets, times, errors = [], [0.0, 0.0], [0, 0]
    for i in range(deals):
        g = Game(make_deal(seed, i), button_for_deal(i), i)
        while not g.is_terminal:
            seat = g.to_act
            bi = seat_bot[seat]
            obs = g.observation(seat)
            t = time.perf_counter()
            try:
                a = bots[bi].choose_action(obs, mems[bi])
                if a not in obs.legal_actions:
                    raise ValueError(f"illegal {a}")
            except Exception as e:
                errors[bi] += 1
                if errors[bi] <= 2:
                    import traceback; traceback.print_exc()
                a = default_action(obs.to_call)
            times[bi] += time.perf_counter() - t
            g.apply(a)
        for seat in (0, 1):
            bi = seat_bot[seat]
            bots[bi].on_hand_end(g.hand_result(seat), mems[bi])
        nets.append(g.result().nets[seat_of_a])
    return nets, times, errors


def match(path_a, path_b, deals=1000, seed=1, quiet=False):
    A, B = load(path_a), load(path_b)
    n1, t1, e1 = play_half((A, B), seed, deals, 0)
    A, B = load(path_a), load(path_b)
    n2, t2, e2 = play_half((A, B), seed, deals, 1)
    # per-deal result in bb = (net_half1 + net_half2) / big_blind
    per = [(x + y) / 2 for x, y in zip(n1, n2)]
    mean = sum(per) / len(per)
    var = sum((v - mean) ** 2 for v in per) / (len(per) - 1)
    se = math.sqrt(var / len(per))
    verdict = "WIN" if mean - 1.96 * se > 0 else ("LOSS" if mean + 1.96 * se < 0 else "CHOP")
    if not quiet:
        print(f"{os.path.basename(path_a)} vs {os.path.basename(path_b)}: "
              f"{mean:+.3f} bb/deal  CI [{mean-1.96*se:+.3f}, {mean+1.96*se:+.3f}]  {verdict}  "
              f"timeA {t1[0]:.2f}/{t2[0]:.2f}s errA {e1[0]+e2[0]} errB {e1[1]+e2[1]}")
    return mean, se, verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b", nargs="+")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    for b in args.b:
        match(args.a, b, args.deals, args.seed)
