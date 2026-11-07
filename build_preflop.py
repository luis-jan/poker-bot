"""Offline: build preflop_equity.npy = equity of every 5-card hole hand vs a
random hand, indexed by fasteval.hole_index. Noise is reduced by averaging
over suit-isomorphic hands. Run once: python build_preflop.py"""
import time
from itertools import combinations
from collections import defaultdict

import numpy as np

import fasteval as F

TRIALS = 120
rng = np.random.default_rng(12345)
hands = np.array(list(combinations(range(36), 5)), dtype=np.int64)
idx = F.hole_index(hands)
assert (np.sort(idx) == np.arange(len(hands))).all()
eq = np.zeros(len(hands))
t = time.time()
B = 4000
for start in range(0, len(hands), B):
    h = hands[start:start + B]
    n = len(h)
    tot = np.zeros(n)
    for _ in range(TRIALS):
        # deal 10 other cards per row: random keys, push own cards to the end
        r = rng.random((n, 36))
        np.put_along_axis(r, h, 2.0, axis=1)
        other = np.argpartition(r, 10, axis=1)[:, :10]
        a = F.best_scores(h, other[:, 5:])
        b = F.best_scores(other[:, :5], other[:, 5:])
        tot += (a > b) + 0.5 * (a == b)
    eq[start:start + n] = tot / TRIALS
    if start % 40000 == 0:
        print(start, time.time() - t, flush=True)

# average over suit isomorphism classes
def canon(hand):
    by = [[] for _ in range(4)]
    for c in hand:
        by[c & 3].append(c >> 2)
    return tuple(sorted((tuple(sorted(x)) for x in by), key=lambda x: (len(x), x)))

groups = defaultdict(list)
for i, hnd in enumerate(hands):
    groups[canon(hnd)].append(i)
out = np.zeros(len(hands), dtype=np.float32)
for g in groups.values():
    m = eq[g].mean()
    out[idx[g]] = m
print("classes", len(groups), "min/mean/max", out.min(), out.mean(), out.max())
np.save("preflop_equity.npy", out)
