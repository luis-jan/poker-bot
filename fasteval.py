"""
fasteval.py - vectorised (numpy) Short Deck PLO5 hand evaluation and equity.

Idea: a five-card score depends only on the multiset of ranks plus whether all
five cards share a suit. Encode a rank multiset additively as sum(5**rank)
(each rank appears at most 4 times, so base 5 is collision-free). Because the
key is additive, key(2 hole cards + 3 board cards) = key(pair) + key(triple),
which lets us score all 100 Omaha combinations for thousands of deals with a
handful of numpy operations.
"""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement
import os

import numpy as np

from evaluator import _score_ranks

_POW5 = np.array([5 ** r for r in range(9)], dtype=np.int64)
_KEYSPACE = 5 ** 9

# Per-card rank key and suit.
CARD_KEY = np.array([5 ** (c >> 2) for c in range(36)], dtype=np.int32)
CARD_SUIT = np.array([c & 3 for c in range(36)], dtype=np.int8)

PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.int64)      # (10, 2)
TRIPLES5 = np.array(list(combinations(range(5), 3)), dtype=np.int64)   # (10, 3)


def _build_tables():
    plain = np.zeros(_KEYSPACE, dtype=np.int32)
    flush = np.zeros(_KEYSPACE, dtype=np.int32)
    for ranks in combinations_with_replacement(range(9), 5):
        if max(ranks.count(r) for r in set(ranks)) > 4:
            continue
        desc = tuple(sorted(ranks, reverse=True))
        key = sum(5 ** r for r in ranks)
        plain[key] = _score_ranks(desc, False)
        if len(set(ranks)) == 5:
            flush[key] = _score_ranks(desc, True)
    return plain, flush


PLAIN, FLUSH = _build_tables()


def best_scores(hole: np.ndarray, board: np.ndarray) -> np.ndarray:
    """hole: (N,5) ints, board: (N,5) ints -> (N,) best Omaha score."""
    hk = CARD_KEY[hole]
    hs = CARD_SUIT[hole].astype(np.int16)
    bk = CARD_KEY[board]
    bs = CARD_SUIT[board].astype(np.int16)
    pk = hk[:, PAIRS[:, 0]] + hk[:, PAIRS[:, 1]]                       # (N,10)
    ps = np.where(hs[:, PAIRS[:, 0]] == hs[:, PAIRS[:, 1]], hs[:, PAIRS[:, 0]], -1)
    tk = bk[:, TRIPLES5[:, 0]] + bk[:, TRIPLES5[:, 1]] + bk[:, TRIPLES5[:, 2]]
    t0 = bs[:, TRIPLES5[:, 0]]
    ts = np.where((t0 == bs[:, TRIPLES5[:, 1]]) & (t0 == bs[:, TRIPLES5[:, 2]]), t0, -2)
    key = pk[:, :, None] + tk[:, None, :]                                # (N,10,10)
    sc = PLAIN[key]
    fl = ps[:, :, None] == ts[:, None, :]
    if fl.any():
        sc = np.where(fl, np.maximum(sc, FLUSH[key]), sc)
    return sc.reshape(len(hole), -1).max(axis=1)


def current_scores(hole: np.ndarray, board_known: np.ndarray) -> np.ndarray:
    """Made-hand strength using 2 hole + 3 of the k known board cards (k=3..5).
    hole: (N,5); board_known: (k,) shared by all rows."""
    k = len(board_known)
    trip = np.array(list(combinations(range(k), 3)), dtype=np.int64)
    bk = CARD_KEY[board_known]
    bs = CARD_SUIT[board_known].astype(np.int16)
    tk = bk[trip].sum(axis=1)                                            # (T,)
    t0 = bs[trip[:, 0]]
    ts = np.where((t0 == bs[trip[:, 1]]) & (t0 == bs[trip[:, 2]]), t0, -2)
    hk = CARD_KEY[hole]
    hs = CARD_SUIT[hole].astype(np.int16)
    pk = hk[:, PAIRS[:, 0]] + hk[:, PAIRS[:, 1]]
    ps = np.where(hs[:, PAIRS[:, 0]] == hs[:, PAIRS[:, 1]], hs[:, PAIRS[:, 0]], -1)
    key = pk[:, :, None] + tk[None, None, :]
    sc = PLAIN[key]
    fl = ps[:, :, None] == ts[None, None, :]
    if fl.any():
        sc = np.where(fl, np.maximum(sc, FLUSH[key]), sc)
    return sc.reshape(len(hole), -1).max(axis=1)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_rest(rng: np.random.Generator, dead: list[int], n: int, k: int) -> np.ndarray:
    """n rows of k distinct cards drawn from the deck minus `dead`."""
    live = np.setdiff1d(np.arange(36), np.array(dead, dtype=np.int64))
    # Random permutation per row via argsort of random keys (fast for 36 cards).
    r = rng.random((n, len(live)))
    idx = np.argpartition(r, k, axis=1)[:, :k]
    return live[idx]


# Preflop strength table: indexed by the combinatorial index of a sorted
# 5-card hole hand. Built offline by build_preflop.py (equity vs random hand).
_COMB = np.zeros((37, 6), dtype=np.int64)
for _n in range(37):
    for _k in range(6):
        from math import comb as _c
        _COMB[_n, _k] = _c(_n, _k)


def hole_index(hole: np.ndarray) -> np.ndarray:
    """(N,5) hole cards -> (N,) combinatorial index in [0, C(36,5))."""
    h = np.sort(hole, axis=1)
    return (_COMB[h[:, 0], 1] + _COMB[h[:, 1], 2] + _COMB[h[:, 2], 3]
            + _COMB[h[:, 3], 4] + _COMB[h[:, 4], 5])


_PF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preflop_equity.npy")
PREFLOP = np.load(_PF_PATH) if os.path.exists(_PF_PATH) else None
