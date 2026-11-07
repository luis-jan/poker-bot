"""
strategy.py - the decision logic behind agent.py (v2: expected-value model).

For every decision:
  1. Sample opponent hands + board runouts (vectorised numpy, fasteval.py).
  2. Weight each sampled opponent hand by how likely THIS opponent would have
     taken the actions they took this hand with it. Likelihoods come from the
     opponent's observed action frequencies, softened by an "informativeness"
     alpha learned from showdowns (does their aggression really mean
     strength?). Bluffs and draws therefore keep some weight.
  3. Compute the EV of folding, calling, and each raise size. Raises include
     fold equity from the opponent's observed fold-to-bet rate, and the equity
     against the part of their range that continues.
  4. Pick the best EV (tiny noise to stay unpredictable).

A time manager scales the sample count to the 8 s bank; agent.py wraps all of
this in try/except so the bot can never crash or return an illegal action.
"""

from __future__ import annotations

import time

import numpy as np

import fasteval as F
from game import Action, Street

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

DEFAULT_PARAMS = {
    "n_samples": 420,
    "time_target": 4.5,          # seconds of thinking we aim to use per half
    # equity realisation (out of position / in position) per street
    "real_oop": (0.94, 0.95, 0.98, 1.0),
    "real_ip": (1.0, 1.0, 1.0, 1.0),
    "prior_n": 10.0,             # pseudo-counts behind each frequency
    "alpha_prior": 0.65,         # prior informativeness of opponent actions
    "alpha_prior_n": 6.0,
    "alpha_min": 0.15,
    "alpha_scale": 1.5,          # trust in observed aggression (tuned)
    "half_extra_cont": 0.07,     # opponents continue more vs half-pot bets
    "reraise_tax": 0.06,         # small penalty for being re-raised off equity
    "slowplay_w": 0.25,          # weight a passive action keeps for strong hands
    "noise": 0.012,              # EV noise as a fraction of the pot
    "pf_blend": 0.25,            # preflop strength share in postflop ranking
    # aggression on later streets is discounted: showdown samples overstate
    # how honest opponents are (we never see the bluffs we folded to), and
    # repeated barrels shouldn't compound into "only the nuts"
    "late_alpha": (1.0, 1.0, 0.85, 0.85),   # alpha multiplier per street
    "repeat_decay": 0.8,          # each extra opp raise in a hand counts less
    "street_n": 8.0,              # shrinkage of per-street frequencies
}

# priors for opponent frequencies
PRIORS = {
    # (street_group, situation) -> (fold, call, raise) for facing a bet;
    # for unopened: (check, -, bet)
    "face0": (0.25, 0.50, 0.25),
    "face1": (0.40, 0.45, 0.15),
    "unop0": (0.55, 0.0, 0.45),   # big blind after a limp: check / raise
    "sbopen": (0.10, 0.55, 0.35),  # small blind first in: fold / limp / raise
    "unop1": (0.55, 0.0, 0.45),
    # facing a raise after having bet/raised themselves on this street
    "reface0": (0.15, 0.60, 0.25),
    "reface1": (0.20, 0.60, 0.20),
}

_PF_RANK = None


def _pf_rank():
    """Percentile (0 = worst, 1 = best) of every hole hand by preflop equity."""
    global _PF_RANK
    if _PF_RANK is None:
        order = np.argsort(F.PREFLOP, kind="stable")
        r = np.empty(len(order), dtype=np.float32)
        r[order] = (np.arange(len(order)) + 0.5) / len(order)
        _PF_RANK = r
    return _PF_RANK


# ---------------------------------------------------------------------------
# Memory / opponent model
# ---------------------------------------------------------------------------

def init_memory(memory: dict, params: dict | None = None) -> None:
    memory["p"] = dict(DEFAULT_PARAMS, **(params or {}))
    memory["rng"] = np.random.default_rng(20261107)
    memory["time_used"] = 0.0
    memory["counts"] = {k: [0.0, 0.0, 0.0] for k in PRIORS}
    # showdown informativeness: sums of strength percentiles of hands the
    # opponent showed down after aggressive / passive postflop actions
    memory["sd_agg"] = [0.0, 0.0, 0]   # [strength sum, honest-expected sum, n]
    memory["hand"] = -1
    memory["hist"] = ()
    memory["my_last"] = None      # (street, action, to_call) of my last action
    memory["board_seen"] = ()
    _pf_rank()


def _parse(history, is_button):
    """Yield (street, is_me, action, situation_key) for every action.

    situation_key tells which frequency table the action belongs to."""
    out = []
    street = None
    last_raise_by = None   # True = me, False = opp, None = nobody this street
    first = True
    for a in history:
        if a.street != street:
            street = a.street
            last_raise_by = None
            raised = {True: False, False: False}   # has each side raised this street?
            first = (street == Street.PREFLOP)
        grp = 0 if street == Street.PREFLOP else 1
        if street == Street.PREFLOP and first:
            key = "sbopen"
        elif last_raise_by is not None and last_raise_by != a.is_me:
            key = f"reface{grp}" if raised[a.is_me] else f"face{grp}"
        elif street == Street.PREFLOP:
            key = "unop0"
        else:
            key = "unop1"
        out.append((int(street), a.is_me, a.action, key))
        if a.action in RAISES:
            last_raise_by = a.is_me
            raised[a.is_me] = True
        first = False
    return out


def _record(memory, key, action, street=None):
    if street is not None and street > 0:
        _record(memory, f"{key}s{street}", action)
    c = memory["counts"].setdefault(key, [0.0, 0.0, 0.0])
    if action == Action.FOLD:
        c[0] += 1
    elif action == Action.CHECK_CALL:
        c[1 if key.startswith(("face", "reface")) or key == "sbopen" else 0] += 1
    else:
        c[2] += 1


def _freqs(memory, key, street=None):
    """(fold_or_check, call, raise) probabilities with a prior. With a
    postflop street, per-street counts shrink towards the pooled estimate."""
    p = memory["p"]
    pri = PRIORS[key]
    c = memory["counts"][key]
    n = c[0] + c[1] + c[2]
    k = p["prior_n"]
    pooled = tuple((c[i] + pri[i] * k) / (n + k) for i in range(3))
    if street is None or street == 0 or p["street_n"] <= 0:
        return pooled
    cs = memory["counts"].get(f"{key}s{street}")
    if not cs:
        return pooled
    ns = cs[0] + cs[1] + cs[2]
    ks = p["street_n"]
    return tuple((cs[i] + pooled[i] * ks) / (ns + ks) for i in range(3))


def _alpha(memory):
    """How much the opponent's aggression really means strength (0..1).

    At showdown we compare the strength percentile of hands they bet/raised
    with against what a fully honest player betting the same frequency would
    show (the average of the top-fr band = 1 - fr/2). Random betting shows
    0.5 on average."""
    p = memory["p"]
    s_sum, exp_sum, n = memory["sd_agg"]
    if n == 0:
        return p["alpha_prior"]
    denom = exp_sum - 0.5 * n
    obs = (s_sum - 0.5 * n) / denom if denom > 1e-6 else p["alpha_prior"]
    obs = min(1.0, max(p["alpha_min"], obs))
    k = p["alpha_prior_n"]
    a = (obs * n + p["alpha_prior"] * k) / (n + k)
    return min(1.0, a * p["alpha_scale"])


def on_hand_end(result, memory: dict) -> None:
    if memory.get("hand") != result.hand_index:
        return
    hist = memory["hist"]
    recs = _parse(hist, None)
    for st, is_me, act, key in recs:
        if not is_me:
            _record(memory, key, act, st)
    # infer the opponent's final action (not in any history we saw)
    last = memory.get("my_last")
    if last is not None:
        st, act, my_to_call = last
        grp = 0 if st == 0 else 1
        # facing our raise after they bet this street = re-raise situation
        opp_bet_first = any((not is_me) and s == st and a in RAISES for s, is_me, a, _ in recs)
        fkey = f"reface{grp}" if opp_bet_first else f"face{grp}"
        opp_folded = (not result.went_to_showdown) and result.my_net > 0
        if act in RAISES:
            if opp_folded:
                _record(memory, fkey, Action.FOLD, st)
            elif result.went_to_showdown:
                _record(memory, fkey, Action.CHECK_CALL, st)
    # showdown informativeness
    if result.opp_hole is not None and len(result.board) == 5:
        try:
            _showdown_update(memory, recs, result)
        except Exception:
            pass


def _strength_pct(hole, board, rng, n=160):
    """Percentile of a hand's made strength on `board` vs random hands."""
    dead = list(hole) + list(board)
    others = F.sample_rest(rng, dead, n, 5)
    b = np.array(board, dtype=np.int64)
    theirs = F.current_scores(others, b)
    mine = F.current_scores(np.array([hole], dtype=np.int64), b)[0]
    return float(((theirs < mine).sum() + 0.5 * (theirs == mine).sum()) / n)


def _showdown_update(memory, recs, result):
    rng = memory["rng"]
    cache = {}
    for st, is_me, act, key in recs:
        if is_me or st == 0 or act not in RAISES:
            continue
        nb = {1: 3, 2: 4, 3: 5}[st]
        if nb not in cache:
            cache[nb] = _strength_pct(result.opp_hole, result.board[:nb], rng)
        fr = _freqs(memory, key, st)[2]
        memory["sd_agg"][0] += cache[nb]
        memory["sd_agg"][1] += 1.0 - fr / 2.0
        memory["sd_agg"][2] += 1


# ---------------------------------------------------------------------------
# Weighted Monte Carlo
# ---------------------------------------------------------------------------

def _ranks01(x):
    """Rank-normalise to (0,1): 1 = strongest."""
    n = len(x)
    r = np.empty(n)
    r[np.argsort(x, kind="stable")] = (np.arange(n) + 0.5) / n
    return r


def _simulate(obs, memory, n):
    """Return (my_win_share per sample, opp strength per street dict)."""
    rng = memory["rng"]
    p = memory["p"]
    hole = np.array(obs.hole_cards, dtype=np.int64)
    board = np.array(obs.board, dtype=np.int64)
    nb = len(board)
    dead = list(obs.hole_cards) + list(obs.board)
    draw = F.sample_rest(rng, dead, n, 5 + (5 - nb))
    opp = draw[:, :5]
    if nb < 5:
        full = np.concatenate([np.broadcast_to(board, (n, nb)), draw[:, 5:]], axis=1)
    else:
        full = np.broadcast_to(board, (n, 5))
    mine = F.best_scores(np.broadcast_to(hole, (n, 5)), full)
    theirs = F.best_scores(opp, full)
    win = (mine > theirs) + 0.5 * (mine == theirs)
    pf = _PF_RANK[F.hole_index(opp)]
    strength = {0: pf}
    for st, k in ((1, 3), (2, 4), (3, 5)):
        if nb >= k:
            cur = F.current_scores(opp, board[:k])
            strength[st] = (1 - p["pf_blend"]) * _ranks01(cur) + p["pf_blend"] * pf
    return win, strength


def _weights(obs, memory, strength, alpha):
    """Likelihood of the opponent's observed actions for every sample."""
    p = memory["p"]
    n = len(strength[0])
    w = np.ones(n)
    sw = p["slowplay_w"]
    n_raises = 0
    for st, is_me, act, key in _parse(obs.action_history, obs.is_button):
        if is_me:
            continue
        s = strength.get(st, strength[0])
        a = alpha if st > 0 else min(1.0, alpha + 0.1)
        a *= p["late_alpha"][st]
        if act in RAISES:
            a *= p["repeat_decay"] ** n_raises
            n_raises += 1
        f0, fc, fr = _freqs(memory, key, st)
        if act in RAISES:
            # strongest fr fraction raise
            lik = (s >= 1 - fr).astype(float)
            base = fr
        elif act == Action.CHECK_CALL:
            if key.startswith(("face", "reface")) or key == "sbopen":
                # calls: the band below the raising hands (+ some slowplays)
                lik = ((s >= 1 - fr - fc) & (s < 1 - fr)) + sw * (s >= 1 - fr)
                base = fc
            else:
                # checks: everything below the betting hands (+ slowplays)
                lik = (s < 1 - fr) + sw * (s >= 1 - fr)
                base = f0
        else:
            continue
        w *= a * lik + (1 - a) * base
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 1e-12:
        return np.ones(n)
    return w


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def _budget_n(memory, obs):
    p = memory["p"]
    used = memory["time_used"]
    left_hands = max(1, 1000 - obs.hand_index)
    bank_left = min(p["time_target"], 7.0) - used
    per_hand = bank_left / left_hands
    n = p["n_samples"]
    if per_hand < 0.004:
        n = 260
    if per_hand < 0.0025:
        n = 140
    if per_hand < 0.0012 or (7.0 - used) < 0.4:
        n = 60
    if (7.4 - used) < 0.15:
        n = 0
    return n


def decide(obs, memory) -> Action:
    t0 = time.perf_counter()
    try:
        if obs.hand_index != memory.get("hand"):
            memory["hand"] = obs.hand_index
            memory["my_last"] = None
        act = _decide(obs, memory)
        # remember the full history including our own action for hand-end
        memory["hist"] = obs.action_history + (_Fake(obs.street, act),)
        memory["my_last"] = (int(obs.street), act, obs.to_call)
        return act
    finally:
        memory["time_used"] += time.perf_counter() - t0


class _Fake:
    """Minimal stand-in for an ObservedAction (our own latest action)."""
    __slots__ = ("street", "is_me", "action")

    def __init__(self, street, action):
        self.street = street
        self.is_me = True
        self.action = action


def _fallback(obs):
    legal = obs.legal_actions
    if obs.to_call == 0 or obs.to_call <= obs.big_blind:
        return Action.CHECK_CALL
    return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL


def _decide(obs, memory) -> Action:
    p = memory["p"]
    rng = memory["rng"]
    legal = obs.legal_actions
    n = _budget_n(memory, obs)
    if n == 0:
        return _fallback(obs)

    street = int(obs.street)
    win, strength = _simulate(obs, memory, n)
    alpha = _alpha(memory)
    w = _weights(obs, memory, strength, alpha)
    sw = w.sum()
    e0 = float((w * win).sum() / sw)

    pot, to_call = obs.pot, obs.to_call
    real = (p["real_ip"] if obs.is_button else p["real_oop"])[street]
    evs = {}
    if Action.FOLD in legal:
        evs[Action.FOLD] = 0.0
    call_amt = min(to_call, obs.my_stack)
    r = 1.0 if (call_amt >= obs.my_stack or obs.opp_stack == 0) else real
    evs[Action.CHECK_CALL] = e0 * (pot + call_amt) * r - call_amt

    # opponent response to OUR raise
    grp = 0 if street == 0 else 1
    opp_raised_now = any((not a.is_me) and int(a.street) == street and a.action in RAISES
                         for a in obs.action_history)
    if opp_raised_now:                    # our raise would be a re-raise
        f0, fc, fr = _freqs(memory, f"reface{grp}", street)
    else:
        f0, fc, fr = _freqs(memory, f"face{grp}", street)
    s_now = strength.get(street, strength[0])
    # weighted rank of current strength within their (weighted) range
    order = np.argsort(s_now, kind="stable")
    cw = np.cumsum(w[order]) / sw         # weighted CDF, weakest first
    for act in RAISES:
        if act not in legal or act not in obs.raise_to:
            continue
        raise_to = obs.raise_to[act]
        add = raise_to - obs.my_bet
        opp_add = min(raise_to - obs.opp_bet, obs.opp_stack)
        fold_p = f0 - (p["half_extra_cont"] if act == Action.RAISE_HALF else 0.0)
        fold_p = min(0.95, max(0.02, fold_p))
        # continuing hands: the strongest (1 - fold_p) of their range,
        # softened by alpha (some weak hands call, some strong ones fold)
        cont_mask = np.zeros(len(w))
        cont_mask[order] = (cw > fold_p).astype(float)
        wc = w * (alpha * cont_mask + (1 - alpha) * (1 - fold_p))
        swc = wc.sum()
        p_cont = min(0.98, max(0.02, swc / sw))
        e_c = float((wc * win).sum() / swc) if swc > 0 else e0
        all_in = add >= obs.my_stack or opp_add >= obs.opp_stack
        rr = 1.0 if all_in else real
        tax = 1.0 if all_in else 1.0 - p["reraise_tax"] * min(1.0, fr / max(1e-6, 1 - fold_p))
        evs[act] = (1 - p_cont) * pot + p_cont * (e_c * (pot + add + opp_add) * rr * tax - add)

    noise = p["noise"] * max(pot, 4)
    best = max(evs, key=lambda k: evs[k] + rng.uniform(-noise, noise))
    return best if best in legal else _fallback(obs)
