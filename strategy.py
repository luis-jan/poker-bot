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
    "real_oop": (0.94, 0.90, 0.95, 1.0),
    "real_ip": (1.0, 0.97, 0.99, 1.0),
    "prior_n": 10.0,             # pseudo-counts behind each frequency
    "alpha_prior": 0.65,         # prior informativeness of opponent actions
    "alpha_prior_n": 6.0,
    "alpha_min": 0.15,
    "alpha_scale": 1.75,         # trust in observed aggression (tuned)
    "half_extra_cont": 0.07,     # opponents continue more vs half-pot bets
    "reraise_tax": 0.15,         # small penalty for being re-raised off equity
    "slowplay_w": 0.25,          # weight a passive action keeps for strong hands
    "noise": 0.012,              # EV noise as a fraction of the pot
    "pf_blend": 0.25,            # preflop strength share in postflop ranking
    # aggression on later streets is discounted: showdown samples overstate
    # how honest opponents are (we never see the bluffs we folded to), and
    # repeated barrels shouldn't compound into "only the nuts"
    "late_alpha": (1.0, 1.0, 0.85, 0.85),   # alpha multiplier per street
    "repeat_decay": 0.8,          # each extra opp raise in a hand counts less
    "mdf_tol": 0.2,               # defense floor tolerance (x pot), 0 = off
    "street_n": 8.0,
    "size_mix": 0.3,              # chance to swap to the other (still +EV) bet size
    "barrel_mult": 0.0,           # weight of "they keep betting" when we call (0 = off)
    "bet_margin": 0.15,
    "raise_gate_lo": 0.10,        # opponent raise rate vs our bets: margin off below ...
    "raise_gate_hi": 0.17,        # ... and fully on above           # extra EV (x pot) a bet into a check must show
    "use_chkface": 1,             # separate stats for "they checked, then faced our bet"
    "adaptive_sw": 1,             # learn how many strong hands the opponent checks
    "pf_raise_real": 1.0,         # equity realisation of our preflop raises (button)
    "pf_raise_real_oop": 1.0,     # ... and from the big blind
    "bluff_cut": 0.35,            # showdown strength below this counts as a bluff
    "bluff_prior": 0.20,
    "floor_lo": 0.15,             # bluff rate at which the floor is fully off ...
    "floor_hi": 0.22,             # ... and fully on (both 0 = always on)              # shrinkage of per-street frequencies
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
    "chkface1": (0.40, 0.42, 0.18),   # facing our bet after checking (check-raise rate)
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
    memory["barrel"] = {1: [0, 0], 2: [0, 0]}   # [times we called their bet, times they bet again]
    memory["sd_bluff"] = [0, 0]        # [weak aggressive showdowns, all aggressive showdowns]
    memory["hand"] = -1
    memory["my_def"] = [0, 0]         # our own [bets faced, folds] postflop
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
            checked = {True: False, False: False}  # has each side checked this street?
            first = (street == Street.PREFLOP)
        grp = 0 if street == Street.PREFLOP else 1
        if street == Street.PREFLOP and first:
            key = "sbopen"
        elif last_raise_by is not None and last_raise_by != a.is_me:
            if raised[a.is_me]:
                key = f"reface{grp}"
            elif grp == 1 and checked[a.is_me]:
                key = "chkface1"           # checked, then faced a bet
            else:
                key = f"face{grp}"
        elif street == Street.PREFLOP:
            key = "unop0"
        else:
            key = "unop1"
        out.append((int(street), a.is_me, a.action, key))
        if a.action in RAISES:
            last_raise_by = a.is_me
            raised[a.is_me] = True
        elif a.action == Action.CHECK_CALL and last_raise_by is None:
            checked[a.is_me] = True
        first = False
    return out


def _record(memory, key, action, street=None):
    if street is not None and street > 0:
        _record(memory, f"{key}s{street}", action)
    c = memory["counts"].setdefault(key, [0.0, 0.0, 0.0])
    if action == Action.FOLD:
        c[0] += 1
    elif action == Action.CHECK_CALL:
        c[1 if key.startswith(("face", "reface", "chkface")) or key == "sbopen" else 0] += 1
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
        opp_chk_first = any((not is_me) and s == st and a == Action.CHECK_CALL and k == "unop1"
                            for s, is_me, a, k in recs)
        fkey = (f"reface{grp}" if opp_bet_first else
                "chkface1" if (grp == 1 and opp_chk_first) else f"face{grp}")
        opp_folded = (not result.went_to_showdown) and result.my_net > 0
        if act in RAISES:
            if opp_folded:
                _record(memory, fkey, Action.FOLD, st)
            elif result.went_to_showdown:
                _record(memory, fkey, Action.CHECK_CALL, st)
    # barrel tracking: after we called their flop/turn bet, did they bet again?
    for st in (1, 2):
        on_st = [r for r in recs if r[0] == st]
        nxt = [r for r in recs if r[0] == st + 1 and not r[1]]
        if not on_st or not nxt:
            continue
        opp_bet = any((not is_me) and a in RAISES for _, is_me, a, _k in on_st)
        last_me = [r for r in on_st if r[1]]
        if opp_bet and last_me and last_me[-1][2] == Action.CHECK_CALL and on_st[-1][1]:
            memory["barrel"][st][0] += 1
            memory["barrel"][st][1] += any(a in RAISES for _s, _m, a, _k in nxt)
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


def p_bluff_cut(memory):
    return memory["p"]["bluff_cut"]


def _bluff_rate(memory):
    """Share of the opponent's shown-down bets/raises that were weak hands."""
    k, n = memory["sd_bluff"]
    p = memory["p"]
    return (k + p["bluff_prior"] * 8) / (n + 8)


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
        # bluff tracking: an aggressive action shown down with a weak hand
        memory["sd_bluff"][0] += cache[nb] < p_bluff_cut(memory)
        memory["sd_bluff"][1] += 1
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
            if key.startswith(("face", "reface", "chkface")) or key == "sbopen":
                # calls: the band below the raising hands (+ some slowplays)
                lik = ((s >= 1 - fr - fc) & (s < 1 - fr)) + sw * (s >= 1 - fr)
                base = fc
            else:
                # checks: everything below the betting hands (+ slowplays).
                # How many strong hands check is learned from how often this
                # opponent check-raises: P(raise | checked) = cr implies
                # sw = cr (1 - b) / (b (1 - cr)) with b = its betting rate.
                sw_here = sw
                if p["adaptive_sw"] and st > 0:
                    cr = _freqs(memory, "chkface1", st)[2]
                    b = min(0.95, max(0.05, fr))
                    sw_here = min(1.0, max(0.1, cr * (1 - b) / (b * max(0.05, 1 - cr))))
                lik = (s < 1 - fr) + sw_here * (s >= 1 - fr)
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
    if to_call > 0 and street in (1, 2) and p["barrel_mult"] > 0 and call_amt < obs.my_stack:
        n_b, k_b = memory["barrel"][street]
        wb = min(1.0, p["barrel_mult"] * (k_b + 0.5 * 6) / (n_b + 6))
        rest = min(obs.my_stack - call_amt, obs.opp_stack)
        ev_allin = e0 * (pot + call_amt + 2 * rest) - (call_amt + rest)
        evs[Action.CHECK_CALL] = (1 - wb) * evs[Action.CHECK_CALL] + wb * ev_allin

    # opponent response to OUR raise
    grp = 0 if street == 0 else 1
    opp_raised_now = any((not a.is_me) and int(a.street) == street and a.action in RAISES
                         for a in obs.action_history)
    opp_checked_now = street > 0 and any(
        (not a.is_me) and int(a.street) == street and a.action == Action.CHECK_CALL
        for a in obs.action_history) and to_call == 0
    if opp_raised_now:                    # our raise would be a re-raise
        f0, fc, fr = _freqs(memory, f"reface{grp}", street)
    elif opp_checked_now and p["use_chkface"]:
        f0, fc, fr = _freqs(memory, "chkface1", street)   # they checked: trap-aware
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
        if street == 0 and not all_in:
            # a re-raised preflop pot is big and hard to play well postflop
            # (especially out of position): realise less of our equity
            rr *= p["pf_raise_real"] if obs.is_button else p["pf_raise_real_oop"]
        tax = 1.0 if all_in else 1.0 - p["reraise_tax"] * min(1.0, fr / max(1e-6, 1 - fold_p))
        evs[act] = (1 - p_cont) * pot + p_cont * (e_c * (pot + add + opp_add) * rr * tax - add)

    if to_call == 0 and street > 0 and p["bet_margin"] > 0:
        # Against opponents that often RAISE our bets, a thin bet is risky:
        # it must beat checking by a margin that grows with their raise rate.
        # (Opponents that rarely raise get our thin value bets as usual.)
        r_face = _freqs(memory, "face1")[2]
        gate = (r_face - p["raise_gate_lo"]) / max(1e-6, p["raise_gate_hi"] - p["raise_gate_lo"])
        margin = p["bet_margin"] * min(1.0, max(0.0, gate))
        for a in RAISES:
            if a in evs:
                evs[a] -= margin * pot
    noise = p["noise"] * max(pot, 4)
    best = max(evs, key=lambda k: evs[k] + rng.uniform(-noise, noise))
    # sizing disguise: when we've decided to bet/raise postflop, don't let the
    # size reveal strength (opponents can learn "pot = strong" at showdown)
    if best in RAISES and street > 0 and p["size_mix"] > 0:
        other = Action.RAISE_HALF if best == Action.RAISE_POT else Action.RAISE_POT
        if other in evs and evs[other] > evs.get(Action.CHECK_CALL, -1e9) and rng.random() < p["size_mix"]:
            best = other

    # Defense floor: EV-based bluffers profit if we fold to bets more often
    # than their bet size allows (to_call / pot). If our own fold rate is
    # above that, call marginal hands that are only slightly -EV.
    if best == Action.FOLD and street > 0 and p["mdf_tol"] > 0:
        faced, folded = memory["my_def"]
        my_fold = (folded + 0.35 * 10) / (faced + 10)
        breakeven = to_call / max(1, pot)
        excess = my_fold - breakeven
        if p["floor_hi"] > p["floor_lo"]:
            gate = (_bluff_rate(memory) - p["floor_lo"]) / (p["floor_hi"] - p["floor_lo"])
            excess *= min(1.0, max(0.0, gate))
        if excess > 0:
            tol = p["mdf_tol"] * min(1.0, excess / 0.15) * pot
            if evs[Action.CHECK_CALL] > -tol:
                best = Action.CHECK_CALL
    if street > 0 and to_call > 0:
        memory["my_def"][0] += 1
        memory["my_def"][1] += best == Action.FOLD
    return best if best in legal else _fallback(obs)
