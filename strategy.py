"""
strategy.py - the decision logic behind agent.py.

Core loop for every decision:
  1. Estimate the opponent's range from what they did this hand, scaled by how
     often THIS opponent bets/raises/folds (learned during the half).
  2. Monte Carlo equity vs that range with a vectorised numpy evaluator
     (~1 ms for 400 samples).
  3. Compare equity with the price (pot odds) to fold/call, and with value
     thresholds to bet/raise. Bluff only against opponents who fold a lot.

Everything is defensive: any exception falls back to check/fold-or-call logic
in agent.py, and the per-half time bank is tracked so we slow down never.
"""

from __future__ import annotations

import time

import numpy as np

import fasteval as F
from game import Action, Street

DEFAULT_PARAMS = {
    "n_samples": 320,
    # minimum extra equity over pot odds needed to call (we won't realise all
    # our equity, especially out of position)
    "call_margin": {0: 0.03, 1: 0.05, 2: 0.03, 3: 0.0},
    "oop_extra": 0.02,
    # equity vs range needed to bet when checked to / raise when facing a bet
    "bet_thresh": {1: 0.58, 2: 0.60, 3: 0.62},
    "raise_thresh": {1: 0.70, 2: 0.72, 3: 0.75},
    "pot_size_thresh": 0.78,         # above this, use the bigger sizing
    # preflop: raise if equity vs (adjusted) range exceeds these
    "pf_open": 0.53,                 # button first in
    "pf_iso": 0.55,                  # big blind vs a limp
    "pf_3bet": 0.57,                 # facing a raise
    "pf_4bet": 0.61,
    "pf_fold_floor": 0.0,            # never fold preflop below price unless...
    # range narrowing
    "prior_raise": 0.30, "prior_w": 12.0,
    "prior_fold": 0.40,
    "min_q": 0.07,
    # bluffing
    "bluff_fold_needed": 0.50,
    "bluff_freq": 0.35,
    "semibluff_lo": 0.42,
}


# ---------------------------------------------------------------------------
# Opponent statistics
# ---------------------------------------------------------------------------

def init_memory(memory: dict, params: dict | None = None) -> None:
    memory["p"] = dict(DEFAULT_PARAMS, **(params or {}))
    memory["rng"] = np.random.default_rng(20261107)
    memory["time_used"] = 0.0
    # opponent action counts: [opportunities, raises] per street group
    memory["opp_raise"] = {0: [0, 0], 1: [0, 0]}   # 0 = preflop, 1 = postflop
    memory["opp_vs_bet"] = [0, 0]                  # [times faced our bet, folds]
    memory["cur_hand"] = -1
    memory["seen"] = 0
    memory["my_last_raise"] = False
    memory["pf_sorted"] = None


def _process_history(obs, memory: dict) -> None:
    """Update opponent stats with opponent actions we haven't seen yet."""
    if obs.hand_index != memory["cur_hand"]:
        memory["cur_hand"] = obs.hand_index
        memory["seen"] = 0
    hist = obs.action_history
    for a in hist[memory["seen"]:]:
        if a.is_me:
            continue
        grp = 0 if a.street == Street.PREFLOP else 1
        st = memory["opp_raise"][grp]
        st[0] += 1
        if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            st[1] += 1
    memory["seen"] = len(hist)


def on_hand_end(result, memory: dict) -> None:
    # Did the opponent face our bet as the last aggressive action?
    if memory.get("my_last_raise") and memory.get("cur_hand") == result.hand_index:
        memory["opp_vs_bet"][0] += 1
        if not result.went_to_showdown and result.my_net > 0:
            memory["opp_vs_bet"][1] += 1
    memory["my_last_raise"] = False


def _raise_freq(memory, grp):
    p = memory["p"]
    opp, r = memory["opp_raise"][grp]
    return (r + p["prior_raise"] * p["prior_w"]) / (opp + p["prior_w"])


def _fold_freq(memory):
    p = memory["p"]
    n, f = memory["opp_vs_bet"]
    return (f + p["prior_fold"] * 8) / (n + 8)


def _range_fraction(obs, memory) -> float:
    """Fraction of random hands the opponent's range is narrowed to."""
    p = memory["p"]
    q = 1.0
    fold_f = _fold_freq(memory)
    facing_bet = False   # was the opponent facing a bet when acting?
    street = None
    for a in obs.action_history:
        if a.street != street:
            street = a.street
            facing_bet = False
        is_raise = a.action in (Action.RAISE_HALF, Action.RAISE_POT)
        if a.is_me:
            facing_bet = is_raise
            continue
        grp = 0 if a.street == Street.PREFLOP else 1
        if is_raise:
            q *= min(1.0, max(0.18, _raise_freq(memory, grp) * 1.25))
            facing_bet = False
        elif a.action == Action.CHECK_CALL and facing_bet:
            # calls tell more about opponents who fold often
            q *= 1.0 - 0.45 * fold_f if grp else 1.0 - 0.25 * fold_f
            facing_bet = False
        elif a.action == Action.CHECK_CALL and grp == 1:
            # checking caps the range a little (strong hands often bet)
            q *= 1.0
    # Preflop: the button that just limps / BB that checks are slightly weaker,
    # but we ignore it (q only narrows towards strong hands).
    return max(p["min_q"], q)


# ---------------------------------------------------------------------------
# Equity
# ---------------------------------------------------------------------------

def equity(obs, memory, q: float, n: int) -> float:
    rng = memory["rng"]
    hole = np.array(obs.hole_cards, dtype=np.int64)
    board = np.array(obs.board, dtype=np.int64)
    nb = len(board)
    dead = list(obs.hole_cards) + list(obs.board)
    total = int(min(1600, max(n, n / q))) if q < 0.999 else n
    draw = F.sample_rest(rng, dead, total, 5 + (5 - nb))
    opp = draw[:, :5]
    if q < 0.999:
        keep = max(30, int(round(total * q)))
        pf = F.PREFLOP[F.hole_index(opp)]
        if nb == 0:
            strength = pf
        else:
            cur = F.current_scores(opp, board)
            # rank-normalise both and blend (made hand matters most postflop)
            r_cur = np.argsort(np.argsort(cur)) / total
            r_pf = np.argsort(np.argsort(pf)) / total
            strength = 0.8 * r_cur + 0.2 * r_pf
        sel = np.argpartition(-strength, keep - 1)[:keep]
        draw = draw[sel]
        opp = draw[:, :5]
    m = len(draw)
    if nb < 5:
        full = np.concatenate([np.broadcast_to(board, (m, nb)), draw[:, 5:]], axis=1)
    else:
        full = np.broadcast_to(board, (m, 5))
    mine = F.best_scores(np.broadcast_to(hole, (m, 5)), full)
    theirs = F.best_scores(opp, full)
    return float(((mine > theirs).sum() + 0.5 * (mine == theirs).sum()) / m)


def preflop_equity_random(hole) -> float:
    return float(F.PREFLOP[F.hole_index(np.array([hole], dtype=np.int64))][0])


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def _pick(obs, *prefs):
    for a in prefs:
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL


def decide(obs, memory) -> Action:
    t0 = time.perf_counter()
    try:
        act = _decide(obs, memory)
    finally:
        memory["time_used"] += time.perf_counter() - t0
    if act in (Action.RAISE_HALF, Action.RAISE_POT):
        memory["my_last_raise"] = True
    elif act == Action.CHECK_CALL:
        memory["my_last_raise"] = False
    return act


def _budget_samples(memory, obs) -> int:
    p = memory["p"]
    used = memory["time_used"]
    left_hands = max(1, 1000 - obs.hand_index)
    bank_left = 7.0 - used               # keep a 1 s safety margin
    per_hand = bank_left / left_hands
    n = p["n_samples"]
    if per_hand < 0.0045:
        n = 200
    if per_hand < 0.0025:
        n = 100
    if per_hand < 0.0012 or bank_left < 0.3:
        n = 0   # emergency: table / heuristics only
    return n


def _decide(obs, memory) -> Action:
    p = memory["p"]
    rng = memory["rng"]
    _process_history(obs, memory)
    legal = obs.legal_actions
    can_raise = Action.RAISE_POT in legal
    street = int(obs.street)
    to_call = obs.to_call
    pot = obs.pot
    n = _budget_samples(memory, obs)
    q = _range_fraction(obs, memory)

    if n == 0:
        if street == 0:
            e = preflop_equity_random(obs.hole_cards)
        else:
            e = equity(obs, memory, 1.0, 60)
    elif street == 0 and q >= 0.999:
        e = preflop_equity_random(obs.hole_cards)
    else:
        e = equity(obs, memory, q, n)

    in_position = obs.is_button  # button acts last postflop
    all_in_call = to_call >= obs.my_stack

    # ------------------------- preflop -------------------------
    if street == 0:
        if to_call > 0:
            price = to_call / (pot + to_call)
            nraise = obs.raises_this_street
            if nraise == 0:            # button, first in (to_call = 1)
                thresh = p["pf_open"]
            elif nraise == 1:
                thresh = p["pf_3bet"]
            else:
                thresh = p["pf_4bet"]
            if can_raise and e >= thresh:
                return Action.RAISE_POT
            margin = 0.0 if all_in_call else p["call_margin"][0] + (0 if in_position else p["oop_extra"])
            if e >= price + margin:
                return Action.CHECK_CALL
            return _pick(obs, Action.FOLD)
        # big blind after a limp
        if can_raise and e >= p["pf_iso"]:
            return Action.RAISE_POT
        return Action.CHECK_CALL

    # ------------------------- postflop -------------------------
    if to_call > 0:
        price = to_call / (pot + to_call)
        if can_raise and e >= p["raise_thresh"][street]:
            # with a strong hand, raise (pot sizing when clearly ahead)
            if e >= p["pot_size_thresh"] or obs.raise_to.get(Action.RAISE_HALF, 0) >= obs.my_bet + obs.my_stack:
                return Action.RAISE_POT
            return Action.RAISE_HALF
        margin = 0.0 if (all_in_call or street == 3) else p["call_margin"][street] + (0 if in_position else p["oop_extra"])
        if e >= price + margin:
            return Action.CHECK_CALL
        return _pick(obs, Action.FOLD)

    # checked to us / we act first
    if can_raise and e >= p["bet_thresh"][street]:
        if e >= p["pot_size_thresh"]:
            return Action.RAISE_POT
        return Action.RAISE_HALF
    if can_raise and street >= 1:
        fold_f = _fold_freq(memory)
        if fold_f >= p["bluff_fold_needed"]:
            # bluff more on later streets and with some equity (semi-bluff)
            freq = p["bluff_freq"] * (1.0 + (fold_f - p["bluff_fold_needed"]) * 2)
            if street < 3 and e >= p["semibluff_lo"]:
                freq *= 1.5
            if rng.random() < min(0.8, freq):
                return Action.RAISE_HALF
    return Action.CHECK_CALL
