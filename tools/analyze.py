import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2h
from h2h import *
other = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]] or [1, 2]
agg = collections.defaultdict(lambda: [0, 0])  # key -> [count, my chips]
pf = collections.Counter()
for seed in seeds:
    for seat_of_a in (0, 1):
        A, B = load_mine({}), load_isolated(other, f"o{time.time_ns()}")
        mems = [{}, {}]
        for b, m in zip((A, B), mems): b.new_half(AgentConfig(), m)
        seat_bot = {seat_of_a: 0, 1 - seat_of_a: 1}
        for i in range(1000):
            g = Game(make_deal(seed, i), button_for_deal(i), i)
            last = None
            while not g.is_terminal:
                seat = g.to_act; bi = seat_bot[seat]
                obs = g.observation(seat)
                a = Action(int((A, B)[bi].choose_action(obs, mems[bi])))
                if obs.street == 0:
                    who = "me" if bi == 0 else "op"
                    pos = "SB" if obs.is_button else "BB"
                    pf[(who, pos, obs.raises_this_street, a.name)] += 1
                last = (bi, int(obs.street), a)
                g.apply(a)
            res = g.result()
            for seat in (0, 1):
                (A, B)[seat_bot[seat]].on_hand_end(g.hand_result(seat), mems[seat_bot[seat]])
            net = res.nets[seat_of_a]
            if res.went_to_showdown:
                key = f"showdown (pot {'big' if res.pot >= 60 else 'small'})"
            else:
                key = f"{'I' if last[0]==0 else 'they'} fold on {['pre','flop','turn','river'][last[1]]}"
            agg[key][0] += 1; agg[key][1] += net
tot_hands = 2000 * len(seeds)
for k, (c, chips) in sorted(agg.items(), key=lambda kv: kv[1][1]):
    print(f"{k:28s} hands {c:5d}  my net {chips/2/len(seeds)*1:+8.1f} bb per match")
print()
for who in ("me", "op"):
    for pos in ("SB", "BB"):
        for nr in range(4):
            row = {a: pf[(who, pos, nr, a)] for a in ("FOLD", "CHECK_CALL", "RAISE_HALF", "RAISE_POT")}
            t = sum(row.values())
            if t: print(who, pos, "raises", nr, " ".join(f"{a}:{v/t:.2f}" for a, v in row.items()), t)
