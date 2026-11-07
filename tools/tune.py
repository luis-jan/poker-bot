import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import strategy
from sim import match
over = json.loads(sys.argv[1])
for k, v in over.items():
    if isinstance(v, dict):
        strategy.DEFAULT_PARAMS[k] = {int(a): b for a, b in v.items()}
    else:
        strategy.DEFAULT_PARAMS[k] = v
opps = ["tools/opponents/opp_mc.py", "tools/opponents/opp_nit.py", "tools/opponents/opp_lag.py", "baseline-bots/baseline_random.py"]
tot = []
for o in opps:
    for seed in (11, 12):
        m, se, v = match("agent.py", o, 600, seed, quiet=True)
        tot.append(m)
print(sys.argv[1], " ".join(f"{x:+.2f}" for x in tot), f"AVG {sum(tot)/len(tot):+.3f}", flush=True)
