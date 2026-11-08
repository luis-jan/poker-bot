"""Score a param config on a mixed field: rivals (folders, h2h) + file bots (sim)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import h2h, sim, strategy
cfg = json.loads(sys.argv[1]); which = sys.argv[2]
for k, v in cfg.items():
    strategy.DEFAULT_PARAMS[k] = {int(a): b for a, b in v.items()} if isinstance(v, dict) else v
FRESH = os.environ.get("FRESH") == "1"
plan = {"OUT": ("/home/claude/scouted/forks/harishkumar123tamu_outliers", [1301, 1302, 1303, 1304, 1305, 1306] if FRESH else [81, 82, 83, 86, 87, 88]),
        "BP": ("/home/claude/scouted/forks/Dushyant-Betala_heads-up-bundle", [1201, 1202, 1203, 1204, 1205, 1206] if FRESH else [1101, 1102, 1103, 1104, 1105, 1106]),
        "CYH": ("/home/claude/scouted/yahwehc/td26-poker-bot-main", [813, 814, 815, 816, 817, 818, 819, 820] if FRESH else [805, 806, 807, 808]),
        "TM": ("/home/claude/teammate/repo", [309, 310, 311, 312] if FRESH else [5, 6]),
        "RF": ("/home/claude/scouted/SahithReddy-GitHub_poker-bot-tamu-datathon", [309, 310, 311, 312] if FRESH else [201, 202]),
        "STATION": ("baseline-bots/baseline_calling_station.py", [92, 93] if FRESH else [91]),
        "MC": ("tools/opponents/opp_mc.py", [92, 93] if FRESH else [91]),
        "NIT": ("tools/opponents/opp_nit.py", [92, 93] if FRESH else [91])}
out = []
for name in which.split(","):
    path, seeds = plan[name]
    ms = []
    for s in seeds:
        if os.path.isdir(path):
            ms.append(h2h.match(path, s, 1000, cfg)[0])
        else:
            ms.append(sim.match("agent.py", path, 1000, s, quiet=True)[0])
    out.append(f"{name} {sum(ms)/len(ms):+.2f}")
print(sys.argv[1], " | ".join(out), flush=True)
