"""Clone public bot repos and play our bot against each of them.

    python tools/scout.py https://github.com/owner/repo [more URLs...] --seeds 1 2 3

Each repo is cloned (shallow) into ../scouted/<owner>_<repo>. Only repos with an
agent.py exposing new_half / choose_action / on_hand_end are played. Results
are printed per repo. Their code is only run locally for matches; nothing is
copied into our bot.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import h2h  # noqa: E402

OUT = os.path.join(os.path.dirname(ROOT), "scouted")


def fetch(url):
    owner_repo = url.rstrip("/").removesuffix(".git").split("github.com/")[-1]
    dest = os.path.join(OUT, owner_repo.replace("/", "_"))
    if os.path.isdir(dest):
        subprocess.run(["git", "-C", dest, "pull", "-q", "--depth", "1"], check=False)
    else:
        os.makedirs(OUT, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", url, dest], check=False, timeout=300)
    return owner_repo, dest


def compatible(dest):
    path = os.path.join(dest, "agent.py")
    if not os.path.isfile(path):
        return False
    src = open(path, encoding="utf-8", errors="ignore").read()
    return all(f"def {fn}(" in src for fn in ("new_half", "choose_action", "on_hand_end"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    for url in args.urls:
        name, dest = fetch(url)
        if not compatible(dest):
            print(f"{name}: skipped (no compatible agent.py)")
            continue
        means = []
        for s in args.seeds:
            try:
                mean, se, *_ = h2h.match(dest, s, 1000, {})
            except Exception as exc:
                print(f"{name}: crashed on seed {s}: {type(exc).__name__}: {exc}")
                break
            v = "WIN" if mean - 1.96 * se > 0 else ("LOSS" if mean + 1.96 * se < 0 else "CHOP")
            means.append(mean)
            print(f"{name} seed {s}: {mean:+.3f} ± {1.96 * se:.3f} {v}", flush=True)
        if means:
            print(f"{name}: AVERAGE {sum(means) / len(means):+.3f} bb/deal over {len(means)} matches\n", flush=True)
