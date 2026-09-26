"""
Collects everything needed to decide the next submission into ONE small file:
  reports/v3_report_for_claude.md
Send that file back (paste it). It contains no raw data rows, only aggregates.
"""

import os
import sys
import glob
import json
import platform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config

REP = os.path.join(config.BASE_DIR, "reports")


def last(entries, phase, n=1):
    xs = [e for e in entries if e.get("phase") == phase]
    return xs[-n:]


def main():
    entries = []
    path = os.path.join(REP, "experiments.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    train = last(entries, "v3_full_universe", 3)
    preds = last(entries, "v3_test_predict", 4)
    logs = sorted(glob.glob(os.path.join(config.BASE_DIR, "artifacts", "train_universe", "train_*.log")))
    tail = open(logs[-1], encoding="utf-8").read().splitlines()[-45:] if logs else []
    try:
        import lightgbm, sklearn, psutil
        vers = f"lightgbm {lightgbm.__version__}, sklearn {sklearn.__version__}, RAM {psutil.virtual_memory().total/2**30:.1f} GB"
    except Exception:
        import lightgbm, sklearn
        vers = f"lightgbm {lightgbm.__version__}, sklearn {sklearn.__version__}"

    out = ["# v3 report for Claude", "",
           f"- python {platform.python_version()} on {platform.platform()}; {vers}",
           "- LEADERBOARD SCORE of the submitted file(s): **<fill in: which tag, score>**", ""]
    out += ["## Training (honest full-universe OOF)", "```json"]
    out += [json.dumps(t, indent=1) for t in train]
    out += ["```", "", "## Test predictions", "```json"]
    for p in preds:
        p = {k: v for k, v in p.items() if k != "universe"}
        out.append(json.dumps(p, indent=1))
    out += ["```", "", "## Train log tail", "```"] + tail + ["```"]
    os.makedirs(REP, exist_ok=True)
    dst = os.path.join(REP, "v3_report_for_claude.md")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
