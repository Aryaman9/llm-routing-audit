"""Kaggle wrapper for GATE A. All logic lives in the repo
(experiments/gate_a_replay.py); this script just wires the Kaggle environment:
clone repo -> point AUDIT_DATA_DIR at the attached private dataset ->
run the replay at two inter-turn gap settings -> outputs in /kaggle/working.

v5: identical to v4; re-push after v4 died at Kaggle startup with an empty log
(infrastructure flake - no user code ran).
"""

import glob
import os
import subprocess
import sys
import time

REPO = "/kaggle/working/repo"

for attempt in range(4):
    r = subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/Aryaman9/llm-routing-audit.git", REPO])
    if r.returncode == 0:
        break
    print(f"clone attempt {attempt + 1} failed (rc={r.returncode}); retrying", flush=True)
    subprocess.run(["rm", "-rf", REPO])
    time.sleep(10)
else:
    sys.exit("git clone failed after 4 attempts")

hits = glob.glob("/kaggle/input/**/wildchat_sample.jsonl", recursive=True)
if not hits:
    listing = glob.glob("/kaggle/input/**/*", recursive=True)[:50]
    sys.exit(f"dataset not mounted; /kaggle/input contains: {listing}")

env = dict(os.environ, AUDIT_DATA_DIR=os.path.dirname(hits[0]))

for gap in ("30", "600"):
    out_dir = f"/kaggle/working/results/gap{gap}"
    print(f"\n================ GATE A REPLAY, gap={gap}s ================", flush=True)
    r = subprocess.run([sys.executable, f"{REPO}/experiments/gate_a_replay.py",
                        "--gap-s", gap],
                       env=dict(env, AUDIT_OUT_DIR=out_dir), cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"replay failed at gap={gap} (rc={r.returncode})")

print("\nGate A wrapper complete.", flush=True)
