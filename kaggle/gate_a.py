"""Kaggle wrapper for GATE A. All logic lives in the repo; this script just
wires the Kaggle environment: clone repo -> point AUDIT_DATA_DIR at the
attached private dataset -> for each inter-turn gap setting, run the replay
(emits aggregate JSON + per-conversation cost arrays) then the bootstrap
(95% CIs over conversations) -> outputs in /kaggle/working.

v7: add bootstrap CI step after each replay (per-conv arrays + 95% CIs).
v6: CUDA smoke test + CPU fallback. v5/v4: infra re-pushes.
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
    genv = dict(env, AUDIT_OUT_DIR=out_dir)

    print(f"\n================ GATE A REPLAY, gap={gap}s ================", flush=True)
    r = subprocess.run([sys.executable, f"{REPO}/experiments/gate_a_replay.py",
                        "--gap-s", gap], env=genv, cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"replay failed at gap={gap} (rc={r.returncode})")

    print(f"\n================ GATE A BOOTSTRAP CIs, gap={gap}s ================", flush=True)
    r = subprocess.run([sys.executable, f"{REPO}/experiments/gate_a_bootstrap.py",
                        "--b", "10000"], env=genv, cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"bootstrap failed at gap={gap} (rc={r.returncode})")

print("\nGate A wrapper complete.", flush=True)
