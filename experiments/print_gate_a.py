"""Pretty-print a gate_a results JSON + apply the Gate A verdict rule."""

import json
import sys
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else "results/gate_a_2026-06-12.json"
r = json.load(open(path, encoding="utf-8"))
print("meta:", {k: v for k, v in r.items() if k != "datasets"})

worst = (0.0, None)
for ds, d in r["datasets"].items():
    print(f"\n#### {ds} | convs: {d['n_convs']} | real-ts convs: {d['real_ts_convs']}")
    for prov, rows in d["providers"].items():
        print(f"--- {prov} ---")
        print(f"{'policy':<30}{'S0%':>8}{'S1%':>8}{'overst%':>9}{'sw/conv':>9}{'hit%':>7}")
        for pname, row in rows.items():
            s0, s1 = row.get("savings_cm0"), row.get("savings_cm1")
            if s0 is None:
                continue
            ov = row.get("overstatement_pct", float("nan"))
            print(f"{pname:<30}{s0 * 100:>8.1f}{s1 * 100:>8.1f}{ov:>9.1f}"
                  f"{row['switches_per_conv']:>9.2f}{row['cache_hit_rate'] * 100:>7.1f}")
            if pname.startswith("routellm@") and s0 > 0.02 and ov == ov:
                if ov > worst[0]:
                    worst = (ov, f"{ds}/{prov}/{pname}")

print(f"\nGATE A: max router-savings overstatement = {worst[0]:.1f}% ({worst[1]})")
print("verdict:", "GO" if worst[0] > 20 else
      "GO (reframe: latency + regime map)" if worst[0] > 10 else "STOP - pivot")
