"""GATE A (Kaggle kernel): does the CM0-vs-CM1 gap have teeth on real conversations?

Pipeline: clone public repo (realcost lib + pricing snapshot) -> load private
conversation samples (WildChat/LMSYS, attached Kaggle dataset) -> score every
user turn with RouteLLM's open BERT router -> simulate routing policies
offline -> account each policy under CM0 (linear, what papers report) and CM1
(cache-aware, what providers bill) -> claimed-vs-real savings gap + verdict.

GO if router policies' claimed savings (vs fixed-large) are overstated by
>20% relative under CM1 at realistic turn gaps; 10-20% GO-REFRAME; else NO-GO.
"""

import json
import random
import subprocess
import sys

subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/Aryaman9/llm-routing-audit.git",
                "/kaggle/working/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"], check=True)
sys.path.insert(0, "/kaggle/working/repo/src")

import numpy as np
import pandas as pd
import torch

from realcost import Trajectory, Turn, load_snapshot, replay

print("cuda:", torch.cuda.is_available())

# ---------------- data ----------------
DATA = "/kaggle/input/routing-audit-convs"
MAX_TURNS = 40


def load_convs(path):
    convs = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        turns, user_tok, user_text = [], 0, ""
        for m in r["turns"]:
            if m["role"] == "user":
                user_tok += m["n_tokens_cl100k"]
                user_text = (m.get("text") or "")[:4000]
            elif m["role"] == "assistant":
                turns.append({"user_tokens": user_tok,
                              "out_tokens": max(1, m["n_tokens_cl100k"]),
                              "route_text": user_text})
                user_tok, user_text = 0, ""
        if len(turns) >= 2:
            convs.append({"conv_id": r["conv_id"], "turns": turns[:MAX_TURNS]})
    return convs


wild = load_convs(f"{DATA}/wildchat_sample.jsonl")
lmsys = load_convs(f"{DATA}/lmsys_sample.jsonl")
print(f"wildchat {len(wild)} convs | lmsys {len(lmsys)} convs")
all_turns = [t for src in (wild, lmsys) for c in src for t in c["turns"]]
prompts = [t["route_text"] for t in all_turns]
print("routed turns:", len(prompts))

# ---------------- router: RouteLLM BERT ----------------
scores = None
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "routellm"], check=True)
    from routellm.routers.routers import ROUTER_CLS
    import inspect
    print("routellm routers:", list(ROUTER_CLS.keys()))
    cls = ROUTER_CLS["bert"]
    print("bert init signature:", inspect.signature(cls.__init__))
    router = cls()
    probe = router.calculate_strong_win_rate("What is 2+2?")
    print("routellm bert OK, probe win rate:", probe)
    scores = [float(router.calculate_strong_win_rate(p)) for p in prompts]
    print("scored via routellm package")
except Exception as e:  # noqa: BLE001
    print("routellm package path failed ->", repr(e))

if scores is None:
    # Fallback: load the published checkpoint directly. Label-column DIRECTION
    # is chosen by self-calibration on easy/hard probe prompts; since routing
    # thresholds below are score-quantiles, any monotone transform of the true
    # win rate yields identical routing decisions.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ckpt = "routellm/bert_gpt4_augmented"
    tok = AutoTokenizer.from_pretrained(ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = AutoModelForSequenceClassification.from_pretrained(ckpt).to(dev).eval()
    print("fallback checkpoint config:", mod.config.num_labels, mod.config.id2label)

    def score_batch(texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], truncation=True, max_length=512,
                    padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                out.append(torch.softmax(mod(**b).logits, -1).cpu())
        return torch.cat(out).numpy()

    easy = ["hi", "what time is it?", "thanks, that helps!"]
    hard = ["Prove that the sum of reciprocals of the primes diverges, rigorously.",
            "Derive the Euler-Lagrange equations from the principle of least action.",
            "Implement a lock-free concurrent skip list in C++ and argue linearizability."]
    pe, ph = score_batch(easy), score_batch(hard)
    col = int(np.argmax(ph.mean(0) - pe.mean(0)))
    print(f"direction calibration -> column {col} "
          f"(easy={pe.mean(0).round(3)}, hard={ph.mean(0).round(3)})")
    scores = score_batch(prompts)[:, col].tolist()

for t, s in zip(all_turns, scores):
    t["win"] = float(s)

# ---------------- policies + replay ----------------
snap = load_snapshot("/kaggle/working/repo/pricing/snapshot_2026-06-11.json")
ws = np.array([t["win"] for t in all_turns])
TH = {"router@20": float(np.quantile(ws, 0.8)),
      "router@50": float(np.quantile(ws, 0.5))}
print("score thresholds:", TH, "| score deciles:", np.percentile(ws, [10, 50, 90]).round(3))

rows = []
for ds_name, convs in (("wildchat", wild), ("lmsys", lmsys)):
    for provider in ("anthropic", "openai"):
        prices = {"small": snap[provider].models["small"],
                  "large": snap[provider].models["large"]}
        caches = {"small": snap[provider].cache, "large": snap[provider].cache}
        for gap in (30, 120, 600):
            rng = random.Random(42)
            for c in convs:
                traj = Trajectory(
                    turns=[Turn(user_tokens=t["user_tokens"], t_seconds=k * gap,
                                output_tokens=t["out_tokens"])
                           for k, t in enumerate(c["turns"])],
                    system_tokens=0, conv_id=c["conv_id"])
                dec = {name: ["large" if t["win"] >= th else "small" for t in c["turns"]]
                       for name, th in TH.items()}
                srate = float(np.mean([d == "large" for d in dec["router@50"]]))
                dec["random@match"] = ["large" if rng.random() < srate else "small"
                                       for _ in c["turns"]]
                dec["fixed-small"] = ["small"] * len(c["turns"])
                dec["fixed-large"] = ["large"] * len(c["turns"])
                for pname, d in dec.items():
                    bd = replay(traj, lambda i, tr, d=d: d[i], prices, caches).breakdown
                    rows.append({"dataset": ds_name, "provider": provider, "gap": gap,
                                 "policy": pname, "conv": c["conv_id"],
                                 "cm0": bd.cm0, "cm1": bd.cm1,
                                 "switches": bd.switches, "hit": bd.cache_hit_rate,
                                 "turns": len(c["turns"])})

df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/gate_a_per_conv.csv", index=False)
print("per-conv rows:", df.shape)

# ---------------- aggregate, bootstrap, verdict ----------------
out = []
for (ds, prov, gap), g in df.groupby(["dataset", "provider", "gap"]):
    fl = g[g.policy == "fixed-large"].set_index("conv")[["cm0", "cm1"]]
    for pol, gg in g.groupby("policy"):
        if pol == "fixed-large":
            continue
        piv = gg.set_index("conv")[["cm0", "cm1"]].join(fl, rsuffix="_fl").dropna()
        S0 = 1 - piv.cm0.sum() / piv.cm0_fl.sum()
        S1 = 1 - piv.cm1.sum() / piv.cm1_fl.sum()
        infl = (S0 - S1) / S0 if S0 > 0 else np.nan
        bs = []
        arr = piv.to_numpy()
        for _ in range(1000):
            s = arr[np.random.randint(0, len(arr), len(arr))]
            b0 = 1 - s[:, 0].sum() / s[:, 2].sum()
            b1 = 1 - s[:, 1].sum() / s[:, 3].sum()
            bs.append((b0 - b1) / b0 if b0 > 0 else np.nan)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        out.append(dict(dataset=ds, provider=prov, gap=gap, policy=pol,
                        S_cm0=round(S0, 4), S_cm1=round(S1, 4),
                        inflation=round(infl, 4),
                        infl_ci=f"[{lo:.3f},{hi:.3f}]",
                        switches_mean=round(gg.switches.mean(), 2),
                        hit_mean=round(gg.hit.mean(), 3)))

summary = pd.DataFrame(out).sort_values(["dataset", "provider", "gap", "policy"])
summary.to_csv("/kaggle/working/gate_a_summary.csv", index=False)
print(summary.to_string(index=False))

routers = summary[summary.policy.str.startswith("router") & (summary.gap <= 120)]
max_row = routers.loc[routers.inflation.idxmax()]
verdict = ("GO" if (routers.inflation > 0.20).any()
           else "GO-REFRAME" if (routers.inflation > 0.10).any() else "NO-GO")
result = {"verdict": verdict,
          "max_inflation_row": max_row.to_dict(),
          "criteria": "GO >0.20 rel. overstatement | GO-REFRAME 0.10-0.20 | NO-GO <0.10",
          "n_convs": {"wildchat": len(wild), "lmsys": len(lmsys)},
          "frozen_history_caveat": "outputs fixed across models; Gate A is cost-mechanism only"}
json.dump(result, open("/kaggle/working/gate_a_verdict.json", "w"), indent=2, default=str)
print("\n==== GATE A VERDICT:", verdict, "====\n")
print(json.dumps(result, indent=2, default=str))
