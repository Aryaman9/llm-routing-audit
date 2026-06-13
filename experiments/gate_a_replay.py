"""GATE A (Phase 0): does the CM0-vs-CM1 gap have teeth on REAL conversations
with a REAL published router?

Replays the RouteLLM BERT router (routellm/bert_gpt4_augmented, the artifact
behind the canonical "saves up to 85%" claim) over WildChat/LMSYS multi-turn
conversations, routing each user turn between a small and a large model, and
accounts cost under CM0 (linear list price, what papers report) and CM1
(prompt-cache-aware provider economics, measured in E1 probes).

Frozen-history caveat (stated in PROJECT_PLAN.md): assistant outputs are the
logged ones regardless of routed model; Gate A audits COST accounting only,
quality matrices come in Phase 3. Output token counts are identical across
policies, so cost differences are pure accounting + routing structure.

GO/NO-GO (from PROJECT_PLAN.md Phase 0.4):
  overstatement of router savings >20% on multi-turn  -> GO
  10-20%                                              -> GO, reframed emphasis
  <10% everywhere                                     -> STOP, pivot

Run:  python experiments/gate_a_replay.py [--limit 300] [--gap-s 30]
"""

import argparse
import functools
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realcost import Trajectory, Turn, load_snapshot, replay  # noqa: E402
from realcost.pricing import default_snapshot_path  # noqa: E402

# Env overrides so the same script runs locally and on Kaggle (read-only input
# dirs, writable /kaggle/working).
DATA_DIR = Path(os.environ.get("AUDIT_DATA_DIR", ROOT / "data"))
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", ROOT / "results"))

print = functools.partial(print, flush=True)  # background runs: never buffer

ROUTER_ID = "routellm/bert_gpt4_augmented"
TARGET_STRONG_RATES = [0.1, 0.3, 0.5, 0.7, 0.9]
SYSTEM_TOKENS = 1000  # typical deployed-assistant system prompt; sensitivity later


# ---------------------------------------------------------------- data loading

def load_convs(path, limit=None):
    """JSONL -> list of conversations as (user,assistant) turn pairs."""
    convs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # merge consecutive same-role messages, then pair user->assistant
            merged = []
            for t in rec["turns"]:
                if merged and merged[-1]["role"] == t["role"]:
                    merged[-1]["n_tokens_cl100k"] += t["n_tokens_cl100k"]
                    merged[-1]["text"] = (merged[-1]["text"] or "") + "\n" + (t["text"] or "")
                else:
                    merged.append(dict(t))
            pairs = []
            i = 0
            while i + 1 < len(merged):
                if merged[i]["role"] == "user" and merged[i + 1]["role"] == "assistant":
                    pairs.append((merged[i], merged[i + 1]))
                    i += 2
                else:
                    i += 1
            if len(pairs) >= 2:
                convs.append({"conv_id": rec["conv_id"], "pairs": pairs})
            if limit and len(convs) >= limit:
                break
    return convs


def parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def to_trajectory(conv, default_gap_s):
    """Build a replay Trajectory; real timestamps if present, else fixed gaps."""
    t0, times = None, []
    for u, a in conv["pairs"]:
        # WildChat stamps the ASSISTANT message (response time); the gap between
        # consecutive responses is exactly the inter-call gap cache TTLs see.
        ts = parse_ts(a.get("ts")) or parse_ts(u.get("ts"))
        times.append(ts)
    use_real = all(t is not None for t in times) and len(times) > 1
    turns = []
    for k, (u, a) in enumerate(conv["pairs"]):
        if use_real:
            if t0 is None:
                t0 = times[k]
            t_rel = times[k] - t0
        else:
            t_rel = k * default_gap_s
        turns.append(Turn(user_tokens=max(u["n_tokens_cl100k"], 1),
                          t_seconds=t_rel,
                          output_tokens=max(a["n_tokens_cl100k"], 1)))
    return Trajectory(turns=turns, system_tokens=SYSTEM_TOKENS,
                      conv_id=conv["conv_id"]), use_real


# ---------------------------------------------------------------- router

def score_conversations(convs, ds_name, batch_size=None):
    """RouteLLM BERT router scores for every user turn, batched, GPU if
    available, checkpointed to OUT_DIR every ~1024 texts so a killed run
    resumes instead of restarting."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = batch_size or (128 if device == "cuda" else 32)
    tok = AutoTokenizer.from_pretrained(ROUTER_ID)
    model = AutoModelForSequenceClassification.from_pretrained(ROUTER_ID).to(device)
    model.eval()
    if device == "cuda":
        # Some Kaggle images ship torch builds without kernels for the
        # allocated GPU (cudaErrorNoKernelImageForDevice) - smoke-test one
        # forward pass and fall back to CPU instead of dying mid-run.
        try:
            with torch.no_grad():
                model(**tok(["smoke test"], return_tensors="pt").to(device))
        except Exception as e:  # noqa: BLE001
            print(f"CUDA unusable ({type(e).__name__}); falling back to CPU")
            device, batch_size = "cpu", 32
            model = model.to(device)
    print(f"router loaded: {ROUTER_ID} | device={device} | batch={batch_size} "
          f"| labels: {model.config.id2label}")

    texts, owners = [], []
    for ci, conv in enumerate(convs):
        for pi, (u, _) in enumerate(conv["pairs"]):
            texts.append((u["text"] or "")[:4000])
            owners.append((ci, pi))

    ckpt = OUT_DIR / f"router_scores_{ds_name}_{len(texts)}.npz"
    done = 0
    probs = np.zeros((len(texts), model.config.num_labels), dtype=np.float64)
    if ckpt.exists():
        saved = np.load(ckpt)
        if saved["probs"].shape == probs.shape:
            probs, done = saved["probs"], int(saved["done"])
            print(f"  resumed from checkpoint: {done}/{len(texts)} already scored")

    with torch.no_grad():
        for s in range(done, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            ins = tok(batch, return_tensors="pt", truncation=True,
                      max_length=512, padding=True).to(device)
            logits = model(**ins).logits
            probs[s:s + len(batch)] = torch.softmax(logits, dim=-1).cpu().numpy()
            if (s - done) % 1024 < batch_size:
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                np.savez(ckpt, probs=probs, done=s + len(batch))
                print(f"  scored {s + len(batch)}/{len(texts)} (checkpointed)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt, probs=probs, done=len(texts))

    # Label-convention sanity check, done BEHAVIORALLY: class-0 probability
    # should differ systematically between trivially-easy and clearly-hard
    # prompts. We define strong_score so that HARD prompts score HIGHER.
    easy = ["hi", "what is 2+2?", "thanks!", "what color is the sky?"]
    hard = ["Prove that there are infinitely many primes p such that p+2 is "
            "a sum of two squares, and give the asymptotic density.",
            "Implement a lock-free MPMC queue in C++20 with hazard pointers; "
            "explain the ABA mitigation and memory ordering choices.",
            "Derive the posterior for a hierarchical Dirichlet process mixture "
            "under a stick-breaking construction with truncation error bounds."]
    with torch.no_grad():
        pe = torch.softmax(model(**tok(easy, return_tensors="pt", truncation=True,
                                       max_length=512, padding=True)).logits, -1).numpy()
        ph = torch.softmax(model(**tok(hard, return_tensors="pt", truncation=True,
                                       max_length=512, padding=True)).logits, -1).numpy()
    flip = pe[:, 0].mean() > ph[:, 0].mean()
    strong = (1 - probs[:, 0]) if flip else probs[:, 0]
    print(f"label sanity: easy p0={pe[:, 0].mean():.3f} hard p0={ph[:, 0].mean():.3f} "
          f"-> strong_score = {'1 - p0' if flip else 'p0'}")

    scores = [[0.0] * len(c["pairs"]) for c in convs]
    for (ci, pi), s in zip(owners, strong):
        scores[ci][pi] = float(s)
    return scores


# ---------------------------------------------------------------- policies

def seq_policy(decisions):
    return lambda i, traj: decisions[i]


def make_policies(conv_scores, thresholds, rng):
    """Per-conversation decision sequences for every audited policy."""
    n = len(conv_scores)
    pol = {}
    pol["fixed-small"] = [["small"] * len(s) for s in conv_scores]
    pol["fixed-large"] = [["large"] * len(s) for s in conv_scores]
    for rate, th in thresholds.items():
        pol[f"routellm@{int(rate * 100)}%strong"] = [
            ["large" if x >= th else "small" for x in s] for s in conv_scores]
        # sticky variant: one routing decision per conversation (first turn)
        pol[f"sticky-routellm@{int(rate * 100)}%strong"] = [
            ["large" if s[0] >= th else "small"] * len(s) for s in conv_scores]
        pol[f"random@{int(rate * 100)}%strong"] = [
            ["large" if rng.random() < rate else "small" for _ in s]
            for s in conv_scores]
    return pol


# ---------------------------------------------------------------- main

def main():
    if sys.platform == "win32":
        # keep the system awake while the run is alive (display may sleep);
        # cleared automatically when the process exits
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="convs per dataset")
    ap.add_argument("--gap-s", type=float, default=30.0,
                    help="synthetic inter-turn gap when no timestamps")
    args = ap.parse_args()

    snap = load_snapshot(default_snapshot_path())
    rng = np.random.default_rng(0)
    report = {"date": str(date.today()), "router": ROUTER_ID,
              "system_tokens": SYSTEM_TOKENS, "gap_s": args.gap_s,
              "datasets": {}}

    for ds_name in ("wildchat", "lmsys"):
        path = DATA_DIR / f"{ds_name}_sample.jsonl"
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        convs = load_convs(path, args.limit)
        print(f"\n#### {ds_name}: {len(convs)} conversations")
        scores = score_conversations(convs, ds_name)
        flat = np.concatenate([np.array(s) for s in scores])
        thresholds = {r: float(np.quantile(flat, 1 - r)) for r in TARGET_STRONG_RATES}
        trajs, used_real = [], 0
        for c in convs:
            tr, real = to_trajectory(c, args.gap_s)
            trajs.append(tr)
            used_real += int(real)
        print(f"real timestamps used: {used_real}/{len(trajs)}")

        policies = make_policies(scores, thresholds, rng)
        ds_out = {"n_convs": len(trajs), "real_ts_convs": used_real, "providers": {}}
        # per-conversation cost arrays for bootstrap CIs (gate_a_bootstrap.py)
        per_conv = {"conv_ids": np.array([t.conv_id for t in trajs])}

        for prov in ("anthropic", "openai"):
            pp = snap[prov]
            prices = {"small": pp.models["small"], "large": pp.models["large"]}
            caches = {m: pp.cache for m in prices}
            rows = {}
            for pname, decisions in policies.items():
                c0 = np.empty(len(trajs)); c1 = np.empty(len(trajs))
                sw = hits = ins = 0.0
                for j, (tr, dec) in enumerate(zip(trajs, decisions)):
                    bd = replay(tr, seq_policy(dec), prices, caches).breakdown
                    c0[j] = bd.cm0; c1[j] = bd.cm1; sw += bd.switches
                    hits += bd.cached_tokens; ins += bd.input_tokens
                per_conv[f"{prov}|{pname}|cm0"] = c0
                per_conv[f"{prov}|{pname}|cm1"] = c1
                rows[pname] = {"cm0": float(c0.sum()), "cm1": float(c1.sum()),
                               "switches_per_conv": sw / len(trajs),
                               "cache_hit_rate": hits / ins if ins else 0}
            base0, base1 = rows["fixed-large"]["cm0"], rows["fixed-large"]["cm1"]
            print(f"\n--- {ds_name} x {prov} ---")
            print(f"{'policy':<28}{'CM0$':>9}{'CM1$':>9}{'S0%':>7}{'S1%':>7}"
                  f"{'overst%':>9}{'sw/conv':>8}{'hit%':>6}")
            for pname, r in rows.items():
                s0 = 1 - r["cm0"] / base0
                s1 = 1 - r["cm1"] / base1
                ov = (s0 - s1) / s0 * 100 if s0 > 1e-9 else float("nan")
                r["savings_cm0"], r["savings_cm1"] = s0, s1
                r["overstatement_pct"] = ov
                print(f"{pname:<28}{r['cm0']:>9.2f}{r['cm1']:>9.2f}"
                      f"{s0 * 100:>7.1f}{s1 * 100:>7.1f}{ov:>9.1f}"
                      f"{r['switches_per_conv']:>8.2f}"
                      f"{r['cache_hit_rate'] * 100:>6.1f}")
            ds_out["providers"][prov] = rows
        report["datasets"][ds_name] = ds_out

        pc_path = OUT_DIR / f"gate_a_perconv_{ds_name}_{date.today()}.npz"
        np.savez_compressed(pc_path, gap_s=args.gap_s, **per_conv)
        print(f"per-conv arrays -> {pc_path}")

    out = OUT_DIR / f"gate_a_{date.today()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")

    # ------- verdict -------
    worst = 0.0
    for ds in report["datasets"].values():
        for prov in ds["providers"].values():
            for pname, r in prov.items():
                if pname.startswith("routellm@") and r.get("savings_cm0", 0) > 0.02:
                    worst = max(worst, r["overstatement_pct"])
    verdict = "GO" if worst > 20 else ("GO (reframe: latency+regime map)"
                                       if worst > 10 else "STOP - pivot")
    print(f"\nGATE A: max router-savings overstatement = {worst:.1f}%  =>  {verdict}")


if __name__ == "__main__":
    main()
