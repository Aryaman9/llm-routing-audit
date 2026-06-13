"""Bootstrap confidence intervals for Gate A overstatement claims.

Consumes the per-conversation cost arrays dumped by gate_a_replay.py
(results/gate_a_perconv_{ds}_{date}.npz) and produces 95% CIs by resampling
CONVERSATIONS with replacement.

Why ratio-of-sums, resampled paired: savings = 1 - cost(policy)/cost(fixed-large)
is a ratio of TOTALS, not a mean of per-conversation ratios (a conversation
with 2x the tokens should count 2x toward the bill). So each bootstrap draw
resamples a set of conversation indices and recomputes the ratio of sums for
BOTH the policy and the fixed-large baseline on the SAME indices (paired) -
this is the statistically correct CI for an aggregate cost-savings claim.

Reported per (dataset, provider, policy):
  savings_cm0, savings_cm1   : point estimate + 95% CI (percentage)
  gap_pp                     : s0 - s1 in percentage POINTS + CI
  overstatement_pct          : (s0 - s1)/s0 * 100 + CI (relative inflation)

Run:  python experiments/gate_a_bootstrap.py [--date 2026-06-13] [--b 10000]
"""

import argparse
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("AUDIT_OUT_DIR", ROOT / "results"))
PROVIDERS = ("anthropic", "openai")


def ci(arr, lo=2.5, hi=97.5):
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def savings(num_sum, base_sum):
    return 1.0 - num_sum / base_sum if base_sum > 0 else np.nan


def analyze(npz_path, B, seed, report_lines):
    d = np.load(npz_path, allow_pickle=True)
    gap_s = float(d["gap_s"]) if "gap_s" in d else float("nan")
    n = len(d["conv_ids"])
    ds_name = npz_path.stem.replace("gate_a_perconv_", "").rsplit("_", 1)[0]
    rng = np.random.default_rng(seed)
    # one shared set of bootstrap index draws -> CIs are comparable across policies
    draws = rng.integers(0, n, size=(B, n))

    for prov in PROVIDERS:
        base0 = d[f"{prov}|fixed-large|cm0"]
        base1 = d[f"{prov}|fixed-large|cm1"]
        policies = sorted({k.split("|")[1] for k in d.files
                           if k.startswith(f"{prov}|") and k.endswith("|cm0")})
        header = (f"\n#### {ds_name} x {prov}  (n={n}, gap={gap_s:.0f}s, B={B})")
        report_lines.append(header)
        report_lines.append(f"{'policy':<26}{'S0% [95% CI]':>22}"
                            f"{'S1% [95% CI]':>22}{'overst% [95% CI]':>24}")

        # precompute bootstrapped baseline sums (paired, reused across policies)
        b0_boot = base0[draws].sum(axis=1)
        b1_boot = base1[draws].sum(axis=1)
        b0_pt, b1_pt = base0.sum(), base1.sum()

        for pname in policies:
            c0, c1 = d[f"{prov}|{pname}|cm0"], d[f"{prov}|{pname}|cm1"]
            s0_pt = savings(c0.sum(), b0_pt) * 100
            s1_pt = savings(c1.sum(), b1_pt) * 100
            ov_pt = (s0_pt - s1_pt) / s0_pt * 100 if abs(s0_pt) > 1e-6 else np.nan

            c0b = c0[draws].sum(axis=1); c1b = c1[draws].sum(axis=1)
            s0b = (1 - c0b / b0_boot) * 100
            s1b = (1 - c1b / b1_boot) * 100
            with np.errstate(divide="ignore", invalid="ignore"):
                ovb = np.where(np.abs(s0b) > 1e-6, (s0b - s1b) / s0b * 100, np.nan)

            s0lo, s0hi = ci(s0b); s1lo, s1hi = ci(s1b)
            ovlo, ovhi = ci(ovb[~np.isnan(ovb)]) if np.any(~np.isnan(ovb)) else (np.nan, np.nan)
            report_lines.append(
                f"{pname:<26}"
                f"{f'{s0_pt:5.1f} [{s0lo:5.1f},{s0hi:5.1f}]':>22}"
                f"{f'{s1_pt:5.1f} [{s1lo:5.1f},{s1hi:5.1f}]':>22}"
                f"{f'{ov_pt:6.1f} [{ovlo:6.1f},{ovhi:6.1f}]':>24}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="date stamp on perconv files")
    ap.add_argument("--b", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pattern = f"gate_a_perconv_*_{args.date}.npz" if args.date else "gate_a_perconv_*.npz"
    files = sorted(OUT_DIR.glob(pattern))
    if not files:
        raise SystemExit(f"no per-conv files matching {pattern} in {OUT_DIR}")

    lines = [f"GATE A bootstrap CIs  (files: {[f.name for f in files]})"]
    for f in files:
        analyze(f, args.b, args.seed, lines)
    text = "\n".join(lines)
    print(text)
    out = OUT_DIR / f"gate_a_bootstrap_ci{('_' + args.date) if args.date else ''}.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
