"""E0 mechanism demo (synthetic, zero API cost).

Shows CM0-vs-CM1 divergence for routing policies on two synthetic workloads:
  1. a chat-like conversation (modest context growth)
  2. an agent-like trajectory (large tool outputs -> fast context growth)

This is NOT evidence for the paper (synthetic). It demonstrates the mechanism
and exercises the full pipeline before real traces arrive (Gate A uses real
LMSYS/WildChat conversations + RouteLLM decisions).

Run:  python experiments/e0_demo.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from realcost import (Trajectory, Turn, alternate, default_snapshot_path,
                      fixed, load_snapshot, replay, switch_at)


def chat_traj(n_turns=12, gap_s=45.0, seed=0):
    rng = random.Random(seed)
    return Trajectory(
        turns=[Turn(user_tokens=rng.randint(40, 200), t_seconds=i * gap_s,
                    output_tokens=rng.randint(150, 450)) for i in range(n_turns)],
        system_tokens=1500, conv_id="chat-demo")


def agent_traj(n_steps=30, gap_s=8.0, seed=0):
    rng = random.Random(seed)
    # each step: short instruction + bulky tool output folded into context
    return Trajectory(
        turns=[Turn(user_tokens=rng.randint(400, 1200), t_seconds=i * gap_s,
                    output_tokens=rng.randint(80, 300)) for i in range(n_steps)],
        system_tokens=4000, conv_id="agent-demo")


def random_router(models, p_large=0.5, seed=1):
    rng = random.Random(seed)
    return lambda i, traj: models[1] if rng.random() < p_large else models[0]


def run(provider_name, provider, traj, label):
    prices = {"small": provider.models["small"], "large": provider.models["large"]}
    caches = {m: provider.cache for m in prices}
    policies = {
        "fixed-small": fixed("small"),
        "fixed-large": fixed("large"),
        "switch-once@mid": switch_at("small", "large", len(traj.turns) // 2),
        "random-50/50": random_router(["small", "large"]),
        "alternate-every-turn": alternate(["small", "large"]),
    }
    print(f"\n=== {label} | provider params: {provider_name} "
          f"(delta={provider.cache.read_multiplier}, ttl={provider.cache.ttl_seconds:.0f}s) ===")
    print(f"{'policy':<22}{'CM0 $':>10}{'CM1 $':>10}{'CM1/CM0':>9}{'switches':>9}{'hit%':>7}")
    for name, pol in policies.items():
        bd = replay(traj, pol, prices, caches).breakdown
        print(f"{name:<22}{bd.cm0:>10.5f}{bd.cm1:>10.5f}{bd.cm1 / bd.cm0:>9.2f}"
              f"{bd.switches:>9}{bd.cache_hit_rate * 100:>6.1f}%")


def main():
    snap = load_snapshot(default_snapshot_path())
    for pname in ("anthropic", "openai", "deepseek"):
        run(pname, snap[pname], chat_traj(), "CHAT workload (12 turns)")
        run(pname, snap[pname], agent_traj(), "AGENT workload (30 steps, bulky tool outputs)")
    print("\nReading: CM1/CM0 > 1 means real cost EXCEEDS idealized accounting; "
          "the gap between a switching policy's ratio and fixed-large's ratio is "
          "the overstatement the audit quantifies on real traces.")


if __name__ == "__main__":
    main()
