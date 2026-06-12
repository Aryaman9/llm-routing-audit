"""Semantic tests for the prefix-cache simulator and CM0/CM1 cost engines.

Each test pins down one billing behavior the audit depends on. If a live-API
probe (E1) contradicts any of these semantics, the test gets updated WITH a
comment citing the probe evidence - never silently.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from realcost import (CacheParams, ModelPrice, PrefixCacheSim, Trajectory,
                      Turn, alternate, cm1_call_cost, fixed, replay,
                      switch_at)

# --- fixtures: a no-write-premium provider (OpenAI-style) and an explicit one ---

AUTO = CacheParams(read_multiplier=0.5, write_multiplier=1.0, ttl_seconds=300,
                   refreshes_on_read=True, min_cacheable_prefix_tokens=1024,
                   block_tokens=128, explicit=False)
EXPL = CacheParams(read_multiplier=0.1, write_multiplier=1.25, ttl_seconds=300,
                   refreshes_on_read=True, min_cacheable_prefix_tokens=1024,
                   block_tokens=1, explicit=True)

P_SMALL = ModelPrice("small", "test", p_in=1.0, p_out=5.0)
P_LARGE = ModelPrice("large", "test", p_in=10.0, p_out=50.0)


def make_sim(params):
    return PrefixCacheSim({"small": params, "large": params})


def test_first_call_auto_provider_equals_cm0():
    sim = make_sim(AUTO)
    acc = sim.account_call("small", 2000, 100, now=0.0)
    assert acc.cache_read_tokens == 0
    assert acc.uncached_tokens == 2000
    c = cm1_call_cost(acc, P_SMALL, AUTO)
    assert abs(c["total"] - (2000 * 1.0 + 100 * 5.0) / 1e6) < 1e-12


def test_second_turn_same_model_hits_cache():
    sim = make_sim(AUTO)
    sim.account_call("small", 2048, 100, now=0.0)
    acc = sim.account_call("small", 3000, 100, now=30.0)
    # hit = 2048 (cached prompt), block-floored to 128s -> 2048
    assert acc.cache_read_tokens == 2048
    assert acc.uncached_tokens == 3000 - 2048


def test_model_switch_forfeits_cache():
    sim = make_sim(AUTO)
    sim.account_call("small", 4096, 100, now=0.0)
    acc = sim.account_call("large", 5000, 100, now=30.0)  # switch: cold cache
    assert acc.cache_read_tokens == 0
    assert acc.uncached_tokens == 5000


def test_ttl_expiry_full_price():
    sim = make_sim(AUTO)
    sim.account_call("small", 4096, 100, now=0.0)
    acc = sim.account_call("small", 5000, 100, now=400.0)  # 400s > 300s TTL
    assert acc.cache_read_tokens == 0


def test_oscillation_rehits_older_shorter_prefix():
    sim = make_sim(AUTO)
    sim.account_call("small", 2048, 100, now=0.0)   # A
    sim.account_call("large", 4096, 100, now=30.0)  # B (A's entry ages, keeps 2048)
    acc = sim.account_call("small", 6016, 100, now=60.0)  # back to A within TTL
    assert acc.cache_read_tokens == 2048
    assert acc.uncached_tokens == 6016 - 2048


def test_explicit_provider_write_premium():
    sim = make_sim(EXPL)
    acc = sim.account_call("small", 2000, 100, now=0.0)
    assert acc.cache_write_tokens == 2000 and acc.uncached_tokens == 0
    c = cm1_call_cost(acc, P_SMALL, EXPL)
    # first call costs MORE than CM0: write premium 1.25x
    assert abs(c["total"] - (2000 * 1.0 * 1.25 + 100 * 5.0) / 1e6) < 1e-12


def test_min_prefix_threshold_blocks_tiny_prompts():
    sim = make_sim(AUTO)
    sim.account_call("small", 500, 50, now=0.0)   # below 1024: not cached
    acc = sim.account_call("small", 800, 50, now=10.0)
    assert acc.cache_read_tokens == 0


def test_block_rounding():
    sim = make_sim(AUTO)
    sim.account_call("small", 2000, 100, now=0.0)
    acc = sim.account_call("small", 2500, 100, now=10.0)
    # 2000 floored to 128-multiple = 1920
    assert acc.cache_read_tokens == 1920


# --- replay-level invariants ---

def _toy_traj(n_turns=10, user=100, out=300, sys_tokens=1500, gap=30.0):
    return Trajectory(
        turns=[Turn(user_tokens=user, t_seconds=i * gap, output_tokens=out)
               for i in range(n_turns)],
        system_tokens=sys_tokens,
    )


PRICES = {"small": P_SMALL, "large": P_LARGE}
CACHES_AUTO = {"small": AUTO, "large": AUTO}


def test_replay_alternating_costs_more_cm1_than_sticky_large():
    """The audit's core mechanism: under CM1 an oscillating router re-prefills
    constantly; under CM0 the same router looks cheaper than fixed-large."""
    traj = _toy_traj(n_turns=12)
    alt = replay(traj, alternate(["small", "large"]), PRICES, CACHES_AUTO).breakdown
    big = replay(traj, fixed("large"), PRICES, CACHES_AUTO).breakdown
    assert alt.cm0 < big.cm0                       # idealized: router wins
    assert alt.cache_hit_rate < big.cache_hit_rate  # mechanism: lost cache hits
    assert (alt.cm1 / alt.cm0) > (big.cm1 / big.cm0)  # router's real/ideal ratio is worse


def test_oscillation_penalty_is_ttl_regime_dependent():
    """Discovered during scaffolding (2026-06-11): with per-model cache state,
    A->B->A oscillation re-hits each model's surviving cache when the revisit
    interval (2x turn gap) is inside the TTL - so 'switching kills the cache'
    is only true in the short-TTL regime. Both regimes pinned here; this
    TTL-dependence is itself an audit finding to quantify on real traces."""
    # Short-TTL regime: revisit interval 60s > TTL 45s > same-model gap 30s
    # -> sticky policies keep hits, oscillation gets none, oscillation loses.
    short = CacheParams(read_multiplier=0.5, write_multiplier=1.0, ttl_seconds=45,
                        refreshes_on_read=True, min_cacheable_prefix_tokens=1024,
                        block_tokens=128, explicit=False)
    caches_short = {"small": short, "large": short}
    traj = _toy_traj(n_turns=12, gap=30.0)
    once = replay(traj, switch_at("small", "large", 6), PRICES, caches_short).breakdown
    alt = replay(traj, alternate(["small", "large"]), PRICES, caches_short).breakdown
    assert once.switches == 1 and alt.switches == 11
    assert alt.cache_hit_rate == 0.0
    assert once.cache_hit_rate > 0.5
    assert once.cm1 < alt.cm1

    # Long-TTL regime (TTL 300s): oscillation's per-model caches survive the
    # 60s revisit, hit rate stays high, and the absolute-cost ranking can even
    # flip depending on which model's prices the cached reads land on.
    alt_long = replay(traj, alternate(["small", "large"]), PRICES, CACHES_AUTO).breakdown
    assert alt_long.cache_hit_rate > 0.5


def test_replay_cm0_independent_of_timestamps():
    a = replay(_toy_traj(gap=10.0), fixed("small"), PRICES, CACHES_AUTO).breakdown
    b = replay(_toy_traj(gap=999.0), fixed("small"), PRICES, CACHES_AUTO).breakdown
    assert abs(a.cm0 - b.cm0) < 1e-12
    assert a.cm1 < b.cm1  # TTL expiry makes slow conversations pricier under CM1
