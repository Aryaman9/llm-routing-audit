"""Cost models.

CM0 - idealized linear pricing, what the routing literature reports:
      cost = input_tokens * p_in + output_tokens * p_out, cache-free.
CM1 - provider API pricing with prompt caching: cached reads discounted,
      explicit-cache writes at a premium, TTL expiry, model switches forfeit cache.

Both operate on the same CallAccounting stream so any divergence is purely the
accounting model, never the workload.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cache_sim import CallAccounting
from .pricing import CacheParams, ModelPrice

_M = 1_000_000


@dataclass
class CostBreakdown:
    cm0: float = 0.0
    cm1: float = 0.0
    cm1_read: float = 0.0
    cm1_write: float = 0.0
    cm1_uncached: float = 0.0
    cm1_output: float = 0.0
    calls: int = 0
    switches: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0

    @property
    def overstatement(self) -> float:
        """How much CM0 overstates (or understates) true cost: cm1/cm0 - 1.
        Positive => real cost HIGHER than idealized; negative => caching makes
        real cost lower than papers assume."""
        return self.cm1 / self.cm0 - 1.0 if self.cm0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.input_tokens if self.input_tokens else 0.0


def cm0_call_cost(acc: CallAccounting, price: ModelPrice) -> float:
    return (acc.input_tokens * price.p_in + acc.output_tokens * price.p_out) / _M


def cm1_call_cost(acc: CallAccounting, price: ModelPrice, cache: CacheParams) -> dict[str, float]:
    read = acc.cache_read_tokens * price.p_in * cache.read_multiplier / _M
    write = acc.cache_write_tokens * price.p_in * cache.write_multiplier / _M
    unc = acc.uncached_tokens * price.p_in / _M
    out = acc.output_tokens * price.p_out / _M
    return {"read": read, "write": write, "uncached": unc, "output": out,
            "total": read + write + unc + out}


def accumulate(
    breakdown: CostBreakdown,
    acc: CallAccounting,
    price: ModelPrice,
    cache: CacheParams,
    switched: bool,
) -> CostBreakdown:
    c1 = cm1_call_cost(acc, price, cache)
    breakdown.cm0 += cm0_call_cost(acc, price)
    breakdown.cm1 += c1["total"]
    breakdown.cm1_read += c1["read"]
    breakdown.cm1_write += c1["write"]
    breakdown.cm1_uncached += c1["uncached"]
    breakdown.cm1_output += c1["output"]
    breakdown.calls += 1
    breakdown.switches += int(switched)
    breakdown.input_tokens += acc.input_tokens
    breakdown.cached_tokens += acc.cache_read_tokens
    return breakdown
