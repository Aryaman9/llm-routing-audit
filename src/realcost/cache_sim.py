"""Prefix-cache simulator: per-(conversation, model) cache state with TTL.

Semantics modeled (calibrated against provider docs; verified by live probes in E1):

- Conversations are append-only token streams. The prompt at turn t is the full
  history; the prompt at turn t+1 extends it (prev prompt + prev output + new user msg).
- A cache entry belongs to ONE model. It covers the prompt prefix sent in the last
  call to that model, expires ttl_seconds after last access, and (optionally)
  refreshes its TTL on every hit.
- Hits require prefix length >= min_cacheable_prefix_tokens and round down to
  block_tokens granularity.
- Switching models gives a cold cache on the new model, while the old model's
  entry keeps aging - so A->B->A within TTL partially re-hits A's older, shorter
  prefix. Oscillation costs fall out of the state machine naturally.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pricing import CacheParams


@dataclass
class _Entry:
    cached_len: int = 0          # tokens of prompt prefix currently cached
    last_access: float = float("-inf")


@dataclass
class CallAccounting:
    """Token-level breakdown of a single call under cache-aware accounting."""
    model: str
    input_tokens: int
    cache_read_tokens: int       # billed at read_multiplier * p_in
    cache_write_tokens: int      # billed at write_multiplier * p_in (explicit caching)
    uncached_tokens: int         # billed at p_in (providers without write premium)
    output_tokens: int

    @property
    def hit_fraction(self) -> float:
        return self.cache_read_tokens / self.input_tokens if self.input_tokens else 0.0


class PrefixCacheSim:
    """Tracks per-model prefix caches for one conversation/trajectory."""

    def __init__(self, cache_params: dict[str, CacheParams]):
        # cache_params: model name -> CacheParams of its provider
        self.params = cache_params
        self.entries: dict[str, _Entry] = {m: _Entry() for m in cache_params}

    def _live_cached_len(self, model: str, now: float) -> int:
        e = self.entries[model]
        p = self.params[model]
        if now - e.last_access > p.ttl_seconds:
            return 0
        return e.cached_len

    def account_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        now: float,
        write_cache: bool = True,
    ) -> CallAccounting:
        """Account one call: split input tokens into read/write/uncached, update state.

        write_cache: for explicit-caching providers, whether the caller pays to
        extend the cache to cover this prompt (standard practice in multi-turn).
        """
        p = self.params[model]
        e = self.entries[model]

        live = self._live_cached_len(model, now)
        hit = min(live, input_tokens)
        # block granularity + minimum prefix threshold
        hit = (hit // p.block_tokens) * p.block_tokens
        if hit < p.min_cacheable_prefix_tokens:
            hit = 0

        fresh = input_tokens - hit
        if p.explicit and write_cache:
            read_t, write_t, unc_t = hit, fresh, 0
        else:
            read_t, write_t, unc_t = hit, 0, fresh

        # state update: cache now covers this prompt (if written / automatic),
        # and the TTL clock restarts on access
        covers_now = input_tokens if (not p.explicit or write_cache) else min(live, input_tokens)
        if covers_now >= p.min_cacheable_prefix_tokens:
            e.cached_len = max(e.cached_len if now - e.last_access <= p.ttl_seconds else 0,
                               covers_now)
            e.last_access = now
        elif p.refreshes_on_read and hit > 0:
            e.last_access = now

        return CallAccounting(
            model=model,
            input_tokens=input_tokens,
            cache_read_tokens=read_t,
            cache_write_tokens=write_t,
            uncached_tokens=unc_t,
            output_tokens=output_tokens,
        )
