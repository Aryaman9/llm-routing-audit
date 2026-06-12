"""Pricing snapshots: load dated provider pricing + cache parameters.

Every number used anywhere in the audit flows through a PricingSnapshot loaded
from a dated JSON file in pricing/. Snapshots are immutable records: when prices
change, add a new file, never edit an old one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelPrice:
    name: str
    provider: str
    p_in: float   # USD per 1M input tokens (uncached)
    p_out: float  # USD per 1M output tokens


@dataclass(frozen=True)
class CacheParams:
    read_multiplier: float          # cached input tokens cost read_multiplier * p_in
    write_multiplier: float         # newly-written cache tokens cost write_multiplier * p_in
    ttl_seconds: float              # cache lifetime since last access
    refreshes_on_read: bool         # whether a hit resets the TTL clock
    min_cacheable_prefix_tokens: int
    block_tokens: int               # cache hit lengths round down to this granularity
    explicit: bool                  # True = caller opts in and pays writes (Anthropic-style)
    verified_by_probe: bool = False


@dataclass(frozen=True)
class ProviderPricing:
    provider: str
    cache: CacheParams
    models: dict[str, ModelPrice] = field(default_factory=dict)


def load_snapshot(path: str | Path) -> dict[str, ProviderPricing]:
    """Load a pricing snapshot JSON into ProviderPricing objects keyed by provider."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, ProviderPricing] = {}
    for pname, p in raw["providers"].items():
        cache = CacheParams(
            read_multiplier=p["cache_read_multiplier"],
            write_multiplier=p["cache_write_multiplier"],
            ttl_seconds=p["cache_ttl_seconds"],
            refreshes_on_read=p.get("cache_ttl_refreshes_on_read", True),
            min_cacheable_prefix_tokens=p["min_cacheable_prefix_tokens"],
            block_tokens=p["cache_block_tokens"],
            explicit=p["caching_is_explicit"],
            verified_by_probe=p.get("verified_by_probe", False),
        )
        models = {
            mname: ModelPrice(name=mname, provider=pname, p_in=m["p_in"], p_out=m["p_out"])
            for mname, m in p["models"].items()
        }
        out[pname] = ProviderPricing(provider=pname, cache=cache, models=models)
    return out


def default_snapshot_path() -> Path:
    """Most recent snapshot in the repo's pricing/ directory (sorted by filename date)."""
    pricing_dir = Path(__file__).resolve().parents[2] / "pricing"
    snaps = sorted(pricing_dir.glob("snapshot_*.json"))
    if not snaps:
        raise FileNotFoundError(f"no pricing snapshots in {pricing_dir}")
    return snaps[-1]
