# llm-routing-audit

**When Do LLM Routers Actually Save Money? Auditing Routing Cost Claims Under Real Serving Economics.**

Published LLM routing/cascading papers report cost savings computed under linear per-token
list pricing (`CM0`). Real serving economics include prompt-cache discounts (10–50x cheaper
cached reads), cache-write premiums, cache TTLs, and — critically — **full cache forfeiture
whenever the router switches models mid-conversation**. This repo audits whether published
savings survive realistic accounting (`CM1`: provider API pricing with caching; `CM2`:
self-hosted measured GPU cost), across single-turn, multi-turn, and agentic workloads.

Project plan: see `PROJECT_PLAN.md` in the parent folder.

## Layout

```
pricing/            dated pricing snapshots (JSON, with sources; never edited in place)
src/realcost/       core library: pricing, prefix-cache simulator, CM0/CM1 cost engines, replay
tests/              unit tests for cache & cost semantics
experiments/        numbered experiment scripts (e0_demo.py = mechanism demo, no API needed)
data/               local datasets (gitignored)
```

## Quickstart

```
pip install -r requirements.txt
python -m pytest tests/ -q          # verify cache/cost semantics
python experiments/e0_demo.py      # see CM0-vs-CM1 divergence on synthetic workloads
```

## Integrity rules (non-negotiable)

- Every pricing number carries a snapshot date and source URL; `verified_by_probe: false`
  until confirmed by live API probes (experiment E1/P1).
- All policy comparisons run on identical traces; bootstrap CIs; metrics frozen before results.
- Negative and hypothesis-contradicting results are reported, never buried.
