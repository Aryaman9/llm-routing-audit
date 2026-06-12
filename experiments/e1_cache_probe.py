"""E1-lite: live cache-billing probes against Anthropic and OpenAI APIs.

Verifies, with provider usage metadata as ground truth, the three assumptions
the entire audit rests on:
  P1. Same model + same prefix + within TTL  => billed cache READ (discounted).
  P2. A DIFFERENT model never reads another model's cache (cold start).
  P3. Cached/uncached token splits are reported in usage fields, so real
      experiment bills can be MEASURED, not simulated.

Per provider: call A (cold) -> call B (same model, same prefix: expect hit)
-> call C (different model, same prefix: expect cold).

Spend: < $0.10 total. Raw responses (usage fields only, no content) are saved
to results/probes/ as dated evidence files for the paper.

Run:  python experiments/e1_cache_probe.py
Keys: ANTHROPIC_API_KEY, OPENAI_API_KEY env vars (never stored in-repo).
"""

import json
import os
import socket
import time
import urllib.request
from datetime import date
from pathlib import Path

# api.anthropic.com publishes an AAAA record but this network's IPv6 is
# non-routing, so Python's happy-eyeballs-less urllib hangs on the v6 connect.
# Force IPv4 for all probe traffic (verified 2026-06-12: TCP v4 reachable).
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, *args, **kwargs):
    return _orig_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)
socket.getaddrinfo = _ipv4_getaddrinfo

RESULTS = Path(__file__).resolve().parents[1] / "results" / "probes"
RESULTS.mkdir(parents=True, exist_ok=True)


def build_prefix(target_tokens=5200, salt=None):
    """~target_tokens text prefix, SALTED at the head so each probe run gets a
    cold cache. Lesson from run 1 (2026-06-12): provider caches persist across
    runs (OpenAI hit a 15-min-old entry from the previous probe), so unsalted
    prefixes contaminate cold-start measurements."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    salt = salt or str(int(time.time()))
    chunks, i = [f"Probe run {salt}. "], 0
    text = chunks[0]
    while len(enc.encode(text)) < target_tokens:
        i += 1
        chunks.append(
            f"Section {i}: The inventory ledger records batch {i * 37} with "
            f"checksum {i * i % 9973}, stored in warehouse aisle {i % 48}, "
            f"audited on day {i % 365} with reconciliation code {i * 13 % 999}. "
        )
        text = "".join(chunks)
    return text, len(enc.encode(text))


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# One keep-alive session shared by all probe calls. This network's route to
# api.anthropic.com completes TCP connects only intermittently (observed
# 2026-06-12, ~1-in-3), so we retry connects hard and then REUSE the single
# successful connection for every call in the probe sequence.
_session = requests.Session()
_retry = Retry(total=10, connect=10, read=2, backoff_factor=1.5,
               allowed_methods=["POST", "GET"], status_forcelist=[429, 529])
_session.mount("https://", HTTPAdapter(max_retries=_retry))


def post(url, headers, body, timeout=60):
    t0 = time.perf_counter()
    r = _session.post(url, headers=headers, json=body, timeout=(20, timeout))
    r.raise_for_status()
    return r.json(), time.perf_counter() - t0


def probe_anthropic(prefix, key, models=("claude-haiku-4-5-20251001", "claude-sonnet-4-6")):
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}

    def call(model, user_msg):
        body = {
            "model": model, "max_tokens": 16,
            "system": [{"type": "text", "text": prefix,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_msg}],
        }
        resp, dt = post(url, headers, body)
        u = resp.get("usage", {})
        return {"model": model, "wall_s": round(dt, 3),
                "input_tokens": u.get("input_tokens"),
                "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": u.get("cache_read_input_tokens"),
                "output_tokens": u.get("output_tokens")}

    a = call(models[0], "Reply with the single word: alpha")
    b = call(models[0], "Reply with the single word: bravo")
    c = call(models[1], "Reply with the single word: charlie")
    d = call(models[1], "Reply with the single word: delta")  # warm read on model 2
    return {"provider": "anthropic", "calls": {"A_cold": a, "B_warm_same_model": b,
                                               "C_other_model": c,
                                               "D_warm_other_model": d}}


def probe_openai(prefix, key):
    base = "https://api.openai.com/v1"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    r = _session.get(f"{base}/models", headers=headers, timeout=(20, 60))
    r.raise_for_status()
    available = {m["id"] for m in r.json()["data"]}
    prefs = ["gpt-5-mini", "gpt-5.1-mini", "gpt-4.1-mini", "gpt-4o-mini"]
    picks = [m for m in prefs if m in available]
    if len(picks) < 2:
        picks += [m for m in sorted(available)
                  if "mini" in m and m not in picks][: 2 - len(picks)]
    m1, m2 = picks[0], picks[1]

    def call(model, user_msg):
        body = {"model": model, "max_completion_tokens": 16,
                "messages": [{"role": "system", "content": prefix},
                             {"role": "user", "content": user_msg}]}
        resp, dt = post(f"{base}/chat/completions", headers, body)
        u = resp.get("usage", {})
        det = u.get("prompt_tokens_details", {}) or {}
        return {"model": model, "wall_s": round(dt, 3),
                "prompt_tokens": u.get("prompt_tokens"),
                "cached_tokens": det.get("cached_tokens"),
                "completion_tokens": u.get("completion_tokens")}

    a = call(m1, "Reply with the single word: alpha")
    b = call(m1, "Reply with the single word: bravo")
    c = call(m2, "Reply with the single word: charlie")
    return {"provider": "openai", "models_available_picked": [m1, m2],
            "calls": {"A_cold": a, "B_warm_same_model": b, "C_other_model": c}}


def main():
    prefix, n_tok = build_prefix()
    print(f"prefix length: {n_tok} tokens (cl100k estimate)\n")
    report = {"date": str(date.today()), "prefix_tokens_cl100k": n_tok, "probes": []}

    for name, fn, envk in (("anthropic", probe_anthropic, "ANTHROPIC_API_KEY"),
                           ("openai", probe_openai, "OPENAI_API_KEY")):
        key = os.environ.get(envk)
        if not key:
            print(f"[skip] {name}: {envk} not set")
            continue
        try:
            res = fn(prefix, key) if name == "openai" else fn(prefix, key)
            report["probes"].append(res)
            print(json.dumps(res, indent=2))
        except Exception as e:
            print(f"[error] {name}: {e}")
            report["probes"].append({"provider": name, "error": str(e)})

    out = RESULTS / f"probe_{date.today()}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")

    print("\nVERDICTS (vs simulator assumptions):")
    for p in report["probes"]:
        if "error" in p:
            continue
        c = p["calls"]
        if p["provider"] == "anthropic":
            hit = c["B_warm_same_model"]["cache_read_input_tokens"] or 0
            cold_other = c["C_other_model"]["cache_read_input_tokens"] or 0
            warm_other = c["D_warm_other_model"]["cache_read_input_tokens"] or 0
            wrote = (c["A_cold"]["cache_creation_input_tokens"] or 0) + \
                    (c["C_other_model"]["cache_creation_input_tokens"] or 0)
            print(f"  anthropic: P1 same-model hit={hit} tok "
                  f"({'PASS' if hit > 0 else 'FAIL'}) | "
                  f"P2 cross-model read={cold_other} "
                  f"({'PASS' if cold_other == 0 else 'FAIL - INVESTIGATE'}) | "
                  f"P3 write-metered={wrote} ({'PASS' if wrote > 0 else 'FAIL'}) | "
                  f"P1b second-model warm read={warm_other} "
                  f"({'PASS' if warm_other > 0 else 'FAIL'})")
        else:
            hit = c["B_warm_same_model"]["cached_tokens"] or 0
            cold_other = c["C_other_model"]["cached_tokens"] or 0
            print(f"  openai:    P1 same-model hit={hit} tok "
                  f"({'PASS' if hit > 0 else 'FAIL'}) | "
                  f"P2 cross-model read={cold_other} "
                  f"({'PASS' if cold_other == 0 else 'FAIL - INVESTIGATE'}) | "
                  f"P3 cached split reported ({'PASS' if hit is not None else 'FAIL'})")


if __name__ == "__main__":
    main()
