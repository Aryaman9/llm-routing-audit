"""Gate A data prep: sample multi-turn conversations from WildChat-1M and
LMSYS-Chat-1M into replay-ready JSONL.

Both datasets are gated: needs HF_TOKEN env var (or `hf auth login`) from an
account that has accepted each dataset's license.

Output (data/, gitignored - raw user conversations must never be committed):
  data/wildchat_sample.jsonl   - has per-turn timestamps -> TTL realism (E3b)
  data/lmsys_sample.jsonl      - scale/diversity; no per-turn timestamps

Each line: {conv_id, n_turns, turns: [{role, n_tokens_cl100k, ts|null}]}
Token counts only - we never persist message text we don't need at this stage
(quality judging in Phase 3 re-streams text for the selected subsample).

Run:  python scripts/download_data.py --per-dataset 2000 --min-turns 4
"""

import argparse
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=2000)
    ap.add_argument("--min-turns", type=int, default=4)  # user+assistant msgs
    args = ap.parse_args()

    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("cl100k_base")

    def ntok(s):
        return len(enc.encode(s, disallowed_special=()))

    jobs = [
        # (hf_id, conversation field, timestamp strategy)
        ("allenai/WildChat-1M", "conversation", "per_message"),
        ("lmsys/lmsys-chat-1m", "conversation", "none"),
    ]
    for hf_id, conv_field, ts_mode in jobs:
        short = hf_id.split("/")[1].split("-")[0].lower()
        out_path = DATA / f"{short}_sample.jsonl"
        print(f"\n=== {hf_id} -> {out_path} ===")
        ds = load_dataset(hf_id, split="train", streaming=True)
        kept = 0
        with out_path.open("w", encoding="utf-8") as f:
            for i, row in enumerate(ds):
                conv = row.get(conv_field) or []
                if len(conv) < args.min_turns:
                    continue
                lang = (row.get("language") or "").lower()
                if lang and lang != "english":
                    continue
                turns = []
                for m in conv:
                    ts = None
                    if ts_mode == "per_message":
                        raw_ts = m.get("timestamp")
                        ts = str(raw_ts) if raw_ts is not None else None
                    turns.append({"role": m.get("role"),
                                  "n_tokens_cl100k": ntok(m.get("content") or ""),
                                  "ts": ts})
                rec = {"conv_id": row.get("conversation_id") or f"{short}-{i}",
                       "model_logged": row.get("model"),
                       "n_turns": len(turns), "turns": turns}
                f.write(json.dumps(rec) + "\n")
                kept += 1
                if kept % 200 == 0:
                    print(f"  kept {kept} (scanned {i + 1})")
                if kept >= args.per_dataset:
                    break
        print(f"  done: {kept} conversations")


if __name__ == "__main__":
    sys.exit(main())
