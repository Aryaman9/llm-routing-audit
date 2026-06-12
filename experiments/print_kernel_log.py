"""Print a Kaggle kernel JSON log, filtering noise; safe on Windows consoles."""

import json
import sys

path = sys.argv[1]
tail = int(sys.argv[2]) if len(sys.argv) > 2 else 50
NOISE = ("SyntaxWarning", "escape sequence", "NbConvert", "FutureWarning",
         "forward_call", "it/s]", "B/s]", "?B/s")

log = json.load(open(path, encoding="utf-8"))
keep = []
for e in log:
    d = e["data"].strip()
    if not d or any(x in d for x in NOISE):
        continue
    keep.append(f"{e['time']:8.1f} {e['stream_name'][:3]} {d[:170]}")

out = "\n".join(keep[-tail:])
sys.stdout.buffer.write(out.encode("utf-8", errors="replace") + b"\n")
