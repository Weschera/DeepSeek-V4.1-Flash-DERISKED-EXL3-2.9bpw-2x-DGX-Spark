#!/usr/bin/env python3
"""Single-stream speed probe: speed_probe.py <base_url> <model> <out.json>
Thinking OFF, temp 1.0 top_p 0.95 (same as bench), no max_tokens. 3 prompt classes x 3 runs + one ~32K-token prefill.
Reports TTFT, decode tok/s (completion_tokens / (end - first token)), prefill tok/s. Medians."""
import json, sys, time, statistics, urllib.request
base, model, out = sys.argv[1:4]
PROMPTS = {
    "prose": "Write an original short story of about 1200 words about a lighthouse keeper who finds a message in a bottle. Plain prose, no headings.",
    "code": "Write a complete, well-commented Python module implementing an LRU cache with TTL expiry, thread safety, and a small pytest test suite. Output only code.",
    "count": "Count from 1 to 400, one number per line, nothing else.",
}
FILLER = ("The archive ledger records shipments, dates, quantities and inspector notes for each warehouse in the northern district. " * 1)

def run(msgs):
    body = {"model": model, "messages": msgs, "stream": True, "stream_options": {"include_usage": True},
            "temperature": 1.0, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices") or []:
                dl = c.get("delta") or {}
                if first is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")): first = time.time()
    t1 = time.time()
    ct = usage["completion_tokens"]; pt = usage["prompt_tokens"]
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": round(first - t0, 3),
            "decode_tok_s": round((ct - 1) / max(t1 - first, 1e-6), 2), "wall_s": round(t1 - t0, 2)}

res = {"model": model, "endpoint": base, "runs": {}}
run([{"role": "user", "content": "Say hi."}])  # warm
for k, p in PROMPTS.items():
    rs = [run([{"role": "user", "content": p}]) for _ in range(3)]
    res["runs"][k] = rs
    res[k] = {"decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in rs),
              "ttft_s_median": statistics.median(r["ttft_s"] for r in rs),
              "completion_tokens": [r["completion_tokens"] for r in rs]}
    print(k, res[k], flush=True)
long = FILLER * 1650 + "\n\nIn one sentence, what does the ledger record?"
r = run([{"role": "user", "content": long}])
r["prefill_tok_s"] = round(r["prompt_tokens"] / r["ttft_s"], 1)
res["prefill_32k"] = r
print("prefill_32k", r, flush=True)
json.dump(res, open(out, "w"), indent=1)
