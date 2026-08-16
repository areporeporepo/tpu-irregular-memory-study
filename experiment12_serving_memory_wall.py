#!/usr/bin/env python3
"""Does an SRAM-first accelerator hit a wall as the KV cache grows? Ask the serving endpoints.

This study's result is about memory placement: a gather is fast when its table sits in on-chip
memory and about 3.3x slower when it does not, and we measured the on-chip budget at 48 MiB on v5p
and 96 MiB on v6e. Two vendors have taken that idea to its limit and built parts whose entire memory
is the fast tier: Groq with 230 MB of SRAM per chip and no DRAM, Cerebras with 44 GB on one wafer at
roughly 21 PB/s.

That design should be visible from the outside, without kernel access, because of one thing it cannot
avoid: weights are static and can be laid into SRAM ahead of time, but the KV cache is not. It grows
with context and batch, and it has to live somewhere. So the prediction is specific and falsifiable:

    at short context   SRAM-first parts should decode far faster than HBM parts, because the weight
                       traffic that dominates decode is coming from on-chip memory
    at long context    their advantage should shrink faster than an HBM part's, because their fast
                       tier is small and fixed while an H100 or a v5p has 80 to 95 GB to absorb KV

If the advantage is flat in context length, the KV cache is not living in the fast tier and the
architecture story is more complicated than the marketing. Either answer is worth having.

Everything here is a free-tier API call over an OpenAI-compatible endpoint, measured by streaming so
that time-to-first-token and per-token decode time can be separated. Prefill is compute-bound and
decode is memory-bound, and conflating them is the usual way these comparisons go wrong.

    CEREBRAS_API_KEY=... GROQ_API_KEY=... python3 experiment12_serving_memory_wall.py
    python3 experiment12_serving_memory_wall.py --contexts 256,4096,32768
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "serving_memory_wall.json"

# Endpoint, env var for the key, and a model that is actually served there. All three speak the
# OpenAI chat-completions dialect, which is the only reason this is one script and not three.
PROVIDERS = [
    {"name": "cerebras", "base": "https://api.cerebras.ai/v1", "key": "CEREBRAS_API_KEY",
     "model": "llama-3.3-70b", "memory": "44 GB on-wafer SRAM, ~21 PB/s"},
    {"name": "groq", "base": "https://api.groq.com/openai/v1", "key": "GROQ_API_KEY",
     "model": "llama-3.3-70b-versatile", "memory": "230 MB SRAM per chip, no DRAM"},
    {"name": "openrouter", "base": "https://openrouter.ai/api/v1", "key": "OPENROUTER_API_KEY",
     "model": "meta-llama/llama-3.3-70b-instruct", "memory": "HBM, whichever GPU it routes to"},
]

OUTPUT_TOKENS = 96          # long enough to get a stable decode rate, short enough to be polite
FILLER = ("The following is an excerpt from a technical report on memory hierarchies in "
          "accelerators, included here only to occupy context. ")


def build_prompt(target_tokens: int) -> str:
    """Roughly `target_tokens` of input. Four characters per token is close enough for this axis."""
    body = FILLER * max(1, (target_tokens * 4) // len(FILLER))
    return (body + "\n\nIgnore the text above. Reply with a single short sentence about memory "
                   "bandwidth.")


def stream_once(prov: dict, prompt: str, timeout: int = 180) -> dict:
    """One streamed completion, timing first token separately from the rest."""
    key = os.environ.get(prov["key"], "")
    if not key:
        return {"error": f"{prov['key']} not set"}
    payload = json.dumps({
        "model": prov["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        prov["base"] + "/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})

    t0 = time.perf_counter()
    ttft = None
    stamps: list[float] = []
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                blob = line[5:].strip()
                if blob == "[DONE]":
                    break
                try:
                    chunk = json.loads(blob)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or [{}]
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    stamps.append(now)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:180]
        return {"error": f"HTTP {exc.code}: {body}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:180]}

    if ttft is None or len(stamps) < 2:
        return {"error": "no tokens streamed back"}
    decode_s = stamps[-1] - stamps[0]
    n = len(stamps) - 1
    return {"ttft_ms": round(ttft * 1e3, 1),
            "decode_tok_s": round(n / decode_s, 1) if decode_s > 0 else None,
            "tokens_streamed": len(stamps),
            "total_ms": round((stamps[-1] - t0) * 1e3, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="256,2048,8192,32768",
                    help="approximate input lengths in tokens, comma separated")
    ap.add_argument("--repeats", type=int, default=3, help="attempts per point, median reported")
    args = ap.parse_args()
    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]

    have = [p for p in PROVIDERS if os.environ.get(p["key"])]
    missing = [p["key"] for p in PROVIDERS if not os.environ.get(p["key"])]
    print(f"providers with keys: {[p['name'] for p in have] or 'none'}")
    if missing:
        print(f"skipping for want of a key: {', '.join(missing)}")
    if not have:
        raise SystemExit("nothing to measure; set at least one key")

    records = []
    for prov in have:
        print(f"\n{prov['name']}  model={prov['model']}  memory={prov['memory']}", flush=True)
        for ctx in contexts:
            prompt = build_prompt(ctx)
            tries = [stream_once(prov, prompt) for _ in range(args.repeats)]
            ok = [t for t in tries if "error" not in t]
            rec = {"provider": prov["name"], "model": prov["model"], "memory": prov["memory"],
                   "context_target": ctx, "attempts": len(tries), "ok": len(ok)}
            if ok:
                rec["ttft_ms"] = round(statistics.median(t["ttft_ms"] for t in ok), 1)
                rates = [t["decode_tok_s"] for t in ok if t.get("decode_tok_s")]
                rec["decode_tok_s"] = round(statistics.median(rates), 1) if rates else None
                rec["prompt_tokens"] = ok[0].get("prompt_tokens")
                print(f"  ctx~{ctx:>6}  reported {str(rec['prompt_tokens']):>6} tok  "
                      f"TTFT {rec['ttft_ms']:>8.1f} ms  decode {rec['decode_tok_s']:>7.1f} tok/s",
                      flush=True)
            else:
                rec["error"] = tries[0].get("error", "unknown")
                print(f"  ctx~{ctx:>6}  failed: {rec['error'][:90]}", flush=True)
            records.append(rec)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps({"output_tokens": OUTPUT_TOKENS, "records": records},
                                      indent=2))

    # The comparison the whole thing exists for: how does each provider's decode rate hold up as the
    # KV cache grows? Normalise each provider to its own shortest context so the axis is degradation
    # rather than absolute speed, because absolute speed conflates the model, the batch and the chip.
    print("\ndecode rate relative to each provider's own shortest context:")
    for prov in have:
        pts = [r for r in records if r["provider"] == prov["name"] and r.get("decode_tok_s")]
        if len(pts) < 2:
            continue
        base = pts[0]["decode_tok_s"]
        trail = "  ".join(f"{p['context_target']}:{p['decode_tok_s'] / base:.2f}x" for p in pts)
        print(f"  {prov['name']:11s} base {base:7.1f} tok/s   {trail}")
    print("\nIf the SRAM-first providers fall off faster than the HBM one, the KV cache is competing")
    print("for the fast tier. If every provider is flat, it is not, and the interesting question")
    print("becomes where they are putting it instead.")


if __name__ == "__main__":
    main()
