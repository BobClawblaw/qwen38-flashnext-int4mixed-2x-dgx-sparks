#!/usr/bin/env python3
"""The recipe's decode ruler: fixed prompts, streamed greedy replies, per-stream and aggregate tokens per second.

    python3 bench/ruler.py --url http://127.0.0.1:8000 [--concurrency 1 4 16] [--runs 3] [--max-tokens 256] [--json out.json]

Four phases, each a distinct prompt per stream (no two streams share a prompt, so the engine cannot reuse a prefix
between them): prose, code, structured (JSON), and a long list. A phase's number is the median over runs of the
per-stream decode rate (completion tokens after the first, over the time after the first token) and the sum of
the streams' rates. TTFT is the time to the first content token. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

PHASES = {
    "prose": ["Write a 200-word explanation of how {topic} works, for a curious adult, no lists.",
              ["a refrigerator", "a tide", "a bicycle gear", "a vaccine", "a search engine index", "a thunderstorm",
               "a credit score", "a glacier", "a sourdough starter", "a transistor", "a coral reef", "a tax bracket",
               "a jet engine", "a compost heap", "a hash table", "an insulin pump"]],
    "code": ["Write a Python function `{name}` with a docstring and two doctests. {spec}",
             ["parse_duration|Parse strings like '2h30m' into seconds.", "merge_ranges|Merge overlapping integer ranges.",
              "rle|Run-length encode a string.", "topk_words|Return the k most frequent words of a text.",
              "is_balanced|Check bracket balance for (), [], {}.", "roman|Convert an int to Roman numerals.",
              "flatten|Flatten nested lists.", "moving_avg|Moving average with window n.",
              "luhn|Validate a card number with Luhn.", "group_anagrams|Group words that are anagrams.",
              "binary_search|Binary search returning the index or -1.", "caesar|Caesar-shift a string by k.",
              "transpose|Transpose a matrix given as lists.", "dedupe|Remove duplicates keeping order.",
              "wc|Count lines, words and bytes of a string.", "chunk|Split a list into chunks of size n."]],
    "structured": ["Return a JSON object describing {thing}: fields name, category, five key facts (array of strings), and a numeric rating from 1 to 10. JSON only.",
                   ["the Nile", "the ukulele", "Mount Fuji", "the octopus", "Rust (the language)", "the espresso machine",
                    "Saturn", "the London Underground", "the baobab tree", "the number pi", "chess", "the violin",
                    "the Sahara", "the honeybee", "the Eiffel Tower", "the yen"]],
    "list": ["List 40 {items}, numbered, one per line, no commentary.",
             ["rivers", "chemical elements", "programming languages", "capital cities", "birds", "card games",
              "fonts", "cheeses", "constellations", "dog breeds", "spices", "musical instruments", "trees",
              "operas", "sorting algorithms", "mountain ranges"]],
}


def one(url: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {}).get("content")
                if delta:
                    if first is None:
                        first = time.monotonic()
                    text.append(delta)
    end = time.monotonic()
    n = (usage or {}).get("completion_tokens") or 0
    decode = (n - 1) / (end - first) if first and n > 1 and end > first else 0.0
    return {"ttft": (first or end) - t0, "total": end - t0, "tokens": n, "decode": decode, "text": "".join(text)}


def wave(url: str, model: str, prompts: list[str], max_tokens: int, timeout: int) -> list[dict]:
    out: list[dict | None] = [None] * len(prompts)

    def go(i: int) -> None:
        try:
            out[i] = one(url, model, prompts[i], max_tokens, timeout)
        except Exception as exc:                      # noqa: BLE001  (a failed stream is reported, not fatal)
            out[i] = {"error": str(exc), "ttft": 0, "total": 0, "tokens": 0, "decode": 0.0, "text": ""}

    threads = [threading.Thread(target=go, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [o for o in out if o is not None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="default")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--phases", nargs="+", default=list(PHASES))
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--json")
    a = ap.parse_args()
    report = {"url": a.url, "max_tokens": a.max_tokens, "runs": a.runs, "cells": []}
    print(f"ruler: {a.url}, max_tokens {a.max_tokens}, {a.runs} runs, median per stream / aggregate tok/s, ttft median")
    for c in a.concurrency:
        for phase in a.phases:
            tmpl, fills = PHASES[phase]
            if c > len(fills):
                print(f"  {phase}: at most {len(fills)} distinct streams", file=sys.stderr)
                continue
            prompts = [tmpl.format(**{tmpl[tmpl.index("{") + 1:tmpl.index("}")]: f.split("|")[0]}) if "|" not in f
                       else tmpl.format(name=f.split("|")[0], spec=f.split("|")[1]) for f in fills[:c]]
            per, agg, ttft, toks = [], [], [], []
            for _ in range(a.runs):
                res = wave(a.url, a.model, prompts, a.max_tokens, a.timeout)
                rates = [r["decode"] for r in res if r["tokens"] > 1]
                if not rates:
                    continue
                per.append(statistics.median(rates))
                agg.append(sum(rates))
                ttft.append(statistics.median(r["ttft"] for r in res))
                toks.append(sum(r["tokens"] for r in res) / len(res))
            if not per:
                print(f"  c={c:<3} {phase:<10} no replies")
                continue
            cell = {"concurrency": c, "phase": phase, "per_stream": statistics.median(per), "aggregate": statistics.median(agg),
                    "ttft": statistics.median(ttft), "tokens_mean": statistics.median(toks), "runs": len(per)}
            report["cells"].append(cell)
            print(f"  c={c:<3} {phase:<10} {cell['per_stream']:7.1f} / {cell['aggregate']:7.1f} tok/s   ttft {cell['ttft']:.2f}s"
                  f"   {cell['tokens_mean']:.0f} tokens a reply")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
