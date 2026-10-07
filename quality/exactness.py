"""Exactness checks: drafted equals undrafted, concurrent equals alone, and a reply set to compare across servers.

Speculative decoding (MTP and copy drafts) must not change a greedy reply; nor may sharing a round with other
streams. Both are checked by reply hash. ``snapshot`` writes the replies of the fixed prompt set so a second
server (one GPU against two ranks, or another quantization) can be compared with ``compare``.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import Client, Result, sha12, text_of

PROMPTS = [
    "Explain, in about 150 words, why the sky is blue and sunsets are red.",
    "Write a Python function that parses an ISO-8601 date string and returns the day of the week, with a docstring and two examples.",
    "List five differences between TCP and UDP as a numbered list, one sentence each.",
    "Translate to French: 'The committee postponed the vote until the engineers had finished their report.'",
    "Write a bash script that renames every .jpeg file in a directory tree to .jpg, skipping files whose target name exists, and prints a summary.",
    "A train leaves at 09:40 and arrives at 14:05 after a 25-minute stop. What is its moving time? Show the steps briefly.",
    "Summarize the plot of a heist film you invent, in exactly four sentences.",
    "Give a JSON array of three objects, each with keys city, country and population (integer), for three large Asian cities.",
    "Rewrite this sentence in the passive voice and then in the imperative: 'The engineer calibrates the sensor every morning.'",
    "What is 17 * 23? Then subtract 91 and divide by 5. Show each step.",
    "Write a haiku about a lighthouse, then explain its syllable count.",
    "Name the planets of the solar system in order from the Sun, one per line, and mark the gas giants with an asterisk.",
]


def replies(client: Client, *, max_tokens: int = 300, draft: bool | None = None, workers: int = 1) -> list[dict]:
    def one(p: str) -> dict:
        d = client.chat([{"role": "user", "content": p}], max_tokens=max_tokens, draft=draft)
        t = text_of(d)
        n = d.get("usage", {}).get("completion_tokens", 0)
        return {"prompt": p, "sha": sha12(t), "tokens": n, "tok_s": n / d["_seconds"] if d["_seconds"] else 0, "text": t}

    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(one, PROMPTS))
    return [one(p) for p in PROMPTS]


def drafts(client: Client) -> Result:
    t0 = time.time()
    on, off = replies(client), replies(client, draft=False)
    same = sum(1 for a, b in zip(on, off) if a["sha"] == b["sha"])
    speed = sum(a["tok_s"] for a in on) / len(on), sum(b["tok_s"] for b in off) / len(off)
    return Result("drafts", same / len(on), len(on), same, time.time() - t0,
                  [f"drafted {speed[0]:.0f} tok/s vs undrafted {speed[1]:.0f} tok/s"],
                  {"drafted": on, "undrafted": off})


def concurrency(client: Client, streams: int = 4) -> Result:
    """Each prompt alone, then all of them together: the shared rounds must not change any reply."""
    t0 = time.time()
    alone = replies(client)
    together = replies(client, workers=streams)
    same = sum(1 for a, b in zip(alone, together) if a["sha"] == b["sha"])
    return Result("concurrent", same / len(alone), len(alone), same, time.time() - t0,
                  [f"{streams} streams together vs alone"], {"alone": alone, "together": together})


def snapshot(client: Client, path: Path) -> Result:
    t0 = time.time()
    rows = replies(client)
    path.write_text(json.dumps(rows, indent=1))
    return Result("snapshot", 1.0, len(rows), len(rows), time.time() - t0, [f"replies written to {path}"])


def compare(path_a: Path, path_b: Path) -> Result:
    a, b = json.loads(path_a.read_text()), json.loads(path_b.read_text())
    same = sum(1 for x, y in zip(a, b) if x["sha"] == y["sha"])
    diverge = []
    for x, y in zip(a, b):
        if x["sha"] != y["sha"]:
            k = next((i for i, (p, q) in enumerate(zip(x["text"], y["text"])) if p != q), min(len(x["text"]), len(y["text"])))
            diverge.append(k)
    return Result("ranks", same / len(a), len(a), same, 0.0,
                  [f"{same}/{len(a)} identical; divergences at chars {diverge}" if diverge else "all identical"],
                  {"diverge_at": diverge})
