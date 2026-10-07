"""Degeneration: the share of repeated 4-grams in long open-ended replies, and replies that stop early or loop."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor

from .common import Client, Result, strip_think

PROMPTS = [
    "Write a 700-word short story about a cartographer who discovers a river that is not on any map.",
    "Explain, in about 600 words, how a city might plan a new tram line, from survey to opening day.",
    "Write a detailed, 600-word product review of an imaginary mechanical keyboard, covering build, switches, software and value.",
    "Describe a week-long walking holiday in the Dolomites, day by day, around 600 words.",
    "Write a 600-word essay on why libraries still matter in the age of the internet.",
    "Narrate, in 600 words, a chess game between two old friends from the first move to the handshake.",
    "Write a 600-word technical overview of how a web browser renders a page, from HTML bytes to pixels.",
    "In 600 words, give advice to someone starting a vegetable garden on a balcony.",
]


def repeated_4gram_rate(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    if not grams:
        return 0.0
    seen: set[tuple] = set()
    repeats = 0
    for g in grams:
        if g in seen:
            repeats += 1
        seen.add(g)
    return repeats / len(grams)


def run(client: Client, *, workers: int = 4, max_tokens: int = 1200) -> Result:
    t0 = time.time()

    def one(p: str) -> dict:
        d = client.chat([{"role": "user", "content": p}], max_tokens=max_tokens)
        text = strip_think(d["choices"][0]["message"].get("content") or "")
        return {"rate": repeated_4gram_rate(text), "words": len(text.split()),
                "finish": d["choices"][0].get("finish_reason"), "tokens": d.get("usage", {}).get("completion_tokens", 0)}

    with ThreadPoolExecutor(workers) as pool:
        outs = list(pool.map(one, PROMPTS))
    rate = sum(o["rate"] for o in outs) / len(outs)
    # pass: a reply under 3% repeated 4-grams that reached at least 350 words and ended on its own
    good = sum(1 for o in outs if o["rate"] < 0.03 and o["words"] >= 350 and o["finish"] == "stop")
    return Result("repetition", good / len(outs), len(outs), good, time.time() - t0,
                  [f"repeated 4-grams {100 * rate:.2f}% mean, max {100 * max(o['rate'] for o in outs):.2f}%; "
                   f"{sum(1 for o in outs if o['finish'] != 'stop')} replies hit max_tokens"], {"replies": outs})
