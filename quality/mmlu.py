"""MMLU: four-way multiple choice across 57 subjects, a fixed stratified sample, exact letter match.

The test split as openai/simple-evals publishes it (one CSV), pinned by sha256. The sample takes ``per_subject``
questions from every subject with a fixed seed, so every run asks the same questions and no subject dominates
(professional law alone is 11% of the split).
"""

from __future__ import annotations

import csv
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .common import Client, Result, fetch, strip_think

URL = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
SHA256 = "15b6785d49e0012602e089558a7a0dfb916baf97e9295aa25b48062f13c6afbb"
PROMPT = ("Answer the following multiple choice question. Think briefly if you need to, then give the answer on the "
          "last line as 'Answer: X' where X is one of A, B, C, D.\n\n{q}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}")


def sample(per_subject: int, seed: int = 2026) -> list[dict]:
    rows = list(csv.DictReader(open(fetch(URL, "mmlu.csv", SHA256), newline="", encoding="utf-8")))
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["Subject"]].append(r)
    rng = random.Random(seed)
    out = []
    for subject in sorted(by):
        out += rng.sample(by[subject], min(per_subject, len(by[subject])))
    return out


def letter(text: str) -> str | None:
    text = strip_think(text)
    m = re.findall(r"Answer\s*[:：]\s*\(?([ABCD])\)?", text)
    if m:
        return m[-1]
    m = re.findall(r"\b([ABCD])\)?\s*$", text.strip())
    return m[-1] if m else None


def run(client: Client, *, per_subject: int = 6, workers: int = 4, max_tokens: int = 768) -> Result:
    rows = sample(per_subject)
    t0 = time.time()

    def one(r: dict) -> tuple[bool, str]:
        reply = client.ask(PROMPT.format(q=r["Question"], a=r["A"], b=r["B"], c=r["C"], d=r["D"]), max_tokens=max_tokens)
        return letter(reply) == r["Answer"].strip(), r["Subject"]

    with ThreadPoolExecutor(workers) as pool:
        outs = list(pool.map(one, rows))
    ok = sum(1 for hit, _ in outs if hit)
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for hit, subj in outs:
        per[subj][0] += int(hit)
        per[subj][1] += 1
    weakest = sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1])[:3]
    return Result("mmlu", ok / len(rows), len(rows), ok, time.time() - t0,
                  [f"{len(per)} subjects x {per_subject}; weakest " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in weakest)],
                  {"per_subject": {k: v for k, v in sorted(per.items())}})
