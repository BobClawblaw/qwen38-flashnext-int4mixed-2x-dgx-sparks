"""GSM8K: grade-school math, exact match on the final number. The test split from the original release (jsonl)."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from .common import Client, Result, fetch, strip_think

# openai/grade-school-math, the repository's test set, pinned to a commit
URL = "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl"
PROMPT = ("{question}\n\nSolve step by step, then give the final answer on its own last line as "
          "'Answer: <number>' with no units.")


def gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def predicted(text: str) -> str | None:
    text = strip_think(text)
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


def run(client: Client, *, n: int = 250, workers: int = 4, max_tokens: int = 1024, thinking: bool = False) -> Result:
    rows = [json.loads(line) for line in fetch(URL, "gsm8k-test.jsonl").read_text().splitlines() if line.strip()]
    rows = rows[:n]
    t0 = time.time()

    def one(row: dict) -> tuple[bool, str]:
        text = client.ask(PROMPT.format(question=row["question"]), max_tokens=max_tokens, thinking=thinking)
        return same(predicted(text), gold(row["answer"])), text

    with ThreadPoolExecutor(workers) as pool:
        outs = list(pool.map(one, rows))
    ok = sum(1 for hit, _ in outs if hit)
    misses = [i for i, (hit, _) in enumerate(outs) if not hit][:10]
    return Result("gsm8k", ok / len(rows), len(rows), ok, time.time() - t0,
                  [f"misses at {misses}" if misses else "all correct"],
                  {"n": len(rows), "thinking": thinking, "misses": misses})
