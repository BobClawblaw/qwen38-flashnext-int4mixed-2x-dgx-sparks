"""Long context: a secret planted in a generated document, asked for at the end; and spread facts asked together.

Both documents are generated from a seed, so a run is repeatable and no dataset is needed. The needle check reports
whether the code came back, the prompt pass time and rate; the spread-facts check plants ten (name, value) pairs at
even depths of one document and asks for all ten in one question (recall across the whole window, not one lookup).
"""

from __future__ import annotations

import random
import re
import time

from .common import Client, Result, strip_think

TOPICS = ["harbour logistics", "glacier surveys", "orchard irrigation", "medieval bookbinding", "railway signalling",
          "coral reef acoustics", "bread fermentation", "tunnel ventilation", "lighthouse optics", "beekeeping records",
          "canal locks", "paper marbling", "wind tunnel calibration", "tea auctions", "subway tiling"]
VERBS = ["measured", "reported", "revised", "catalogued", "compared", "scheduled", "inspected", "archived"]
CHARS_PER_TOKEN = 3.9


def document(target_tokens: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    n = int(target_tokens * CHARS_PER_TOKEN / 300) + 1
    paras = []
    for i in range(n):
        t = TOPICS[i % len(TOPICS)]
        paras.append(f"Section {i}. The {t} team {rng.choice(VERBS)} {rng.randint(3, 900)} items during week "
                     f"{rng.randint(1, 52)}, noting that {rng.choice(TOPICS)} depends on {rng.choice(TOPICS)} more than "
                     f"expected. Their log lists {rng.randint(10, 99)} entries, the longest about {rng.choice(TOPICS)}, "
                     f"and ends with a reminder to compare results with the {rng.randint(1990, 2025)} baseline.\n\n")
    return paras


def needle(client: Client, tokens: int, depth: float, *, seed: int = 7) -> dict:
    paras = document(tokens, seed + int(depth * 1000))
    code = f"{seed % 9}-{(seed * 7) % 9}-{(seed * 3) % 9}-{(seed + 5) % 9}-{(seed * 11) % 9}"
    paras.insert(int(len(paras) * depth), f"The secret code for the blue door is {code}. Remember it.\n\n")
    prompt = "Read this document carefully.\n\n" + "".join(paras) + \
        "\n\nQuestion: What is the secret code for the blue door? Answer with the digits only."
    d = client.chat([{"role": "user", "content": prompt}], max_tokens=24)
    text = strip_think(d["choices"][0]["message"].get("content") or "")
    pt = d.get("usage", {}).get("prompt_tokens") or 0
    found = code.replace("-", "") in re.sub(r"[^0-9]", "", text)
    return {"tokens": pt, "depth": depth, "found": found, "seconds": d["_seconds"],
            "prefill_tok_s": pt / d["_seconds"] if d["_seconds"] else 0, "reply": text[:60]}


def spread_facts(client: Client, tokens: int, *, seed: int = 11, count: int = 10) -> dict:
    rng = random.Random(seed)
    paras = document(tokens, seed)
    names = ["Avery", "Blake", "Casey", "Devin", "Emery", "Finley", "Harper", "Jordan", "Kendall", "Morgan"][:count]
    values = [rng.randint(100, 999) for _ in names]
    for i, (nm, v) in enumerate(zip(names, values)):
        at = int(len(paras) * (i + 0.5) / count)
        paras.insert(at, f"Note: {nm}'s locker number is {v}.\n\n")
    prompt = "Read this document carefully.\n\n" + "".join(paras) + \
        "\n\nQuestion: List every person's locker number mentioned in the document, as 'Name: number' lines, nothing else."
    d = client.chat([{"role": "user", "content": prompt}], max_tokens=200)
    text = strip_think(d["choices"][0]["message"].get("content") or "")
    hits = sum(1 for nm, v in zip(names, values) if re.search(rf"{nm}\D{{0,12}}{v}\b", text))
    pt = d.get("usage", {}).get("prompt_tokens") or 0
    return {"tokens": pt, "recalled": hits, "of": count, "seconds": d["_seconds"],
            "prefill_tok_s": pt / d["_seconds"] if d["_seconds"] else 0, "reply": text[:200]}


def run(client: Client, *, lengths: tuple[int, ...] = (8000, 32000, 64000), depths: tuple[float, ...] = (0.1, 0.5, 0.9),
        facts_lengths: tuple[int, ...] = (32000, 64000)) -> Result:
    """Prefills up to 64k tokens (longer fills say little about quality and tie the pair up for minutes)."""
    t0 = time.time()
    rows, passed, cases = [], 0, 0
    for n in lengths:
        for depth in depths:
            r = needle(client, n, depth)
            rows.append(r)
            cases += 1
            passed += int(r["found"])
    facts = []
    for n in facts_lengths:
        r = spread_facts(client, n)
        facts.append(r)
        cases += r["of"]
        passed += r["recalled"]
    worst = min((r["prefill_tok_s"] for r in rows), default=0)
    return Result("long-context", passed / cases if cases else 0, cases, passed, time.time() - t0,
                  [f"needles {sum(int(r['found']) for r in rows)}/{len(rows)}, facts " +
                   ", ".join(f"{f['recalled']}/{f['of']}@{f['tokens'] // 1000}k" for f in facts) +
                   f", slowest prefill {worst:.0f} tok/s"],
                  {"needles": rows, "facts": facts})
