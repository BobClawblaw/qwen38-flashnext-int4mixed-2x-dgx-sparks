"""Serving behaviour a client relies on, each scored per case:

- ``streaming``: a streamed greedy reply equals the non-streamed one, text and token count.
- ``multiturn``: facts given across a six-turn conversation are recalled exactly in the last turn.
- ``stop``: a stop string ends the reply before it, and is not part of the text.
- ``thinking``: with thinking on, the reply carries ``reasoning_content``, no think tags leak into ``content``, and
  the final answer is right (ten arithmetic word problems with known answers).
- ``copy``: a passage and a code block repeated verbatim on request (the copy-draft path) come back byte for byte.
- ``max_tokens``: a small max_tokens is honoured exactly with ``finish_reason`` length.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request

from .common import Client, Result, strip_think, text_of


def _stream(client: Client, messages: list[dict], max_tokens: int, extra: dict | None = None) -> tuple[str, int]:
    body = {"model": client.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    body.update(extra or {})
    req = urllib.request.Request(client.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    parts, tokens = [], 0
    with urllib.request.urlopen(req, timeout=client.timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                tokens = d["usage"].get("completion_tokens", 0)
            for ch in d.get("choices", []):
                parts.append(ch.get("delta", {}).get("content") or "")
    return "".join(parts), tokens


STREAM_PROMPTS = [
    "Explain how a bloom filter works in about 120 words.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "List seven countries that border Germany, one per line.",
    "Describe the water cycle to a ten-year-old in five sentences.",
]

TURNS = [
    ("My name is Odile and I live in Lyon.", None),
    ("My cat is called Pistache and she is 7 years old.", None),
    ("I work as a luthier, I repair violins.", None),
    ("My favourite number is 314.", None),
    ("I'm flying to Montreal on the 14th of November.", None),
    ("Summarize, as five short lines: my name, my city, my cat's name and age, my job, my favourite number and my "
     "travel destination with its date.", ["Odile", "Lyon", "Pistache", "7", "luthier|violin", "314", "Montreal", "14"]),
]

THINK = [
    ("A shop sells pens at 3 for $2. How many dollars do 27 pens cost?", "18"),
    ("A train travels 240 km in 3 hours. At the same speed, how many km does it travel in 5 hours?", "400"),
    ("What is the sum of the first 20 positive odd numbers?", "400"),
    ("A rectangle has perimeter 50 and width 10. What is its area?", "150"),
    ("If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines need for 100 widgets?", "5"),
    ("How many minutes are there in a week?", "10080"),
    ("A number doubled and then increased by 7 gives 39. What is the number?", "16"),
    ("What is 15% of 15% of 10000?", "225"),
    ("A clock shows 3:15. What is the smaller angle between the hands, in degrees? Give a decimal if needed.", "7.5"),
    ("How many positive divisors does 360 have?", "24"),
]

PASSAGE = ("The lighthouse keeper rose at five, trimmed the wick, wiped the salt from the lens, and wrote in the log: "
           "wind north-north-west, force six; two trawlers sighted at dawn; the supply boat late again. By noon the fog "
           "had lifted, and the island's single road shone like a ribbon laid across the heather.")
CODE = '''def merge(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    return out + a[i:] + b[j:]
'''


def run(client: Client) -> list[Result]:
    results = []
    # streaming
    t0 = time.time()
    ok, notes = 0, []
    for p in STREAM_PROMPTS:
        d = client.chat([{"role": "user", "content": p}], max_tokens=256)
        s_text, s_tok = _stream(client, [{"role": "user", "content": p}], 256)
        same = s_text == text_of(d) and s_tok == d.get("usage", {}).get("completion_tokens")
        ok += int(same)
        if not same:
            notes.append(f"{p[:30]!r} differs")
    results.append(Result("streaming", ok / len(STREAM_PROMPTS), len(STREAM_PROMPTS), ok, time.time() - t0, notes or ["identical"]))
    # multiturn
    t0 = time.time()
    msgs: list[dict] = []
    reply = ""
    for user, _ in TURNS:
        msgs.append({"role": "user", "content": user})
        reply = text_of(client.chat(msgs, max_tokens=200))
        msgs.append({"role": "assistant", "content": reply})
    want = TURNS[-1][1]
    hits = sum(1 for w in want if re.search(w, reply, re.I))
    results.append(Result("multiturn", hits / len(want), len(want), hits, time.time() - t0,
                          [f"recalled {hits}/{len(want)} facts"], {"reply": reply}))
    # stop strings
    t0 = time.time()
    cases = [("Count from 1 to 20, separated by commas.", "11"), ("List the days of the week, one per line.", "Friday"),
             ("Write the alphabet in capitals separated by spaces.", "M")]
    ok = 0
    for p, stop in cases:
        d = client.chat([{"role": "user", "content": p}], max_tokens=200, extra={"stop": [stop]})
        t = text_of(d)
        ok += int(stop not in t and len(t) > 0)
    results.append(Result("stop", ok / len(cases), len(cases), ok, time.time() - t0, [f"{ok}/{len(cases)} stopped before the string"]))
    # thinking
    t0 = time.time()
    ok, leaks, empty = 0, 0, 0
    for q, a in THINK:
        d = client.chat([{"role": "user", "content": q + " End with 'Answer: <number>'."}], max_tokens=4096, thinking=True)
        msg = d["choices"][0]["message"]
        content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
        leaks += int("<think>" in content or "</think>" in content)
        empty += int(not reasoning.strip())
        m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", content)
        ok += int(bool(m) and float(m[-1].replace(",", "")) == float(a) and "<think>" not in content and bool(reasoning.strip()))
    results.append(Result("thinking", ok / len(THINK), len(THINK), ok, time.time() - t0,
                          [f"{empty} replies without reasoning_content, {leaks} with think tags in content"]))
    # verbatim copy
    t0 = time.time()
    ok = 0
    for src in (PASSAGE, CODE):
        fence = "```" if src is CODE else ""
        reply = client.ask(f"Repeat the following text exactly, character for character, with nothing before or after it:\n\n{fence}\n{src}\n{fence}" if fence
                           else f"Repeat the following text exactly, character for character, with nothing before or after it:\n\n{src}",
                           max_tokens=400)
        got = strip_think(reply).strip()
        got = re.sub(r"^```\w*\n|\n?```$", "", got).strip()
        ok += int(got == src.strip())
    results.append(Result("copy", ok / 2, 2, ok, time.time() - t0, [f"{ok}/2 byte-identical"]))
    # max_tokens
    t0 = time.time()
    ok = 0
    for n in (1, 7, 33):
        d = client.chat([{"role": "user", "content": "Write a long essay about rivers."}], max_tokens=n)
        ok += int(d.get("usage", {}).get("completion_tokens") == n and d["choices"][0].get("finish_reason") == "length")
    results.append(Result("max-tokens", ok / 3, 3, ok, time.time() - t0, [f"{ok}/3 exact"]))
    return results
