"""MGSM: GSM8K-style problems in other languages (the multilingual set openai/simple-evals publishes), exact number
match. Shows whether quantization costs more in languages the calibration data saw less of. Each language file is
pinned by sha256; the first ``n`` problems of each are asked."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .common import Client, Result, fetch
from .gsm8k import predicted, same

URL = "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_{lang}.tsv"
SHA256 = {
    "de": "4dfea30fede44b813e2e496f5f0534049e83c7cc8aed6e00600b30ec62053626",
    "es": "5bd27ebdf00140cec845c5298dc17715f5cd57d8edda6df1976db40bf3f0750a",
    "fr": "36207c1c03fd7cd3ea491441eb755408199f94e4a647851df531bbaedc99d606",
    "ja": "59a2b50debe77981fd784cb3b2bef1505e3abf2a37116dc9d7a366ab029b4637",
    "zh": "b2fa63151022370a0de1f4211c8c284eae74b0f5a3b003b1d5982c0d4a73f661",
    "ru": "6bd30fd2e80c5bac23f566fb4ae0e6a55a19578401fbf9a01fb21beb3645fef8",
    "sw": "2bac828d77229e65d7c7197b1ad4a2cb5b1fe99b163f2cbdd66501de6a2115c4",
    "bn": "6b00bc7cc635547e866989284afa924d789b5affa2eb8de623c385dd943ad977",
}
LANGS = ("de", "es", "fr", "ja", "zh", "ru", "sw", "bn")
PROMPT = "{question}\n\nSolve step by step in the language of the question, then end with 'Answer: <number>' on its own last line."


def rows(lang: str) -> list[tuple[str, str]]:
    text = fetch(URL.format(lang=lang), f"mgsm_{lang}.tsv", SHA256.get(lang)).read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        if "\t" in line:
            q, a = line.rsplit("\t", 1)
            out.append((q, a.strip().replace(",", "")))
    return out


def run(client: Client, *, langs: tuple[str, ...] = LANGS, n: int = 30, workers: int = 4, max_tokens: int = 1024) -> Result:
    t0 = time.time()
    items = [(lang, q, a) for lang in langs for q, a in rows(lang)[:n]]

    def one(item: tuple[str, str, str]) -> tuple[str, bool]:
        lang, q, a = item
        return lang, same(predicted(client.ask(PROMPT.format(question=q), max_tokens=max_tokens)), a)

    with ThreadPoolExecutor(workers) as pool:
        outs = list(pool.map(one, items))
    per = {lang: sum(1 for l, hit in outs if l == lang and hit) for lang in langs}
    ok = sum(per.values())
    return Result("mgsm", ok / len(items), len(items), ok, time.time() - t0,
                  [", ".join(f"{k} {v}/{n}" for k, v in per.items())], {"per_language": per, "n": n})
