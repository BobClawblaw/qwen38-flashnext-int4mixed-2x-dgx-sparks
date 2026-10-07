"""Instruction following: IFEval's prompts (Google's release, pinned) with our own checkers for the verifiable
instruction types we implement. Scored per prompt (every instruction of the prompt must hold), strict and loose
(loose also accepts the reply with its first or last line removed and with * markers stripped, as the paper does).
Prompts carrying an instruction type we do not implement are left out, so the number reported is on the covered
subset and is not the paper's number.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from .common import Client, Result, fetch, strip_think

URL = ("https://raw.githubusercontent.com/google-research/google-research/"
       "26d8ccdab6fec61b5c83ad6327ea8bda9e580288/instruction_following_eval/data/input_data.jsonl")
URL_FALLBACK = "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl"


def _words(t: str) -> list[str]:
    return re.findall(r"\b\w+\b", t)


def _sentences(t: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()])


def _rel(count: int, relation: str, n: int) -> bool:
    return count >= n if relation == "at least" else count < n


def check(instr: str, kw: dict, r: str) -> bool | None:
    """True/False, or None when the instruction type is not implemented."""
    low = r.lower()
    if instr == "keywords:existence":
        return all(k.lower() in low for k in kw["keywords"])
    if instr == "keywords:frequency":
        return _rel(len(re.findall(rf"\b{re.escape(kw['keyword'].lower())}\b", low)), kw["relation"], kw["frequency"])
    if instr == "keywords:forbidden_words":
        return not any(re.search(rf"\b{re.escape(w.lower())}\b", low) for w in kw["forbidden_words"])
    if instr == "keywords:letter_frequency":
        return _rel(low.count(kw["letter"].lower()), kw["let_relation"], kw["let_frequency"])
    if instr == "length_constraints:number_words":
        return _rel(len(_words(r)), kw["relation"], kw["num_words"])
    if instr == "length_constraints:number_sentences":
        return _rel(_sentences(r), kw["relation"], kw["num_sentences"])
    if instr == "length_constraints:number_paragraphs":
        parts = [p for p in re.split(r"\s?\*\*\*\s?", r) if p.strip()]
        return len(parts) == kw["num_paragraphs"]
    if instr == "length_constraints:nth_paragraph_first_word":
        paras = [p for p in re.split(r"\n\n", r) if p.strip()]
        n, i = kw["num_paragraphs"], kw["nth_paragraph"]
        if len(paras) != n or not 1 <= i <= len(paras):
            return False
        first = re.findall(r"\w+", paras[i - 1])
        return bool(first) and first[0].lower() == kw["first_word"].lower()
    if instr == "detectable_content:number_placeholders":
        return len(re.findall(r"\[[^\[\]]+\]", r)) >= kw["num_placeholders"]
    if instr == "detectable_content:postscript":
        m = kw["postscript_marker"]
        return re.search(rf"^\s*{re.escape(m)}", r, re.M | re.I) is not None
    if instr == "detectable_format:number_bullet_lists":
        return len(re.findall(r"^\s*[\*\-]\s", r, re.M)) == kw["num_bullets"]
    if instr == "detectable_format:constrained_response":
        return any(opt in r for opt in ("My answer is yes.", "My answer is no.", "My answer is maybe."))
    if instr == "detectable_format:number_highlighted_sections":
        return len(re.findall(r"\*[^\n\*]+\*", r)) >= kw["num_highlights"]
    if instr == "detectable_format:multiple_sections":
        sp = kw["section_spliter"]
        return len(re.findall(rf"^\s*{re.escape(sp)}\s*\d+", r, re.M | re.I)) == kw["num_sections"]
    if instr == "detectable_format:json_format":
        t = re.sub(r"^```(json)?|```$", "", r.strip(), flags=re.M).strip()
        try:
            json.loads(t)
            return True
        except ValueError:
            return False
    if instr == "detectable_format:title":
        return re.search(r"<<[^<>\n]+>>", r) is not None
    if instr == "combination:two_responses":
        parts = r.split("******")
        return len(parts) == 2 and parts[0].strip() and parts[1].strip() and parts[0].strip() != parts[1].strip()
    if instr == "combination:repeat_prompt":
        return r.strip().startswith(kw["prompt_to_repeat"].strip())
    if instr == "startend:end_checker":
        return r.strip().endswith(kw["end_phrase"].strip())
    if instr == "startend:quotation":
        t = r.strip()
        return len(t) > 1 and t[0] == '"' and t[-1] == '"'
    if instr == "change_case:capital_word_frequency":
        caps = [w for w in _words(r) if w.isupper()]
        return _rel(len(caps), kw["capital_relation"], kw["capital_frequency"])
    if instr == "change_case:english_capital":
        return r == r.upper()
    if instr == "change_case:english_lowercase":
        return r == r.lower()
    if instr == "punctuation:no_comma":
        return "," not in r
    return None


IMPLEMENTED = frozenset((
    "keywords:existence", "keywords:frequency", "keywords:forbidden_words", "keywords:letter_frequency",
    "length_constraints:number_words", "length_constraints:number_sentences", "length_constraints:number_paragraphs",
    "length_constraints:nth_paragraph_first_word", "detectable_content:number_placeholders",
    "detectable_content:postscript", "detectable_format:number_bullet_lists", "detectable_format:constrained_response",
    "detectable_format:number_highlighted_sections", "detectable_format:multiple_sections",
    "detectable_format:json_format", "detectable_format:title", "combination:two_responses", "combination:repeat_prompt",
    "startend:end_checker", "startend:quotation", "change_case:capital_word_frequency", "change_case:english_capital",
    "change_case:english_lowercase", "punctuation:no_comma"))


def covered(row: dict) -> bool:
    return all(i in IMPLEMENTED for i in row["instruction_id_list"])


def variants(r: str) -> list[str]:
    lines = r.strip().split("\n")
    out = [r]
    if len(lines) > 1:
        out += ["\n".join(lines[1:]), "\n".join(lines[:-1])]
        if len(lines) > 2:
            out.append("\n".join(lines[1:-1]))
    return out + [v.replace("*", "") for v in list(out)]


def run(client: Client, *, n: int = 150, workers: int = 4, max_tokens: int = 1200) -> Result:
    try:
        path = fetch(URL, "ifeval-input_data.jsonl")
    except OSError:
        path = fetch(URL_FALLBACK, "ifeval-input_data.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if covered(r)][:n]
    t0 = time.time()

    def one(row: dict) -> tuple[bool, bool]:
        text = strip_think(client.ask(row["prompt"], max_tokens=max_tokens))
        kws = row.get("kwargs") or [{} for _ in row["instruction_id_list"]]

        def all_hold(t: str) -> bool:
            return all(check(i, {k: v for k, v in (kw or {}).items() if v is not None}, t)
                       for i, kw in zip(row["instruction_id_list"], kws))

        strict = all_hold(text)
        loose = strict or any(all_hold(v) for v in variants(text))
        return strict, loose

    with ThreadPoolExecutor(workers) as pool:
        outs = list(pool.map(one, rows))
    strict = sum(1 for s, _ in outs if s)
    loose = sum(1 for _, l in outs if l)
    return Result("ifeval", loose / len(rows), len(rows), loose, time.time() - t0,
                  [f"loose {loose}/{len(rows)}, strict {strict}/{len(rows)}, {len(rows)} covered prompts"],
                  {"strict": strict, "loose": loose, "n": len(rows)})
