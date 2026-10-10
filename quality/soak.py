"""Soak: several chats resume long kept prompt states at once while another stream decodes, turn after turn.

The pattern that broke CUDA graph relocation upstream (TensorFold #615) and that no other check here covered:
concurrent resumes of 5-6k-token kept prefixes beside a stream decoding a long reply. Each chat grows a turn a round
and resumes its previous turn's state; afterwards every chat is replayed alone, turn by turn, as the reference. Every
concurrent reply must equal its lone replay, every request must succeed, at least half the concurrent turns must
actually resume a kept state, and the server must still answer afterwards. AGENTS.md: run it on every engine change.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor

from .common import Client, Result, sha12, text_of

WORDS = ("harbor lantern meadow copper signal orbit candle marble thunder river stone cloud garden violet forest "
         "engine ledger compass glacier pepper saddle quarry ribbon timber velvet walnut anchor bramble cinder").split()
FACTS = ("The vault code is {n}.", "The ferry leaves at {n} past the hour.", "Room {n} holds the archive.")


def document(i: int, words: int) -> tuple[str, str]:
    """Chat ``i``'s long context (repeatable) and the fact hidden in its middle."""
    rng = random.Random(7000 + i)
    body = [rng.choice(WORDS) for _ in range(words)]
    fact = FACTS[i % len(FACTS)].format(n=1000 + 37 * i)
    body.insert(words // 2, fact)
    return f"Notes {i}. " + " ".join(body), fact


def history(i: int, words: int, turn: int = 0) -> list[dict]:
    """Chat ``i`` at ``turn``: the long context, then one fixed exchange a turn and the question that resumes it.

    The answers in the history are fixed text, not the model's, so each turn's request is the same whatever was
    served before; each turn extends the last one, as a real chat does, so it resumes the previous turn's kept state."""
    doc, _ = document(i, words)
    msgs = [{"role": "system", "content": "Answer from the notes the user gave. Be brief."},
            {"role": "user", "content": doc + "\n\nAcknowledge the notes in one sentence."},
            {"role": "assistant", "content": "I have read the notes."}]
    for k in range(turn):
        msgs += [{"role": "user", "content": QUESTIONS[k % len(QUESTIONS)].format(k=k)},
                 {"role": "assistant", "content": f"Answered question {k}."}]
    return msgs + [{"role": "user", "content": QUESTIONS[turn % len(QUESTIONS)].format(k=turn)}]


QUESTIONS = ("Question {k}: which sentence in the notes states a number? Quote it.",
             "Question {k}: name three words that appear often in the notes.",
             "Question {k}: in one sentence, what are the notes about?")


def background(round_: int) -> list[dict]:
    return [{"role": "user", "content": f"Write a long, detailed story (part {round_}) about a lighthouse keeper "
                                        "who repairs clocks. Keep going until you are stopped."}]


def run(client: Client, *, chats: int = 3, rounds: int = 20, words: int = 4600, max_tokens: int = 64,
        min_resumed: float = 0.5) -> Result:
    t0 = time.time()

    def ask(i: int, turn: int) -> dict:
        try:
            d = client.chat(history(i, words, turn), max_tokens=max_tokens)
            u = d.get("usage") or {}
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
            return {"chat": i, "turn": turn, "sha": sha12(text_of(d)), "prompt": u.get("prompt_tokens"),
                    "cached": cached, "error": None}
        except Exception as e:  # noqa: BLE001 - every failure counts against the soak
            return {"chat": i, "turn": turn, "sha": None, "prompt": None, "cached": None, "error": repr(e)[:200]}

    warm = [ask(i, 0) for i in range(chats)]          # each chat's first turn alone, cold
    rows, errors = [], [w["error"] for w in warm if w["error"]]
    for k in range(1, rounds + 1):
        with ThreadPoolExecutor(chats + 1) as pool:   # turn k of every chat at once, beside a long decode
            bg = pool.submit(lambda: client.chat(background(k), max_tokens=512))
            together = list(pool.map(lambda i: ask(i, k), range(chats)))
            try:
                bg.result()
            except Exception as e:  # noqa: BLE001
                errors.append(f"background {k}: {e!r}"[:200])
        rows.append({"round": k, "together": together})
    # the reference: every chat replayed alone, one turn after another and one chat after another, so each turn
    # resumes the previous one's state on its own (no request is sent twice in a row: a repeat is a different path)
    alone = {(i, k): ask(i, k) for i in range(chats) for k in range(1, rounds + 1)}
    cases = same = 0
    for row in rows:
        for g in row["together"]:
            a = alone[(g["chat"], g["turn"])]
            cases += 1
            errors += [x["error"] for x in (g, a) if x["error"]]
            same += g["sha"] is not None and g["sha"] == a["sha"]
        row["alone"] = [alone[(g["chat"], g["turn"])] for g in row["together"]]
    try:
        client.chat([{"role": "user", "content": "Say hello."}], max_tokens=8)
        alive = True
    except Exception as e:  # noqa: BLE001
        alive, errors = False, errors + [f"after: {e!r}"[:200]]
    served = [g for row in rows for g in row["together"] if g["error"] is None and g["prompt"]]
    reported = [g for g in served if g["cached"] is not None]
    hit = sum(1 for g in reported if g["cached"] >= g["prompt"] // 2)   # resumed: most of the prompt was kept
    rate = hit / len(reported) if reported else None  # None: the server does not report cached tokens
    exercised = rate is None or rate >= min_resumed
    ok = alive and exercised
    sizes = sorted(g["prompt"] for g in served)
    notes = [f"{chats} chats x {rounds} turns at once beside a decoding stream"
             + (f" ({sizes[0] / 1000:.1f}k-{sizes[-1] / 1000:.1f}k-token prompts)" if sizes else "")
             + f": {same}/{cases} equal to the same request alone, {len(errors)} errors, server "
             + ("answers" if alive else "DEAD") + " after"
             + (f"; {hit}/{len(reported)} concurrent turns resumed a kept state" if reported
                else "; cached tokens not reported")
             + ("" if exercised else f" (under {min_resumed:.0%}: resumes not exercised)")]
    return Result("soak", same / cases if ok else 0.0, cases, same if ok else 0, time.time() - t0, notes,
                  {"warm": warm, "rounds": rows, "errors": errors[:50]})
