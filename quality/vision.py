"""Vision: images drawn here (shapes, colours, counts, positions) with known answers; the PNG writer is stdlib.

Twelve probes: which shapes and colours, how many of a kind, which side a shape is on, and a blank image. A probe
passes when the reply names what it must and nothing it must not (for counts, the right number word or digit).
"""

from __future__ import annotations

import base64
import re
import struct
import time
import zlib

from .common import Client, Result, strip_think

W = H = 256
COLORS = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 170, 70), "yellow": (240, 210, 40),
          "white": (255, 255, 255), "black": (20, 20, 20)}


def png(pixels: list[list[tuple[int, int, int]]]) -> bytes:
    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in pixels)

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def canvas() -> list[list[tuple[int, int, int]]]:
    return [[COLORS["white"] for _ in range(W)] for _ in range(H)]


def circle(px, cx: int, cy: int, r: int, color: str) -> None:
    c = COLORS[color]
    for y in range(max(0, cy - r), min(H, cy + r + 1)):
        for x in range(max(0, cx - r), min(W, cx + r + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                px[y][x] = c


def square(px, cx: int, cy: int, half: int, color: str) -> None:
    c = COLORS[color]
    for y in range(max(0, cy - half), min(H, cy + half)):
        for x in range(max(0, cx - half), min(W, cx + half)):
            px[y][x] = c


def triangle(px, cx: int, cy: int, half: int, color: str) -> None:
    c = COLORS[color]
    for y in range(max(0, cy - half), min(H, cy + half)):
        t = (y - (cy - half)) / (2 * half)                    # 0 at the apex, 1 at the base
        w = int(half * t)
        for x in range(max(0, cx - w), min(W, cx + w + 1)):
            px[y][x] = c


def probes() -> list[dict]:
    out = []

    def add(name, draw, question, must, must_not=()):
        px = canvas()
        draw(px)
        out.append({"name": name, "png": png(px), "question": question, "must": must, "must_not": must_not})

    add("red-circle-blue-square", lambda p: (circle(p, 80, 128, 40, "red"), square(p, 180, 128, 36, "blue")),
        "Which shapes are in the image and what colour is each? One short sentence.", ["red", "circle", "blue", "square"], ["triangle", "green"])
    add("green-triangle", lambda p: triangle(p, 128, 128, 70, "green"),
        "What shape is shown and what colour is it? Answer in a few words.", ["green", "triangle"], ["circle", "square", "red"])
    add("three-circles", lambda p: [circle(p, 50 + 78 * i, 128, 28, "red") for i in range(3)],
        "How many circles are in the image? Answer with a number.", [r"\b(3|three)\b"], [r"\b(2|two|4|four)\b"])
    add("two-squares", lambda p: [square(p, 80 + 96 * i, 128, 30, "blue") for i in range(2)],
        "How many squares are in the image? Answer with a number.", [r"\b(2|two)\b"], [r"\b(3|three|1|one)\b"])
    add("left-right", lambda p: (square(p, 64, 128, 34, "blue"), circle(p, 192, 128, 36, "red")),
        "Is the blue square on the left or on the right of the red circle? Answer left or right.", [r"\bleft\b"], [r"\bright\b"])
    add("right-left", lambda p: (circle(p, 64, 128, 36, "red"), square(p, 192, 128, 34, "blue")),
        "Is the blue square on the left or on the right of the red circle? Answer left or right.", [r"\bright\b"], [r"\bleft\b"])
    add("yellow-circle", lambda p: circle(p, 128, 128, 60, "yellow"),
        "What colour is the circle? One word.", ["yellow"], ["red", "blue", "green"])
    add("four-triangles", lambda p: [triangle(p, 40 + 58 * i, 128, 24, "green") for i in range(4)],
        "How many triangles do you see? Answer with a number.", [r"\b(4|four)\b"], [r"\b(3|three|5|five)\b"])
    add("blank", lambda p: None, "Describe what is in this image in one sentence.", [r"(blank|empty|white|plain|nothing|solid)"],
        ["circle", "square", "triangle"])
    add("black-square-top", lambda p: (square(p, 128, 60, 30, "black"), circle(p, 128, 196, 30, "red")),
        "Is the black square above or below the red circle? Answer above or below.", [r"\babove\b"], [r"\bbelow\b"])
    add("mixed-count", lambda p: (circle(p, 60, 80, 24, "red"), circle(p, 196, 80, 24, "red"), square(p, 128, 180, 28, "blue")),
        "How many red circles are there? Answer with a number.", [r"\b(2|two)\b"], [r"\b(3|three|1|one)\b"])
    add("colours-list", lambda p: (square(p, 50, 128, 26, "red"), square(p, 128, 128, 26, "green"), square(p, 206, 128, 26, "blue")),
        "List the colours of the squares from left to right, comma separated.", [r"red.*green.*blue"], [])
    return out


def run(client: Client) -> Result:
    t0 = time.time()
    rows = []
    for p in probes():
        url = "data:image/png;base64," + base64.b64encode(p["png"]).decode()
        try:
            d = client.chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                          {"type": "text", "text": p["question"]}]}], max_tokens=80)
            text = strip_think(d["choices"][0]["message"].get("content") or "")
            err = None
        except Exception as exc:                 # noqa: BLE001
            text, err = "", str(exc)[:80]
        low = text.lower()
        ok = (not err and all(re.search(m, low) for m in p["must"]) and not any(re.search(m, low) for m in p["must_not"]))
        rows.append({"name": p["name"], "ok": ok, "reply": text[:120], "error": err})
    passed = sum(1 for r in rows if r["ok"])
    bad = [f"{r['name']}: {r['error'] or r['reply'][:50]!r}" for r in rows if not r["ok"]][:4]
    return Result("vision", passed / len(rows), len(rows), passed, time.time() - t0, bad or ["all probes answered"],
                  {"probes": rows})
