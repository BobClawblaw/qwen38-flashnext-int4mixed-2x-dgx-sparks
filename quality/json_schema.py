"""JSON output: thirty prompts, each with a schema the reply must satisfy, checked by a small validator of our own.

Two modes: ``free`` (the schema is described in the prompt, the reply must be JSON that fits it) and ``guided``
(``response_format`` json_schema, served only where the engine takes grammars: the serial profile). The validator
covers the subset the prompts use: object/array/string/integer/number/boolean, required, enum, minimum/maximum,
minItems/maxItems, additionalProperties false.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from .common import Client, Result, strip_think

CASES = [
    ("a person named Ada Lovelace born in 1815 in London",
     {"type": "object", "required": ["name", "born", "city"], "additionalProperties": False,
      "properties": {"name": {"type": "string"}, "born": {"type": "integer", "minimum": 1000, "maximum": 2100}, "city": {"type": "string"}}}),
    ("three primary colours",
     {"type": "object", "required": ["colours"], "properties": {"colours": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "string"}}}}),
    ("a thermostat reading of 21.5 degrees Celsius taken at 7am, status ok",
     {"type": "object", "required": ["celsius", "hour", "status"], "properties": {"celsius": {"type": "number"}, "hour": {"type": "integer", "minimum": 0, "maximum": 23}, "status": {"type": "string", "enum": ["ok", "warn", "fail"]}}}),
    ("a shopping list with milk (2), eggs (12) and bread (1)",
     {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "minItems": 3, "items": {"type": "object", "required": ["name", "qty"], "properties": {"name": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}}}}}}),
    ("whether Paris is the capital of France, with a confidence between 0 and 1",
     {"type": "object", "required": ["answer", "confidence"], "properties": {"answer": {"type": "boolean"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}),
    ("the planets Mercury, Venus and Earth with their order from the Sun",
     {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "required": ["name", "order"], "properties": {"name": {"type": "string"}, "order": {"type": "integer", "minimum": 1}}}}),
    ("a book: title Dune, author Frank Herbert, year 1965, genres science fiction and adventure",
     {"type": "object", "required": ["title", "author", "year", "genres"], "properties": {"title": {"type": "string"}, "author": {"type": "string"}, "year": {"type": "integer"}, "genres": {"type": "array", "items": {"type": "string"}, "minItems": 2}}}),
    ("a weather forecast for Monday: high 18, low 9, rain likely",
     {"type": "object", "required": ["day", "high", "low", "rain"], "properties": {"day": {"type": "string", "enum": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}, "high": {"type": "integer"}, "low": {"type": "integer"}, "rain": {"type": "boolean"}}}),
    ("a rectangle 4 by 6 with its area",
     {"type": "object", "required": ["width", "height", "area"], "properties": {"width": {"type": "number"}, "height": {"type": "number"}, "area": {"type": "number"}}}),
    ("a user record: id 42, email a@b.co, roles admin and editor, active",
     {"type": "object", "required": ["id", "email", "roles", "active"], "additionalProperties": False, "properties": {"id": {"type": "integer"}, "email": {"type": "string"}, "roles": {"type": "array", "items": {"type": "string", "enum": ["admin", "editor", "viewer"]}}, "active": {"type": "boolean"}}}),
]
# twenty more by variation: the same schemas, different content
VARIANTS = [
    ("a person named Grace Hopper born in 1906 in New York", 0), ("a person named Alan Turing born in 1912 in London", 0),
    ("three noble gases", 1), ("three string instruments", 1),
    ("a thermostat reading of 19 degrees Celsius taken at 11pm, status warn", 2),
    ("a thermostat reading of 24.25 degrees taken at noon, status fail", 2),
    ("a shopping list with rice (1), apples (6), butter (2) and tea (1)", 3),
    ("whether the Moon is larger than Earth, with a confidence between 0 and 1", 4),
    ("whether water boils at 100 C at sea level, confidence between 0 and 1", 4),
    ("the planets Mars, Jupiter and Saturn with their order from the Sun", 5),
    ("a book: title Emma, author Jane Austen, year 1815, genres romance and satire", 6),
    ("a book: title Neuromancer, author William Gibson, year 1984, genres cyberpunk and thriller", 6),
    ("a weather forecast for Friday: high 30, low 21, no rain", 7),
    ("a weather forecast for Sunday: high 12, low 4, rain likely", 7),
    ("a rectangle 2.5 by 8 with its area", 8), ("a rectangle 10 by 10 with its area", 8),
    ("a user record: id 7, email x@y.org, role viewer, inactive", 9),
    ("a user record: id 1001, email ops@example.net, roles editor and viewer, active", 9),
    ("three continents", 1), ("a person named Marie Curie born in 1867 in Warsaw", 0),
]


def cases() -> list[tuple[str, dict]]:
    out = list(CASES)
    for text, i in VARIANTS:
        out.append((text, CASES[i][1]))
    return out


def validate(value, schema: dict, path: str = "$") -> list[str]:
    errs: list[str] = []
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return [f"{path}: not an object"]
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}.{k}: missing")
        props = schema.get("properties", {})
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}.{k}: not allowed")
    elif t == "array":
        if not isinstance(value, list):
            return [f"{path}: not an array"]
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        for i, v in enumerate(value):
            errs += validate(v, schema.get("items", {}), f"{path}[{i}]")
    elif t == "string":
        if not isinstance(value, str):
            errs.append(f"{path}: not a string")
    elif t == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            errs.append(f"{path}: not an integer")
    elif t == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errs.append(f"{path}: not a number")
    elif t == "boolean":
        if not isinstance(value, bool):
            errs.append(f"{path}: not a boolean")
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: above {schema['maximum']}")
    return errs


def extract_json(text: str):
    text = strip_think(text)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError("no JSON")
    dec = json.JSONDecoder()
    value, _ = dec.raw_decode(text[start:])
    return value


def run(client: Client, *, guided: bool = False, workers: int = 4) -> Result:
    t0 = time.time()
    items = cases()

    def one(item: tuple[str, dict]) -> tuple[bool, str]:
        text, schema = item
        prompt = (f"Return JSON only (no prose, no code fence) describing {text}. It must satisfy this JSON schema:\n"
                  f"{json.dumps(schema)}")
        kw = {"response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}} if guided else {}
        try:
            reply = client.ask(prompt, max_tokens=400, **kw)
            errs = validate(extract_json(reply), schema)
        except Exception as exc:                 # noqa: BLE001
            errs = [str(exc)[:80]]
        return not errs, "; ".join(errs[:2])

    with ThreadPoolExecutor(workers) as pool:
        outs = list(pool.map(one, items))
    ok = sum(1 for hit, _ in outs if hit)
    bad = [f"#{i}: {why}" for i, (hit, why) in enumerate(outs) if not hit][:5]
    return Result("json-guided" if guided else "json", ok / len(items), len(items), ok, time.time() - t0, bad or ["all valid"])
