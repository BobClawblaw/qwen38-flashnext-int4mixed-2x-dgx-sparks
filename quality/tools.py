"""Tool calls: sixty prompts against a fixed tool set, scored on the exact function and the exact argument set.

A case passes when the reply carries exactly the expected call(s): same function, arguments equal as JSON after
trimming strings, no extra arguments (an optional argument the user did not give must stay absent), and no call
at all where none is expected. Parallel calls are checked as a set.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

from .common import Client, Result


def _tool(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS = [
    _tool("get_weather", "Current weather for a city.",
          {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Optional. Only when the user names a unit."}}, ["city"]),
    _tool("convert_currency", "Convert an amount between currencies.",
          {"amount": {"type": "number"}, "from_currency": {"type": "string", "description": "ISO code"}, "to_currency": {"type": "string", "description": "ISO code"}}, ["amount", "from_currency", "to_currency"]),
    _tool("create_reminder", "Create a reminder.",
          {"text": {"type": "string"}, "when": {"type": "string", "description": "ISO 8601 date or datetime"}, "repeat": {"type": "string", "enum": ["daily", "weekly", "monthly"], "description": "Optional."}}, ["text", "when"]),
    _tool("search_files", "Search file names in a folder.",
          {"query": {"type": "string"}, "folder": {"type": "string"}, "extension": {"type": "string", "description": "Optional, like pdf"}}, ["query"]),
    _tool("send_email", "Send an email.",
          {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}, "cc": {"type": "array", "items": {"type": "string"}, "description": "Optional."}}, ["to", "subject", "body"]),
    _tool("calculate", "Evaluate an arithmetic expression.", {"expression": {"type": "string"}}, ["expression"]),
    _tool("translate", "Translate text.",
          {"text": {"type": "string"}, "target_language": {"type": "string"}, "formal": {"type": "boolean", "description": "Optional register; only when the user asks for formal or informal."}}, ["text", "target_language"]),
    _tool("set_timer", "Start a countdown timer.", {"seconds": {"type": "integer"}, "label": {"type": "string", "description": "Optional."}}, ["seconds"]),
    _tool("book_table", "Reserve a restaurant table.",
          {"restaurant": {"type": "string"}, "people": {"type": "integer"}, "date": {"type": "string"}, "time": {"type": "string"}, "notes": {"type": "string", "description": "Optional."}}, ["restaurant", "people", "date", "time"]),
    _tool("stock_price", "Latest price of a ticker.", {"ticker": {"type": "string"}}, ["ticker"]),
    _tool("add_to_cart", "Add a product to the cart.", {"product_id": {"type": "string"}, "quantity": {"type": "integer"}}, ["product_id", "quantity"]),
    _tool("get_directions", "Directions between two places.",
          {"origin": {"type": "string"}, "destination": {"type": "string"}, "mode": {"type": "string", "enum": ["driving", "walking", "transit", "cycling"], "description": "Optional; only when the user says how they travel."}}, ["origin", "destination"]),
]

# (prompt, expected calls as [(name, args)]; [] = no tool call)
CASES: list[tuple[str, list[tuple[str, dict]]]] = [
    ("What's the weather in Lisbon?", [("get_weather", {"city": "Lisbon"})]),
    ("Weather in Oslo in fahrenheit please.", [("get_weather", {"city": "Oslo", "unit": "fahrenheit"})]),
    ("Is it raining in Tokyo right now?", [("get_weather", {"city": "Tokyo"})]),
    ("Convert 250 euros to US dollars.", [("convert_currency", {"amount": 250, "from_currency": "EUR", "to_currency": "USD"})]),
    ("How much is 1200 JPY in GBP?", [("convert_currency", {"amount": 1200, "from_currency": "JPY", "to_currency": "GBP"})]),
    ("Remind me to call the dentist on 2026-11-03.", [("create_reminder", {"text": "call the dentist", "when": "2026-11-03"})]),
    ("Set a weekly reminder to water the plants starting 2026-10-10.", [("create_reminder", {"text": "water the plants", "when": "2026-10-10", "repeat": "weekly"})]),
    ("Find files about the budget in /docs.", [("search_files", {"query": "budget", "folder": "/docs"})]),
    ("Search my files for 'invoice'.", [("search_files", {"query": "invoice"})]),
    ("Look for PDF files named roadmap in /shared.", [("search_files", {"query": "roadmap", "folder": "/shared", "extension": "pdf"})]),
    ("Email sam@example.com with subject 'Lunch' saying 'Noon at the usual place?'", [("send_email", {"to": "sam@example.com", "subject": "Lunch", "body": "Noon at the usual place?"})]),
    ("Send an email to ops@example.com, cc lee@example.com, subject 'Outage', body 'Resolved at 10:40.'", [("send_email", {"to": "ops@example.com", "subject": "Outage", "body": "Resolved at 10:40.", "cc": ["lee@example.com"]})]),
    ("What is 48 * 17 + 3?", [("calculate", {"expression": "48 * 17 + 3"})]),
    ("Compute (1250 - 375) / 25.", [("calculate", {"expression": "(1250 - 375) / 25"})]),
    ("Translate 'good morning' into Spanish.", [("translate", {"text": "good morning", "target_language": "Spanish"})]),
    ("Translate 'thank you very much' to Japanese, formal register.", [("translate", {"text": "thank you very much", "target_language": "Japanese", "formal": True})]),
    ("How do you say 'where is the station' in German?", [("translate", {"text": "where is the station", "target_language": "German"})]),
    ("Set a timer for 5 minutes.", [("set_timer", {"seconds": 300})]),
    ("Start a 90 second timer called eggs.", [("set_timer", {"seconds": 90, "label": "eggs"})]),
    ("Book a table for 4 at Luigi's on 2026-12-24 at 19:30.", [("book_table", {"restaurant": "Luigi's", "people": 4, "date": "2026-12-24", "time": "19:30"})]),
    ("Reserve Nobu for two on 2026-10-20 at 20:00, note: window seat.", [("book_table", {"restaurant": "Nobu", "people": 2, "date": "2026-10-20", "time": "20:00", "notes": "window seat"})]),
    ("What's NVDA trading at?", [("stock_price", {"ticker": "NVDA"})]),
    ("Price of AAPL?", [("stock_price", {"ticker": "AAPL"})]),
    ("Add 3 of product SKU-778 to my cart.", [("add_to_cart", {"product_id": "SKU-778", "quantity": 3})]),
    ("Put one B-1021 in the cart.", [("add_to_cart", {"product_id": "B-1021", "quantity": 1})]),
    ("Directions from Berlin Hauptbahnhof to Alexanderplatz.", [("get_directions", {"origin": "Berlin Hauptbahnhof", "destination": "Alexanderplatz"})]),
    ("How do I cycle from Oxford to Abingdon?", [("get_directions", {"origin": "Oxford", "destination": "Abingdon", "mode": "cycling"})]),
    ("Walking directions from the Louvre to Notre-Dame.", [("get_directions", {"origin": "the Louvre", "destination": "Notre-Dame", "mode": "walking"})]),
    # parallel calls
    ("Weather in Rome and in Madrid?", [("get_weather", {"city": "Rome"}), ("get_weather", {"city": "Madrid"})]),
    ("Give me the prices of MSFT and TSLA.", [("stock_price", {"ticker": "MSFT"}), ("stock_price", {"ticker": "TSLA"})]),
    ("Set two timers: 10 minutes for pasta and 3 minutes for tea.", [("set_timer", {"seconds": 600, "label": "pasta"}), ("set_timer", {"seconds": 180, "label": "tea"})]),
    ("Convert 100 USD to EUR and 100 USD to CHF.", [("convert_currency", {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}), ("convert_currency", {"amount": 100, "from_currency": "USD", "to_currency": "CHF"})]),
    # no tool should be called
    ("Who wrote Pride and Prejudice?", []),
    ("Explain what a mutex is in two sentences.", []),
    ("Write a limerick about a cat.", []),
    ("What tools can you use?", []),
    ("Is 97 a prime number? Just answer yes or no.", []),
    ("Give me a synonym for 'quick'.", []),
    ("What is the capital of Australia?", []),
    ("Summarize the rules of tic-tac-toe.", []),
    # more single calls, varied phrasing
    ("I need the weather for Cape Town, celsius.", [("get_weather", {"city": "Cape Town", "unit": "celsius"})]),
    ("Change 75 CAD into AUD for me.", [("convert_currency", {"amount": 75, "from_currency": "CAD", "to_currency": "AUD"})]),
    ("Reminder: submit the report, 2026-10-15T09:00.", [("create_reminder", {"text": "submit the report", "when": "2026-10-15T09:00"})]),
    ("Daily reminder to stretch from 2026-10-08.", [("create_reminder", {"text": "stretch", "when": "2026-10-08", "repeat": "daily"})]),
    ("Find docx files mentioning 'contract'.", [("search_files", {"query": "contract", "extension": "docx"})]),
    ("Mail hr@example.com: subject 'Leave request', body 'May I take 2026-11-05 off?'", [("send_email", {"to": "hr@example.com", "subject": "Leave request", "body": "May I take 2026-11-05 off?"})]),
    ("What's 2 to the power of 20?", [("calculate", {"expression": "2 ** 20"})]),
    ("Translate 'see you tomorrow' into Italian, informally.", [("translate", {"text": "see you tomorrow", "target_language": "Italian", "formal": False})]),
    ("Timer, 45 seconds.", [("set_timer", {"seconds": 45})]),
    ("Table for 6 at The Grove, 2026-10-31, 18:00.", [("book_table", {"restaurant": "The Grove", "people": 6, "date": "2026-10-31", "time": "18:00"})]),
    ("Check the AMZN price.", [("stock_price", {"ticker": "AMZN"})]),
    ("Add two units of P-55 to the cart.", [("add_to_cart", {"product_id": "P-55", "quantity": 2})]),
    ("Transit directions from Shibuya to Narita Airport.", [("get_directions", {"origin": "Shibuya", "destination": "Narita Airport", "mode": "transit"})]),
    ("Drive me from Austin to Dallas.", [("get_directions", {"origin": "Austin", "destination": "Dallas", "mode": "driving"})]),
    ("Weather in Reykjavik and the price of BTC-USD.", [("get_weather", {"city": "Reykjavik"}), ("stock_price", {"ticker": "BTC-USD"})]),
    ("Translate 'open the window' to French.", [("translate", {"text": "open the window", "target_language": "French"})]),
    ("Remind me to renew the passport on 2027-01-15.", [("create_reminder", {"text": "renew the passport", "when": "2027-01-15"})]),
    ("How much is 999 SEK in NOK?", [("convert_currency", {"amount": 999, "from_currency": "SEK", "to_currency": "NOK"})]),
    ("What's 15% of 240?", [("calculate", {"expression": "0.15 * 240"})]),
    ("Thanks, that's all for today.", []),
]


def _norm(v):
    if isinstance(v, str):                       # free text: case and a trailing full stop or question mark are not errors
        return v.strip().lower().rstrip(".?!")
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _calls(d: dict) -> list[tuple[str, dict]]:
    out = []
    for c in d["choices"][0]["message"].get("tool_calls") or []:
        fn = c.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {"_unparsable": fn.get("arguments")}
        out.append((fn.get("name"), _norm(args)))
    return out


def _match(got: list, want: list) -> bool:
    if len(got) != len(want):
        return False
    pool = [(n, _norm(a)) for n, a in want]
    for call in [(n, _norm(a)) for n, a in got]:
        if call in pool:
            pool.remove(call)
        else:
            return False
    return True


def run(client: Client, *, workers: int = 4) -> Result:
    t0 = time.time()

    def one(case: tuple[str, list]) -> tuple[bool, str]:
        prompt, want = case
        d = client.chat([{"role": "user", "content": prompt}], tools=TOOLS, max_tokens=300)
        got = _calls(d)
        ok = _match(got, want)
        return ok, "" if ok else f"{prompt[:40]!r}: got {got}"

    with ThreadPoolExecutor(workers) as pool:
        outs = list(pool.map(one, CASES))
    ok = sum(1 for hit, _ in outs if hit)
    return Result("tools", ok / len(CASES), len(CASES), ok, time.time() - t0,
                  [why for hit, why in outs if not hit][:4] or ["all exact"])
