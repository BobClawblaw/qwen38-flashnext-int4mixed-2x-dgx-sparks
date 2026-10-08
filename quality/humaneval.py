"""HumanEval: 164 Python functions written from their docstrings, scored by running the dataset's unit tests.

The model's code runs inside a throwaway container (no network, 1 GiB memory, read-only source, 10 s a problem)
built from the recipe's own image, never on the host. pass@1 on greedy replies. The problem file is openai/human-eval
at a pinned commit.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import Client, Result, fetch, strip_think

URL = "https://raw.githubusercontent.com/openai/human-eval/463c980b59e818ace59f6f9803cd92c749ceae61/data/HumanEval.jsonl.gz"
SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
PROMPT = ("Complete the following Python function. Reply with the complete function (signature and body) in one "
          "```python code block, and nothing else.\n\n```python\n{prompt}```")

RUNNER = r'''
import json, os, subprocess, sys
out = {}
for name in sorted(os.listdir("/w")):
    if not name.endswith(".py"):
        continue
    try:
        p = subprocess.run([sys.executable, "-I", "/w/" + name], capture_output=True, timeout=10, cwd="/tmp")
        out[name[:-3]] = "pass" if p.returncode == 0 else "fail: " + p.stderr.decode(errors="replace")[-200:]
    except subprocess.TimeoutExpired:
        out[name[:-3]] = "timeout"
print(json.dumps(out))
'''


def problems() -> list[dict]:
    return [json.loads(l) for l in gzip.decompress(fetch(URL, "HumanEval.jsonl.gz", SHA256).read_bytes()).decode().splitlines()
            if l.strip()]


def extract(reply: str, prompt: str, entry: str) -> str:
    """The function from the reply; a bare body is appended to the prompt's signature."""
    text = strip_think(reply)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    code = max(blocks, key=len) if blocks else text
    if re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", code, re.M):
        head = "\n".join(l for l in prompt.splitlines() if l.startswith(("import ", "from ")))
        return (head + "\n\n" + code) if head else code
    return prompt + code


def program(problem: dict, code: str) -> str:
    return f"{code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"


def execute(programs: dict[str, str], image: str) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        for name, src in programs.items():
            Path(tmp, f"{name}.py").write_text(src)
        Path(tmp, "_runner.txt").write_text(RUNNER)
        os.chmod(tmp, 0o755)
        argv = ["docker", "run", "--rm", "--network", "none", "--memory", "1g", "--pids-limit", "256",
                "--cpus", "4", "--read-only", "--tmpfs", "/tmp:size=256m", "--user", "65534:65534",
                "-v", f"{tmp}:/w:ro", "--entrypoint", "python3", image, "-I", "-c", RUNNER]
        p = subprocess.run(argv, capture_output=True, timeout=60 + 12 * len(programs))
        if p.returncode:
            raise RuntimeError(f"sandbox failed: {p.stderr.decode(errors='replace')[-300:]}")
        return json.loads(p.stdout.decode().strip().splitlines()[-1])


def run(client: Client, *, image: str = "tf-qwen38-int4mixed:py0.6-ed78d6f", workers: int = 4, max_tokens: int = 1024,
        n: int | None = None) -> Result:
    probs = problems()[:n] if n else problems()
    t0 = time.time()

    def one(pr: dict) -> str:
        return extract(client.ask(PROMPT.format(prompt=pr["prompt"]), max_tokens=max_tokens), pr["prompt"], pr["entry_point"])

    with ThreadPoolExecutor(workers) as pool:
        codes = list(pool.map(one, probs))
    progs = {pr["task_id"].replace("/", "_"): program(pr, c) for pr, c in zip(probs, codes)}
    verdicts = execute(progs, image)
    ok = sum(1 for v in verdicts.values() if v == "pass")
    failed = [k for k, v in sorted(verdicts.items()) if v != "pass"]
    return Result("humaneval", ok / len(probs), len(probs), ok, time.time() - t0,
                  [f"pass@1 greedy; failed {failed[:6]}{' ...' if len(failed) > 6 else ''}" if failed else "all pass"],
                  {"verdicts": verdicts})
