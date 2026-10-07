"""The two machines: the head (where this runs) and the worker (over ssh), with one small command interface."""

from __future__ import annotations

import shlex
import subprocess
import sys
import time


def log(*parts: object) -> None:
    print(time.strftime("%H:%M:%S"), "==>", *parts, flush=True)


class Node:
    """Runs commands locally or over ssh (BatchMode: keys only, no prompts)."""

    def __init__(self, name: str, ssh: str | None = None) -> None:
        self.name, self.ssh = name, ssh

    def __repr__(self) -> str:
        return f"Node({self.name}{' via ' + self.ssh if self.ssh else ''})"

    def run(self, argv: list[str], *, check: bool = True, timeout: int | None = None,
            stdin: bytes | None = None) -> subprocess.CompletedProcess:
        """Run ``argv`` on this node; stdout and stderr are captured (text)."""
        if self.ssh:
            argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.ssh, "--",
                    " ".join(shlex.quote(a) for a in argv)]
        proc = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)
        out = subprocess.CompletedProcess(argv, proc.returncode, proc.stdout.decode(errors="replace"),
                                          proc.stderr.decode(errors="replace"))
        if check and out.returncode:
            raise RuntimeError(f"{self.name}: `{' '.join(argv[-1:] if self.ssh else argv)}` failed "
                               f"({out.returncode}): {out.stderr.strip() or out.stdout.strip()}")
        return out

    def script(self, text: str, *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
        """Run a bash script (its text goes over stdin, so no quoting crosses the ssh boundary)."""
        return self.run(["bash", "-s"], check=check, timeout=timeout, stdin=text.encode())

    def reachable(self) -> bool:
        try:
            return self.run(["true"], check=False, timeout=20).returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def stream(self, argv: list[str]) -> int:
        """Run with inherited stdout (long copies: show nothing, return the exit code)."""
        if self.ssh:
            argv = ["ssh", "-o", "BatchMode=yes", self.ssh, "--", " ".join(shlex.quote(a) for a in argv)]
        return subprocess.call(argv, stdout=sys.stderr)


def pipe(producer: list[str], consumer_node: Node, consumer: list[str]) -> None:
    """``producer | ssh node consumer`` (an image or a folder going over the link)."""
    if consumer_node.ssh:
        consumer = ["ssh", "-o", "BatchMode=yes", consumer_node.ssh, "--", " ".join(shlex.quote(a) for a in consumer)]
    p1 = subprocess.Popen(producer, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(consumer, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p1.stdout is not None
    p1.stdout.close()
    out, err = p2.communicate()
    p1.wait()
    if p1.returncode or p2.returncode:
        raise RuntimeError(f"pipe failed: {' '.join(producer)} -> {consumer_node.name}: {err.decode(errors='replace')[-400:]}")
