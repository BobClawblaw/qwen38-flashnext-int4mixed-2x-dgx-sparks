"""Bring the pair up and down: image, weights, both ranks, readiness, the memory watchdog."""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.request
from pathlib import Path

from . import docker, weights
from .config import Settings
from .nodes import Node, log

MEMGUARD = r"""
# Host memory watchdog (unified memory: a thrashing node takes minutes to reach the OOM killer, which then picks
# small services first). Kill the serve when available RAM and free swap are both low for three samples.
name=$1; min_avail=$(( $2 * 1024 )); min_swap=$(( $3 * 1024 )); hits=0; n=0
while :; do
  avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo); swapfree=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
  if (( avail < min_avail && swapfree < min_swap )); then hits=$((hits+1)); else hits=0; fi
  if (( hits >= 3 )); then
    echo "$(date -Is) MemAvailable=${avail}kB SwapFree=${swapfree}kB: docker kill $name"
    logger -t tf-qwen38-int4mixed-memguard "MemAvailable=${avail}kB SwapFree=${swapfree}kB: docker kill $name" || true
    docker kill "$name" >/dev/null 2>&1 || true; exit 0
  fi
  n=$((n+1))
  if (( n % 8 == 0 )) && [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then exit 0; fi
  sleep 2
done
"""


def nodes(s: Settings) -> tuple[Node, Node | None]:
    head = Node("head")
    worker = Node("worker", s.worker) if s.tp == 2 else None
    return head, worker


def _stamp(s: Settings) -> Path:
    return s.tf_cache / "stopped_at"


def settle(s: Settings) -> None:
    """Wait out settle_seconds since the last stop before a rank starts (the worker's first ibv_reg_mr fails sooner)."""
    try:
        since = time.time() - float(_stamp(s).read_text().strip())
    except (OSError, ValueError):
        return
    if since < s.settle_seconds:
        log(f"Waiting {s.settle_seconds - since:.0f}s more after the last stop for the ranks' memory to settle")
        time.sleep(s.settle_seconds - since)


def drop_caches(node: Node) -> None:
    """sync, drop_caches, compact_memory where sudo is passwordless; the NIC wants higher-order pages for its tables."""
    out = node.script("sudo -n true 2>/dev/null && { sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; "
                      "echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1 || true; echo ran; } || echo skipped",
                      check=False)
    log(f"{node.name}: drop_caches {out.stdout.strip() or 'skipped'}")


def start_memguard(s: Settings, node: Node) -> None:
    if not s.memguard:
        return
    logf = s.tf_cache / "memguard.log"
    script = (f"mkdir -p {s.tf_cache}; setsid nohup bash -c {json.dumps(MEMGUARD)!s} memguard {s.container} "
              f"{s.memguard_min_avail_mb} {s.memguard_min_swap_free_mb} >> {logf} 2>&1 < /dev/null & disown; echo started")
    node.script(script, check=False)


def port_busy(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def health(s: Settings, timeout: float = 3) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{s.port}/health", timeout=timeout) as r:
            return json.load(r)
    except (OSError, ValueError):
        return None


def wait_ready(s: Settings, head: Node, worker: Node | None, limit: int = 3600) -> None:
    log(f"Waiting for http://127.0.0.1:{s.port}/health and /v1/models")
    t0 = time.time()
    n = 0
    while time.time() - t0 < limit:
        h = health(s)
        if h and h.get("ok"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{s.port}/v1/models", timeout=3) as r:
                    body = r.read().decode()
            except OSError:
                body = ""
            if s.served_name in body:
                log(f"Ready: http://0.0.0.0:{s.port}/v1 (context={s.context}, {time.time() - t0:.0f}s)")
                return
        if docker.state(head, s.container) != "running":
            raise RuntimeError(f"rank 0 exited:\n{docker.logs(head, s.container, 40)}")
        if worker is not None and n % 6 == 5 and docker.state(worker, s.container) != "running":
            docker.stop(head, s.container)
            raise RuntimeError(f"rank 1 exited on {worker.name}:\n{docker.logs(worker, s.container, 40)}")
        n += 1
        if n % 12 == 0:
            log(f"still loading ({time.time() - t0:.0f}s): docker logs -f {s.container}")
        time.sleep(5)
    raise RuntimeError(f"not ready after {limit}s")


def down(s: Settings) -> None:
    """Rank 0 first (its store closes and rank 1 exits by itself), then rank 1, then the stop stamp."""
    head, worker = nodes(s)
    docker.stop(head, s.container)
    if worker is not None and worker.reachable():
        for _ in range(30):                  # rank 1 leaves when rank 0's store closes
            if docker.state(worker, s.container) != "running":
                break
            time.sleep(1)
        docker.stop(worker, s.container)
    s.tf_cache.mkdir(parents=True, exist_ok=True)
    _stamp(s).write_text(f"{time.time():.0f}\n")


def up(s: Settings, *, src_mount: dict[str, Path] | None = None, download: bool = True) -> None:
    """``src_mount``: {"head": path, "worker": path} tensorfold source trees mounted over the image's (development)."""
    head, worker = nodes(s)
    for node in (head, worker) if worker else (head,):
        if node is worker and not worker.reachable():
            raise RuntimeError(f"cannot ssh to the worker {s.worker}")
        busy = docker.foreign_gpu_containers(node, s.container)
        if busy:
            raise RuntimeError(f"{node.name}: {', '.join(busy)} hold the GPU or InfiniBand; stop them first")
    if port_busy(s.port) and docker.state(head, s.container) != "running":
        raise RuntimeError(f"port {s.port} is in use by something else")
    docker.ensure_image(s, head)
    if download:
        weights.ensure_converted(s, head)
    elif not weights.converted_ready(s, head):
        raise RuntimeError("the converted checkpoint is missing; run `python3 -m spark convert`")
    stopped = docker.stop(head, s.container)
    if worker is not None:
        # a 25 GB image load leaves the worker's memory fragmented: NCCL's first ibv_reg_mr then fails with ENOMEM
        # (seen on the first start of this recipe), so a copy counts as a stop for the settle below
        stopped |= docker.sync_image(s, head, worker)
        weights.sync_converted(s, head, worker)
        stopped |= docker.stop(worker, s.container)
    if stopped:
        _stamp(s).write_text(f"{time.time():.0f}\n")
    settle(s)
    order = ([worker] if worker else []) + [head]
    for node in order:
        drop_caches(node)
        if s.profile == "long":
            weights.ensure_long_profile(s, node)
        docker.run_rank(s, node, 1 if node is worker else 0, src_mount=(src_mount or {}).get(node.name))
        start_memguard(s, node)
        if node is worker:
            time.sleep(5)
            if docker.state(worker, s.container) != "running":
                raise RuntimeError(f"rank 1 died at start:\n{docker.logs(worker, s.container, 40)}")
    wait_ready(s, head, worker)


def status(s: Settings) -> str:
    head, worker = nodes(s)
    lines = [f"head:   {s.container} {docker.state(head, s.container)}"]
    if worker is not None:
        lines.append(f"worker: {s.container} {docker.state(worker, s.container) if worker.reachable() else 'unreachable'}")
    h = health(s)
    if h:
        st = h.get("streams", {})
        lines.append(f"api:    ok, context {h.get('context_length')}, streams {st.get('decoding')} decoding / "
                     f"{st.get('prefilling')} prefilling of {st.get('max')}, {h.get('requests_total')} requests served")
    else:
        lines.append(f"api:    no answer on port {s.port}")
    return "\n".join(lines)
