"""python3 -m spark <command>: the recipe's one entry point (stdlib only, Python 3.11+)."""

from __future__ import annotations

import argparse
import sys

from . import config, docker, serve, weights
from .nodes import Node, log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m spark", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="check the settings and print them; touches nothing")
    up = sub.add_parser("up", help="image, weights, both ranks, wait for the API")
    up.add_argument("--no-download", action="store_true", help="refuse to download or convert; serve what is there")
    sub.add_parser("down", help="stop rank 0, then rank 1")
    sub.add_parser("status", help="containers and /health")
    lg = sub.add_parser("logs", help="a rank's container log")
    lg.add_argument("--rank", type=int, default=0)
    lg.add_argument("--tail", type=int, default=80)
    img = sub.add_parser("image", help="build the image on the head (and check its labels)")
    img.add_argument("--rebuild", action="store_true", help="rebuild when the image carries another patch")
    sub.add_parser("convert", help="download the snapshot and convert it on the head (in the engine's image)")
    a = ap.parse_args(argv)
    try:
        s = config.load()
    except config.ConfigError as exc:
        print(f"settings: {exc}", file=sys.stderr)
        return 2
    try:
        if a.cmd == "validate":
            print(s.summary())
            return 0
        if a.cmd == "up":
            serve.up(s, download=not a.no_download)
        elif a.cmd == "down":
            serve.down(s)
        elif a.cmd == "status":
            print(serve.status(s))
        elif a.cmd == "logs":
            node = Node("head") if a.rank == 0 else Node("worker", s.worker)
            print(docker.logs(node, s.container, a.tail))
        elif a.cmd == "image":
            docker.ensure_image(s, Node("head"), rebuild=a.rebuild)
        elif a.cmd == "convert":
            docker.ensure_image(s, Node("head"))
            weights.ensure_converted(s, Node("head"))
    except RuntimeError as exc:
        log(f"failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
