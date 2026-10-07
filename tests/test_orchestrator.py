"""The orchestrator's pure parts, offline: the checkpoint completeness check, the conversion marker, the settle stamp
and the docker command line (no Docker, no ssh, no GPU)."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark import config, docker, serve, weights  # noqa: E402
from spark.nodes import Node  # noqa: E402

CLEAN = {k: v for k, v in dict(__import__("os").environ).items()
         if k not in {f.name.upper() for f in config.Settings.__dataclass_fields__.values()}}


def settings(**env: str) -> config.Settings:
    return config.load(cluster_file="none", env={**CLEAN, **env})


def shard(path: Path, nbytes: int, truncate: int = 0) -> None:
    header = json.dumps({"w": {"dtype": "U8", "shape": [nbytes], "data_offsets": [0, nbytes]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * (nbytes - truncate))


class SnapshotCheck(unittest.TestCase):
    def test_missing_truncated_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in weights.REQUIRED:
                if name != "model.safetensors.index.json":
                    (d / name).write_text("x")
            (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}))
            shard(d / "a.safetensors", 64)
            self.assertIn("b.safetensors (FileNotFoundError)", weights.snapshot_problems(d)[0])
            shard(d / "b.safetensors", 64, truncate=8)
            self.assertEqual(weights.snapshot_problems(d), ["b.safetensors (truncated)"])
            shard(d / "b.safetensors", 64)
            self.assertEqual(weights.snapshot_problems(d), [])
            (d / "tokenizer.json").unlink()
            self.assertEqual(weights.snapshot_problems(d), ["tokenizer.json"])


class Markers(unittest.TestCase):
    def test_marker_names_the_source_or_the_hub_copy(self) -> None:
        s = settings()
        self.assertTrue(weights.conversion_marker(s).startswith(s.revision + " "))
        self.assertEqual(weights.conversion_marker(s).split()[1], weights.converter_sha())
        h = settings(CONVERTED_REPO="x/y", CONVERTED_REVISION="c" * 40)
        self.assertEqual(weights.conversion_marker(h), "hub x/y@" + "c" * 40)


class Settle(unittest.TestCase):
    def test_waits_out_the_remainder_after_a_recent_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = settings(TF_CACHE=tmp, SETTLE_SECONDS="30")
            (Path(tmp) / "stopped_at").write_text(f"{time.time() - 28:.0f}\n")
            with mock.patch("spark.serve.time.sleep") as sleep:
                serve.settle(s)
            self.assertTrue(sleep.called)
            self.assertTrue(1.0 <= sleep.call_args[0][0] <= 3.0)
            (Path(tmp) / "stopped_at").write_text("0\n")
            with mock.patch("spark.serve.time.sleep") as sleep:
                serve.settle(s)
            self.assertFalse(sleep.called)
            (Path(tmp) / "stopped_at").unlink()
            with mock.patch("spark.serve.time.sleep") as sleep:
                serve.settle(s)
            self.assertFalse(sleep.called)


class DockerCommand(unittest.TestCase):
    def test_run_rank_builds_the_expected_command(self) -> None:
        s = settings(PROFILE="long")
        seen: list[list[str]] = []

        class Fake(Node):
            def run(self, argv, **kw):          # noqa: ANN001
                seen.append(list(argv))
                return mock.Mock(returncode=0, stdout="", stderr="")

        docker.run_rank(s, Fake("worker", "user@host"), 1, src_mount=Path("/src/tf"))
        argv = seen[-1]
        self.assertEqual(argv[:3], ["docker", "run", "-d"])
        for flag in ("--init", "--gpus", "--network", "--ipc", "--device", "--cap-add", "--ulimit"):
            self.assertIn(flag, argv)
        self.assertIn(f"{s.hf_cache}:{config.CONTAINER_HF}:ro", argv)
        self.assertIn("/src/tf:/usr/local/lib/python3.12/dist-packages/tensorfold:ro", argv)
        self.assertIn("NCCL_IB_HCA=" + s.hca, argv)
        i = argv.index("serve")
        self.assertEqual(argv[i + 1], s.model_in_container)
        self.assertTrue(argv[i + 1].endswith("/long-1048576"))
        self.assertEqual(argv[argv.index("--rank") + 1], "1")
        self.assertNotIn("--port", argv)


if __name__ == "__main__":
    unittest.main()
