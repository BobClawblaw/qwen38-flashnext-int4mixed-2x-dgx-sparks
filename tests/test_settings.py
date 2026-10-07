"""The settings guards and the engine command line, with no Docker and no GPU: python3 -m unittest discover -s tests -q"""

from __future__ import annotations

import hashlib
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spark import config  # noqa: E402

CLEAN = {k: v for k, v in os.environ.items() if k not in {f.name.upper() for f in config.Settings.__dataclass_fields__.values()}}


def load(**env: str) -> config.Settings:
    return config.load(cluster_file="none", env={**CLEAN, **env})


class GuardTests(unittest.TestCase):
    def refused(self, needle: str, **env: str) -> None:
        with self.assertRaises(config.ConfigError) as got:
            load(**env)
        self.assertIn(needle, str(got.exception), env)

    def test_defaults_load_and_name_the_pinned_patch(self) -> None:
        s = load()
        self.assertEqual((s.profile, s.tp, s.context, s.parallel), ("serial", 2, 262144, 1))
        self.assertEqual(hashlib.sha256(s.patch_path.read_bytes()).hexdigest(), s.patch_sha256)

    def test_profiles_fill_only_what_nothing_else_set(self) -> None:
        self.assertEqual(load(PROFILE="concurrent").parallel, 16)
        self.assertEqual(load(PROFILE="concurrent", PARALLEL="4").parallel, 4)
        self.assertEqual(load(PROFILE="long").context, 1048576)
        self.assertEqual(load(PROFILE="long", CONTEXT="524288").context, 524288)
        self.refused("must be one of", PROFILE="fast")

    def test_windows(self) -> None:
        self.refused("exceeds the trained window", CONTEXT="262400")
        self.refused("on the long profile must be above", PROFILE="long", CONTEXT="262144")
        self.refused("on the long profile must be above", PROFILE="long", CONTEXT="1048832")
        self.refused("multiple of 256", CONTEXT="1000")

    def test_integers_and_flags(self) -> None:
        self.refused("not a decimal integer", MAX_TOKENS="0100")
        self.refused("not a decimal integer", PARALLEL="x")
        self.refused("outside", MTP_DRAFTS="16")
        self.refused("outside", TP="4")
        self.refused("0/1 or true/false", VISION="yes")
        self.refused("must be a decimal in [0, 1]", MTP_CONFIDENCE=".7")
        self.refused("must be one of", KV_DTYPE="fp8")
        self.refused("40-hex", REVISION="main")

    def test_extra_args_cannot_reset_owned_flags(self) -> None:
        for word in ("--tp 1", "--context=8192", "--paral 8", "--vis", "--mtp-d 3", "--port 1"):
            self.refused("which the recipe passes itself", EXTRA_ARGS=word)
        self.refused("not used by this checkpoint", EXTRA_ARGS="--prefill-fp8")
        self.assertEqual(load(EXTRA_ARGS="--temperature 0.6 --alias q").extra_args, "--temperature 0.6 --alias q")
        self.refused("not KEY=VALUE", EXTRA_ENV="NCCL_PROTO")

    def test_converted_repo_needs_a_revision(self) -> None:
        self.refused("converted_revision", CONVERTED_REPO="x/y")
        self.assertEqual(load(CONVERTED_REPO="x/y", CONVERTED_REVISION="a" * 40).converted_repo, "x/y")

    def test_patch_pin(self) -> None:
        self.refused("not a sha256", PATCH_SHA256="abc")
        self.refused("not the pinned", PATCH_SHA256="b" * 64)
        self.refused("is missing", PATCH="nope.patch")


class ServeArgsTests(unittest.TestCase):
    def test_both_ranks_share_the_engine_settings(self) -> None:
        s = load()
        r0, r1 = s.serve_args(0), s.serve_args(1)
        for flag in ("--context", "--kv-dtype", "--mtp-drafts", "--mtp-confidence", "--tp", "--master", "--master-port"):
            self.assertEqual(r0[r0.index(flag) + 1], r1[r1.index(flag) + 1], flag)
        self.assertEqual(r1[r1.index("--rank") + 1], "1")
        for flag in ("--name", "--host", "--port", "--max-tokens", "--no-thinking", "--tool-system"):
            self.assertIn(flag, r0)
            self.assertNotIn(flag, r1)
        self.assertIn("--vision", r0)
        self.assertIn("--vision", r1)                    # both ranks admit the same geometry; the tower on rank 0
        self.assertNotIn("--vision-urls", r0)

    def test_switches(self) -> None:
        r = load(MTP_DRAFTS="0").serve_args(0)
        self.assertIn("--no-drafts", r)
        self.assertNotIn("--mtp-drafts", r)
        self.assertIn("--thinking", load(THINKING="1").serve_args(0))
        self.assertNotIn("--parallel", load().serve_args(0))
        self.assertIn("--parallel", load(PROFILE="concurrent").serve_args(1))
        r0 = load(VISION_URLS="1", VISION_MAX_IMAGES="8").serve_args(0)
        self.assertIn("--vision-urls", r0)
        self.assertEqual(r0[r0.index("--vision-max-images") + 1], "8")
        self.assertNotIn("--vision", load(VISION="0").serve_args(0))
        r = load(TP="1").serve_args(0)
        for flag in ("--tp", "--rank", "--master"):
            self.assertNotIn(flag, r)

    def test_model_folder_follows_the_profile(self) -> None:
        self.assertTrue(load().model_in_container.endswith("/affine-experts-v1"))
        self.assertTrue(load(PROFILE="long").model_in_container.endswith("/long-1048576"))

    def test_container_env(self) -> None:
        env = load(EXTRA_ENV="NCCL_PROTO=LL", MEMORY_RESERVE_GIB="12").container_env()
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["NCCL_PROTO"], "LL")
        self.assertEqual(env["TENSORFOLD_MEMORY_RESERVE_GIB"], "12.0")
        self.assertEqual(env["NCCL_IB_HCA"], load().hca)


if __name__ == "__main__":
    unittest.main()
