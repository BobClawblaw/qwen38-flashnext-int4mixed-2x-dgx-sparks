"""The settings guards and the engine command line, with no Docker and no GPU: python3 -m unittest discover -s tests -q"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
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
        self.assertEqual((s.profile, s.context, s.parallel, s.yarn, s.drafts), ("concurrent", 1048576, 16, True, True))
        self.assertEqual(hashlib.sha256(s.patch_path.read_bytes()).hexdigest(), s.patch_sha256)

    def test_profiles_fill_only_what_nothing_else_set(self) -> None:
        self.assertEqual(load(PROFILE="serial").parallel, 1)
        self.assertEqual(load(PROFILE="concurrent", PARALLEL="4").parallel, 4)
        self.assertEqual(load(CONTEXT="524288").context, 524288)
        self.assertEqual(load(YARN="0", CONTEXT="262144").context, 262144)
        self.refused("must be one of", PROFILE="fast")
        self.refused("must be one of", PROFILE="long")

    def test_windows(self) -> None:
        self.refused("exceeds the trained window", YARN="0", CONTEXT="262400")
        self.refused("with yarn must be above", CONTEXT="262144")
        self.refused("with yarn must be above", CONTEXT="1048832")
        self.refused("multiple of 256", YARN="0", CONTEXT="1000")

    def test_integers_and_flags(self) -> None:
        self.refused("not a decimal integer", MAX_TOKENS="0100")
        self.refused("not a decimal integer", PARALLEL="x")
        self.refused("outside", PARALLEL="17")
        self.refused("0/1 or true/false", VISION="yes")
        self.refused("must be one of", KV_DTYPE="fp8")
        self.refused("must be one of", KV_DTYPE="bf16")
        self.refused("40-hex", REVISION="main")

    def test_extra_args_cannot_reset_owned_flags(self) -> None:
        for word in ("--tp 1", "--context=8192", "--paral 8", "--vis", "--no-dr", "--port 1"):
            self.refused("which the recipe passes itself", EXTRA_ARGS=word)
        self.assertEqual(load(EXTRA_ARGS="--temperature 0.6").extra_args, "--temperature 0.6")
        self.refused("not KEY=VALUE", EXTRA_ENV="NCCL_PROTO")

    def test_converted_repo_needs_a_revision(self) -> None:
        self.refused("converted_revision", CONVERTED_REPO="x/y", CONVERTED_REVISION="")
        self.assertEqual(load(CONVERTED_REPO="x/y", CONVERTED_REVISION="a" * 40).converted_repo, "x/y")

    def test_removed_settings_say_what_replaced_them(self) -> None:
        self.refused("only the native engine", ENGINE="python")
        self.refused("two ranks", TP="1")
        self.refused("drafts = false", MTP_DRAFTS="0")
        self.refused("it is patch now", NATIVE_PATCH="x.patch")
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "cluster.toml"
            f.write_text('[engine]\nnative_image = "x"\n')
            with self.assertRaises(config.ConfigError) as got:
                config.load(cluster_file=f, env=CLEAN)
            self.assertIn("it is image now", str(got.exception))

    def test_cache_folder(self) -> None:
        self.assertTrue(str(load().converted_dir).endswith(load().cache_folder + "/affine-experts-v2"))
        self.refused("one folder name", CACHE_FOLDER="a/b")

    def test_patch_pin(self) -> None:
        self.refused("not a sha256", PATCH_SHA256="abc")
        self.refused("not the pinned", PATCH_SHA256="b" * 64)
        self.refused("is missing", PATCH="nope.patch")


class ServeArgsTests(unittest.TestCase):
    def test_both_ranks_share_the_engine_settings(self) -> None:
        s = load()
        r0, r1 = s.native_args(0), s.native_args(1)
        for flag in ("--context", "--tp", "--master", "--master-port"):
            self.assertEqual(r0[r0.index(flag) + 1], r1[r1.index(flag) + 1], flag)
        self.assertEqual(r1[r1.index("--rank") + 1], "1")
        for flag in ("--name", "--host", "--port", "--max-tokens", "--no-thinking", "--tool-system", "--parallel"):
            self.assertIn(flag, r0)
            self.assertNotIn(flag, r1)                   # rank 1 serves no HTTP
        self.assertEqual(r0[r0.index("--parallel") + 1], "16")
        self.assertIn("--vision", r0)
        self.assertIn("--vision", r1)                    # both ranks admit the same geometry; the tower on rank 0
        self.assertIn("--vision-urls", r0)               # on by default; rank 0 fetches
        self.assertNotIn("--vision-urls", r1)
        self.assertNotIn("--vision-urls", load(VISION_URLS="0").native_args(0))
        self.assertNotIn("--kv-dtype", r0)               # int8 is the engine's default
        for rank in (0, 1):                              # both ranks keep the same cache
            r = load(KV_DTYPE="int4").native_args(rank)
            self.assertEqual(r[r.index("--kv-dtype") + 1], "int4")

    def test_switches(self) -> None:
        self.assertIn("--no-drafts", load(DRAFTS="0").native_args(1))
        self.assertNotIn("--no-drafts", load().native_args(0))
        self.assertIn("--thinking", load(THINKING="1").native_args(0))
        r0 = load(PROFILE="serial").native_args(0)
        self.assertEqual(r0[r0.index("--parallel") + 1], "1")
        r0 = load(VISION_URLS="1", VISION_MAX_IMAGES="8").native_args(0)
        self.assertIn("--vision-urls", r0)
        self.assertEqual(r0[r0.index("--vision-max-images") + 1], "8")
        self.assertNotIn("--vision", load(VISION="0").native_args(0))
        self.assertEqual(load(EXTRA_ARGS="--temperature 0").native_args(0)[-2:], ["--temperature", "0"])

    def test_model_folder_follows_the_profile(self) -> None:
        self.assertTrue(load(YARN="0", CONTEXT="262144").model_in_container.endswith("/affine-experts-v2"))
        self.assertTrue(load().model_in_container.endswith("/long-1048576"))

    def test_container_env(self) -> None:
        env = load(EXTRA_ENV="NCCL_PROTO=LL", MEMORY_RESERVE_GIB="12").container_env()
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["NCCL_PROTO"], "LL")
        self.assertEqual(env["TENSORFOLD_MEMORY_RESERVE_GIB"], "12.0")
        self.assertEqual(env["NCCL_IB_HCA"], load().hca)


if __name__ == "__main__":
    unittest.main()
