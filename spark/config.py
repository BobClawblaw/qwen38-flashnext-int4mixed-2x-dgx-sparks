"""The recipe's settings: recipe.toml defaults, cluster.toml overrides, environment overrides, then the guards.

The engine is TensorFold's native (Zig) release with this recipe's patch, on two ranks.

Every value is validated before anything touches Docker or the worker. A profile (concurrent, serial) fills the
settings it owns only where nothing more specific set them, so PROFILE=concurrent with PARALLEL=4 serves 4 streams.
The window is independent of the profile: yarn = true serves up to 1,048,576 tokens (static YaRN x4), yarn = false
the trained 262,144.
"""

from __future__ import annotations

import hashlib
import os
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("concurrent", "serial")
KV_DTYPES = ("int8", "int4")
CONTAINER_HF = "/cache/huggingface"
CONTAINER_TF = "/cache/tf"
# flags `native_args` builds itself; EXTRA_ARGS may not set them (the ranks would disagree), nor any prefix of them
OWNED_FLAGS = ("--tp", "--rank", "--master", "--master-port", "--host", "--port", "--name", "--context", "--kv-dtype",
               "--parallel", "--thinking", "--no-thinking", "--max-tokens", "--no-update-check", "--no-drafts",
               "--tool-system", "--vision", "--vision-urls", "--vision-max-images")
# settings this recipe no longer reads (the Python engine was removed on 2026-10-10): refused with what replaced them
REMOVED = {
    "engine": "the recipe serves only the native engine now (TensorFold's Python engine is frozen upstream)",
    "native_tensorfold_sha": "it is tensorfold_sha now", "native_patch": "it is patch now",
    "native_patch_sha256": "it is patch_sha256 now", "native_image": "it is image now",
    "tp": "the native engine always serves two ranks",
    "mtp_drafts": "use drafts = false to serve without drafts (the engine sets the depth itself)",
    "mtp_confidence": "the engine sets its draft cut itself",
}


class ConfigError(ValueError):
    """A setting the guards refuse; the message says which and why."""


@dataclass
class Settings:
    # [model]
    repo: str
    revision: str
    served_name: str
    converted: str
    converted_repo: str
    converted_revision: str
    # [engine]
    tensorfold_repo: str
    tensorfold_sha: str              # the native release the image builds from
    patch: str
    patch_sha256: str
    image: str
    base_image: str
    container: str
    # [cluster]
    head_ip: str
    worker: str
    iface: str
    hca: str
    master_port: int
    port: int
    hf_cache: Path
    tf_cache: Path
    cache_folder: str                # tf_cache/<cache_folder>: the converted checkpoint, profile folders
    # [serve]
    profile: str
    yarn: bool
    context: int
    kv_dtype: str
    drafts: bool
    parallel: int
    thinking: bool
    max_tokens: int
    vision: bool
    vision_urls: bool
    vision_max_images: int
    tool_system: str
    memory_reserve_gib: float
    extra_args: str
    extra_env: str
    settle_seconds: int
    memguard: bool
    memguard_min_avail_mb: int
    memguard_min_swap_free_mb: int
    # [yarn]
    yarn_factor: int = 4
    trained_context: int = 262144

    # ----- derived paths -------------------------------------------------------------------------------------
    @property
    def snapshot_rel(self) -> str:
        return f"hub/models--{self.repo.replace('/', '--')}/snapshots/{self.revision}"

    @property
    def snapshot_dir(self) -> Path:
        return self.hf_cache / self.snapshot_rel

    @property
    def snapshot_in_container(self) -> str:
        return f"{CONTAINER_HF}/{self.snapshot_rel}"

    @property
    def cache_dir(self) -> Path:
        """Host folder mounted at /cache/tf: the converted checkpoint and the profile folders."""
        return self.tf_cache / self.cache_folder

    @property
    def converted_dir(self) -> Path:
        return self.cache_dir / self.converted

    @property
    def converted_in_container(self) -> str:
        return f"{CONTAINER_TF}/{self.converted}"

    @property
    def long_context(self) -> int:
        return self.trained_context * self.yarn_factor

    @property
    def long_dir(self) -> Path:
        return self.cache_dir / f"long-{self.long_context}"

    @property
    def model_in_container(self) -> str:
        return f"{CONTAINER_TF}/long-{self.long_context}" if self.yarn else self.converted_in_container

    @property
    def patch_path(self) -> Path:
        return ROOT / "docker" / "patches" / self.patch

    @property
    def extra_env_pairs(self) -> list[str]:
        return [kv for kv in self.extra_env.split() if kv]

    def native_args(self, rank: int) -> list[str]:
        """``tensorfold-native serve <model>`` arguments: the flags the native server reads (it refuses the rest)."""
        args = ["--context", str(self.context), "--no-update-check", "--tp", "2", "--rank", str(rank),
                "--master", self.head_ip, "--master-port", str(self.master_port)]
        if self.kv_dtype != "int8":
            args += ["--kv-dtype", self.kv_dtype]   # both ranks; int8 is the native default
        if not self.drafts:
            args.append("--no-drafts")
        if rank == 0:
            args += ["--name", self.served_name, "--host", "0.0.0.0", "--port", str(self.port),
                     "--max-tokens", str(self.max_tokens), "--thinking" if self.thinking else "--no-thinking",
                     "--parallel", str(self.parallel)]
            if self.tool_system:
                args += ["--tool-system", self.tool_system]
            if self.vision and self.vision_urls:
                args.append("--vision-urls")
            if self.vision and self.vision_max_images:
                args += ["--vision-max-images", str(self.vision_max_images)]
        if self.vision:
            args.append("--vision")                 # both ranks: the tower on rank 0, rank 1 receives the features
        return args + self.extra_args.split()

    def container_env(self) -> dict[str, str]:
        env = {"HF_HOME": CONTAINER_HF, "HF_HUB_OFFLINE": "1", "TENSORFOLD_NO_UPDATE_CHECK": "1",
               "NCCL_SOCKET_IFNAME": self.iface, "NCCL_IB_HCA": self.hca, "NCCL_DEBUG": "WARN"}
        if self.memory_reserve_gib:
            env["TENSORFOLD_MEMORY_RESERVE_GIB"] = str(self.memory_reserve_gib)
        for kv in self.extra_env_pairs:
            k, v = kv.split("=", 1)
            env[k] = v
        return env

    def summary(self) -> str:
        return (f"profile={self.profile} tp=2 ctx={self.context} yarn={int(self.yarn)} kv={self.kv_dtype} "
                f"drafts={int(self.drafts)} parallel={self.parallel} vision={int(self.vision)} image={self.image} "
                f"patch={self.patch} model={self.repo}@{self.revision[:8]} port={self.port}")


# ----- loading ---------------------------------------------------------------------------------------------------

_SECTIONS = ("model", "engine", "cluster", "serve")
_INT = re.compile(r"^(0|[1-9][0-9]*)$")              # bash would read 010 as octal; refuse zero padding everywhere


def _coerce(name: str, kind: type, raw) -> object:
    text = raw if isinstance(raw, str) else None
    if kind is bool:
        if isinstance(raw, bool):
            return raw
        if text in ("1", "true", "True"):
            return True
        if text in ("0", "false", "False"):
            return False
        raise ConfigError(f"{name}={raw!r} must be 0/1 or true/false")
    if kind is int:
        if isinstance(raw, bool) or (not isinstance(raw, int) and not (text and _INT.match(text))):
            raise ConfigError(f"{name}={raw!r} is not a decimal integer")
        return int(raw)
    if kind is float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise ConfigError(f"{name}={raw!r} is not a number") from None
    if kind is Path:
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            raise ConfigError(f"{name}={raw!r} must be an absolute path (or start with ~)")
        return p
    return str(raw)


def load(cluster_file: Path | str | None = None, env: dict[str, str] | None = None) -> Settings:
    """Settings from recipe.toml, then ``cluster.toml`` (or ``cluster_file``; "none" skips it), then the environment."""
    env = os.environ if env is None else env
    raw = tomllib.loads((ROOT / "recipe.toml").read_text())
    values: dict[str, object] = {}
    for section in _SECTIONS:
        values.update(raw.get(section, {}))
    profiles = raw.get("profiles", {})
    yarn = raw.get("yarn", {})
    values["yarn_factor"] = yarn.get("factor", Settings.__dataclass_fields__["yarn_factor"].default)
    values["trained_context"] = yarn.get("trained_context", Settings.__dataclass_fields__["trained_context"].default)
    explicit: set[str] = set()
    cluster = cluster_file if cluster_file is not None else env.get("SPARK_CLUSTER", ROOT / "cluster.toml")
    if str(cluster) != "none" and Path(cluster).is_file():
        extra = tomllib.loads(Path(cluster).read_text())
        for section, body in extra.items():
            if not isinstance(body, dict):
                raise ConfigError(f"{cluster}: top-level key {section} must be a table")
            for k, v in body.items():
                if k in REMOVED:
                    raise ConfigError(f"{cluster}: {k} is no longer a setting: {REMOVED[k]}")
                values[k] = v
                explicit.add(k)
    for k, why in REMOVED.items():
        if k.upper() in env:
            raise ConfigError(f"{k.upper()} is no longer a setting: {why}")
    kinds = {f.name: f.type for f in fields(Settings)}
    for name in kinds:
        if name.upper() in env:
            values[name] = env[name.upper()]
            explicit.add(name)
    unknown = sorted(k for k in values if k not in kinds)
    if unknown:
        raise ConfigError(f"unknown setting(s): {', '.join(unknown)}")
    profile = str(values.get("profile", "concurrent"))
    if profile not in PROFILES:
        raise ConfigError(f"profile={profile} must be one of {', '.join(PROFILES)}")
    for k, v in profiles.get(profile, {}).items():          # a profile fills what nothing more specific set
        if k in kinds and k not in explicit:
            values[k] = v
    typed = {}
    for name, kind in kinds.items():
        if name not in values:
            raise ConfigError(f"recipe.toml lacks {name}")
        real = {"str": str, "int": int, "bool": bool, "float": float, "Path": Path}.get(kind if isinstance(kind, str) else kind.__name__, kind)
        typed[name] = _coerce(name, real, values[name])
    s = Settings(**typed)
    validate(s)
    return s


def validate(s: Settings) -> None:
    for name, lo, hi in (("port", 1, 65535), ("master_port", 1, 65535), ("parallel", 1, 16),
                         ("max_tokens", 1, 1 << 24), ("vision_max_images", 0, 64),
                         ("settle_seconds", 0, 600), ("memguard_min_avail_mb", 0, 1 << 20),
                         ("memguard_min_swap_free_mb", 0, 1 << 20), ("yarn_factor", 1, 16)):
        v = getattr(s, name)
        if not lo <= v <= hi:
            raise ConfigError(f"{name}={v} is outside [{lo}, {hi}]")
    if s.kv_dtype not in KV_DTYPES:
        raise ConfigError(f"kv_dtype={s.kv_dtype} must be one of {', '.join(KV_DTYPES)} (the engine's caches)")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", s.cache_folder):
        raise ConfigError(f"cache_folder={s.cache_folder!r} must be one folder name under tf_cache")
    for name in ("revision", "tensorfold_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", getattr(s, name)):
            raise ConfigError(f"{name}={getattr(s, name)} is not a 40-hex commit")
    if s.converted_repo and not re.fullmatch(r"[0-9a-f]{40}", s.converted_revision):
        raise ConfigError(f"converted_revision={s.converted_revision!r} must be the 40-hex revision of {s.converted_repo}")
    if s.yarn:
        if not s.trained_context < s.context <= s.long_context:
            raise ConfigError(f"context={s.context} with yarn must be above the trained {s.trained_context} "
                              f"and at most {s.long_context} (YaRN x{s.yarn_factor}); yarn = false serves up to "
                              f"{s.trained_context}")
    elif s.context > s.trained_context:
        raise ConfigError(f"context={s.context} exceeds the trained window {s.trained_context}; yarn = true serves up "
                          f"to {s.long_context}")
    if s.context % 256:
        raise ConfigError(f"context={s.context} must be a multiple of 256")
    if not re.fullmatch(r"[A-Za-z0-9._-]+\.patch", s.patch):
        raise ConfigError(f"patch={s.patch} must be a file name under docker/patches/")
    if not s.patch_path.is_file():
        raise ConfigError(f"docker/patches/{s.patch} is missing")
    if not re.fullmatch(r"[0-9a-f]{64}", s.patch_sha256):
        raise ConfigError(f"patch_sha256={s.patch_sha256} is not a sha256 (recipe.toml [engine])")
    have = hashlib.sha256(s.patch_path.read_bytes()).hexdigest()
    if have != s.patch_sha256:
        raise ConfigError(f"docker/patches/{s.patch} has sha256 {have}, not the pinned {s.patch_sha256}: never edit a "
                          "patch in place, regenerate it and re-pin (docker/patches/README.md)")
    for word in s.extra_args.split():
        flag = word.split("=", 1)[0]
        if not flag.startswith("--"):
            continue
        for owned in OWNED_FLAGS:
            if owned.startswith(flag):
                raise ConfigError(f"extra_args sets {word} (the engine reads it as {owned}), which the recipe passes "
                                  "itself; use the matching setting instead")
    for kv in s.extra_env_pairs:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", kv):
            raise ConfigError(f"extra_env entry {kv!r} is not KEY=VALUE")
    if "@" not in s.worker and "." not in s.worker:
        raise ConfigError(f"worker={s.worker!r} should be an ssh destination such as user@10.100.8.2")
