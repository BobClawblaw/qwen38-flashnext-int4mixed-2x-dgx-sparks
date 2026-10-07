"""The checkpoint on disk: the pinned snapshot, its one-time conversion, the worker's copy, the long profile folder."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

from .config import CONTAINER_TF, ROOT, Settings
from .nodes import Node, log

REQUIRED = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
            "model.safetensors.index.json")
CONVERTER = ROOT / "spark" / "convert_mixed.py"


def snapshot_problems(folder: Path) -> list[str]:
    """Why a checkpoint folder is incomplete: missing files, or a shard whose header does not end at its size."""
    bad = [name for name in REQUIRED if not (folder / name).is_file() or (folder / name).stat().st_size == 0]
    index = folder / "model.safetensors.index.json"
    if not index.is_file():
        return bad
    try:
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    except (OSError, ValueError, KeyError) as exc:
        return bad + [f"{index.name} ({exc})"]
    for shard in shards:
        p = folder / shard
        try:
            with p.open("rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
            if 8 + n + end != p.stat().st_size:
                bad.append(f"{shard} (truncated)")
        except (OSError, ValueError, struct.error) as exc:
            bad.append(f"{shard} ({exc.__class__.__name__})")
    return bad


def ensure_snapshot(s: Settings, *, download: bool = True) -> None:
    """The pinned revision, complete, under hf_cache; downloaded with the hf CLI when it is not."""
    problems = snapshot_problems(s.snapshot_dir)
    if not problems:
        log(f"Snapshot {s.repo}@{s.revision[:8]} complete ({s.snapshot_dir})")
        return
    if not download:
        raise RuntimeError(f"snapshot incomplete: {', '.join(problems[:6])}")
    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf is None:
        raise RuntimeError(f"no hf CLI on PATH and the snapshot is incomplete ({', '.join(problems[:4])}); "
                           "pip install -U huggingface_hub")
    log(f"Downloading {s.repo}@{s.revision[:8]} (about 165 GB; the body first, then the bf16 n-gram shards)")
    rc = subprocess.call([hf, "download", s.repo, "--revision", s.revision],
                         env={**os.environ, "HF_HOME": str(s.hf_cache), "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1")})
    if rc:
        raise RuntimeError(f"hf download failed ({rc})")
    problems = snapshot_problems(s.snapshot_dir)
    if problems:
        raise RuntimeError(f"snapshot still incomplete after the download: {', '.join(problems[:6])}")


def converter_sha() -> str:
    return hashlib.sha256(CONVERTER.read_bytes()).hexdigest()


def conversion_marker(s: Settings) -> str:
    """What the served folder must declare in .source: the local conversion (snapshot + converter) or the Hub copy."""
    if s.converted_repo:
        return f"hub {s.converted_repo}@{s.converted_revision}"
    return f"{s.revision} {converter_sha()}"


def converted_ready(s: Settings, node: Node) -> bool:
    out = node.run(["cat", str(s.converted_dir / ".source")], check=False)
    return out.returncode == 0 and out.stdout.strip() == conversion_marker(s)


def download_converted(s: Settings, head: Node) -> None:
    """The converted checkpoint from the Hub at its pinned revision, straight into tf_cache/<sha>/<converted>."""
    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf is None:
        raise RuntimeError("no hf CLI on PATH; pip install -U huggingface_hub, or set converted_repo = \"\" to convert locally")
    tmp = s.converted_dir.with_name(s.converted + ".partial")
    head.run(["rm", "-rf", str(tmp), str(s.converted_dir)])
    head.run(["mkdir", "-p", str(tmp)])
    log(f"Downloading the converted checkpoint {s.converted_repo}@{s.converted_revision[:8]} (about 130 GB)")
    rc = subprocess.call([hf, "download", s.converted_repo, "--revision", s.converted_revision, "--local-dir", str(tmp)],
                         env={**os.environ, "HF_HOME": str(s.hf_cache), "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1")})
    if rc:
        raise RuntimeError(f"hf download of {s.converted_repo} failed ({rc})")
    problems = snapshot_problems(tmp)
    if problems:
        raise RuntimeError(f"the downloaded checkpoint is incomplete: {', '.join(problems[:6])}")
    (tmp / ".source").write_text(conversion_marker(s) + "\n")
    tmp.rename(s.converted_dir)


def ensure_converted(s: Settings, head: Node) -> None:
    """The served folder on the head: downloaded ready-made when converted_repo is set, else converted here (GPU)."""
    if converted_ready(s, head):
        log(f"Converted checkpoint {s.converted_dir}")
        return
    if s.converted_repo:
        download_converted(s, head)
        log(f"Converted checkpoint {s.converted_dir} (from the Hub)")
        return
    ensure_snapshot(s)
    log(f"Converting {s.repo}@{s.revision[:8]} to the engine's layout (once; about 25 minutes, GPU) -> {s.converted_dir}")
    tmp = s.converted_dir.with_name(s.converted + ".partial")
    head.run(["rm", "-rf", str(tmp), str(s.converted_dir)])
    head.run(["mkdir", "-p", str(s.cache_dir)])
    rc = head.stream(["docker", "run", "--rm", "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}",
                      "-v", f"{s.hf_cache}:/cache/huggingface:ro", "-v", f"{s.cache_dir}:{CONTAINER_TF}",
                      "-v", f"{CONVERTER.parent}:/work:ro", "-e", "HF_HUB_OFFLINE=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
                      "--entrypoint", "python3", s.image, "-I", "/work/convert_mixed.py", s.snapshot_in_container,
                      f"{CONTAINER_TF}/{tmp.name}"])
    if rc:
        raise RuntimeError(f"the conversion failed ({rc}); see the output above")
    (tmp / ".source").write_text(conversion_marker(s) + "\n")
    tmp.rename(s.converted_dir)
    log(f"Converted: {s.converted_dir}")


def sync_converted(s: Settings, head: Node, worker: Node) -> None:
    """The worker serves the head's converted folder, byte for byte (rsync over the link; about 130 GB once)."""
    if converted_ready(s, worker):
        return
    log(f"Copying the converted checkpoint to {worker.name} (about 130 GB over the link)")
    worker.run(["mkdir", "-p", str(s.cache_dir)])
    rc = subprocess.call(["rsync", "-a", "--delete", "--inplace", "-e", "ssh -o BatchMode=yes",
                          f"{s.converted_dir}/", f"{worker.ssh}:{s.converted_dir}/"])
    if rc or not converted_ready(s, worker):
        raise RuntimeError(f"rsync of the converted checkpoint to {worker.name} failed ({rc})")


def ensure_long_profile(s: Settings, node: Node) -> None:
    """profile=long: a folder of links to the converted checkpoint (container paths) with a YaRN config.json."""
    marker = node.run(["cat", str(s.long_dir / ".source")], check=False).stdout.strip()
    if marker == conversion_marker(s):
        return
    script = f"""set -euo pipefail
dir={s.long_dir}
src={s.converted_dir}
rm -rf "$dir"; mkdir -p "$dir"
for f in "$src"/*; do b=$(basename "$f"); [[ "$b" == config.json ]] && continue; ln -s "{s.converted_in_container}/$b" "$dir/$b"; done
python3 - "$src/config.json" "$dir/config.json" {s.yarn_factor} {s.trained_context} {s.long_context} <<'PY'
import json, sys
src, dst, factor, native, window = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
c = json.load(open(src))
t = c.get("text_config", c)
if int(t.get("max_position_embeddings", 0)) != native:
    sys.exit(f"{{src}}: max_position_embeddings is {{t.get('max_position_embeddings')}}, not {{native}}")
rope = dict(t.get("rope_parameters") or {{}})
rope.pop("type", None)
rope.update({{"rope_type": "yarn", "factor": factor, "original_max_position_embeddings": native}})
t["rope_parameters"] = rope
t["max_position_embeddings"] = window
if "text_config" in c:
    c["max_position_embeddings"] = window
json.dump(c, open(dst, "w"), indent=1)
PY
printf '%s\\n' "{conversion_marker(s)}" > "$dir/.source"
"""
    node.script(script)
    log(f"Long profile folder {s.long_dir} on {node.name} (YaRN x{s.yarn_factor}, {s.long_context} tokens)")
