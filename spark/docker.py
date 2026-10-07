"""The engine image and its containers: build, label check, copy to the worker, start a rank, stop it."""

from __future__ import annotations

from pathlib import Path

from .config import CONTAINER_HF, CONTAINER_TF, ROOT, Settings
from .nodes import Node, log, pipe


def labels(node: Node, image: str) -> tuple[str, str] | None:
    """(tensorfold sha, patch sha256) from the image's labels; None when the node has no such image."""
    out = node.run(["docker", "image", "inspect", "-f",
                    '{{ index .Config.Labels "tensorfold.sha" }} {{ index .Config.Labels "tensorfold.patch_sha" }}',
                    image], check=False)
    if out.returncode:
        return None
    sha, _, patch = out.stdout.strip().partition(" ")
    return sha, patch


def image_id(node: Node, image: str) -> str | None:
    out = node.run(["docker", "image", "inspect", "-f", "{{.Id}}", image], check=False)
    return out.stdout.strip() if out.returncode == 0 else None


def ensure_image(s: Settings, head: Node) -> None:
    """Build the image on the head when it is missing; refuse one whose labels name another engine or patch."""
    have = labels(head, s.image)
    if have is None:
        log(f"Building {s.image} (TensorFold {s.tensorfold_sha[:8]}, patch {s.patch}) from docker/Dockerfile")
        rc = head.stream(["docker", "build", "--build-arg", f"TF_REPO={s.tensorfold_repo}",
                          "--build-arg", f"TF_SHA={s.tensorfold_sha}", "--build-arg", f"TF_PATCH={s.patch}",
                          "--build-arg", f"TF_PATCH_SHA={s.patch_sha256}", "--build-arg", f"BASE={s.base_image}",
                          "-t", s.image, str(ROOT / "docker")])
        if rc:
            raise RuntimeError(f"docker build of {s.image} failed ({rc})")
        have = labels(head, s.image)
    if have != (s.tensorfold_sha, s.patch_sha256):
        raise RuntimeError(f"image {s.image} carries TensorFold/patch {have}, not ({s.tensorfold_sha}, "
                           f"{s.patch_sha256}): rebuild it or point `image` at the matching tag")
    log(f"Image {s.image} (TensorFold {s.tensorfold_sha[:8]}, patch {s.patch} sha256 {s.patch_sha256[:8]})")


def sync_image(s: Settings, head: Node, worker: Node) -> bool:
    """The worker runs the head's exact image: same ID, or the head's copy goes over the link. True when it copied."""
    local, remote = image_id(head, s.image), image_id(worker, s.image)
    if local == remote:
        return False
    log(f"Copying {s.image} to {worker.name} (worker has {remote or 'none'})")
    pipe(["docker", "save", s.image], worker, ["docker", "load"])
    if image_id(worker, s.image) != local:
        raise RuntimeError(f"{worker.name}: the image copy did not take")
    return True


def state(node: Node, name: str) -> str:
    """running, exited or missing."""
    out = node.run(["docker", "inspect", "-f", "{{.State.Running}}", name], check=False)
    if out.returncode:
        return "missing"
    return "running" if out.stdout.strip() == "true" else "exited"


def stop(node: Node, name: str, timeout: int = 30) -> bool:
    """Stop and remove the container; True when there was one."""
    if state(node, name) == "missing":
        return False
    log(f"Stopping {name} on {node.name}")
    node.run(["docker", "stop", "-t", str(timeout), name], check=False, timeout=timeout + 60)
    node.run(["docker", "rm", "-f", name], check=False)
    return True


def run_rank(s: Settings, node: Node, rank: int, *, src_mount: Path | None = None) -> None:
    """Start one rank's container (detached). ``src_mount``: a tensorfold source tree mounted over the image's."""
    hf, tf = s.hf_cache, s.cache_dir
    node.run(["mkdir", "-p", str(hf), str(tf)])
    argv = ["docker", "run", "-d", "--name", s.container, "--init", "--restart", "no", "--oom-score-adj", "1000",
            "--ulimit", "core=1", "--gpus", "all", "--network", "host", "--ipc", "host", "--device", "/dev/infiniband",
            "--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1:-1",
            "-v", f"{hf}:{CONTAINER_HF}:ro", "-v", f"{tf}:{CONTAINER_TF}"]
    if src_mount is not None:
        argv += ["-v", f"{src_mount}:/usr/local/lib/python3.12/dist-packages/tensorfold:ro"]
    for k, v in s.container_env().items():
        argv += ["-e", f"{k}={v}"]
    argv += [s.image, "tensorfold", "serve", s.model_in_container, *s.serve_args(rank)]
    # --init: rank 1 installs no SIGTERM handler (docker stop would wait out its timeout). memlock + IPC_LOCK: the
    # n-gram table pages are locked. --ulimit core=1: no multi-GiB core dumps in host memory.
    log(f"Starting {s.container} rank={rank} on {node.name}: {s.summary()}")
    node.run(argv)


def logs(node: Node, name: str, tail: int = 80) -> str:
    return node.run(["docker", "logs", "--tail", str(tail), name], check=False).stdout + \
        node.run(["docker", "logs", "--tail", str(tail), name], check=False).stderr


def foreign_gpu_containers(node: Node, own: str) -> list[str]:
    """Other containers holding the GPU or the InfiniBand devices (the pair needs both exclusively)."""
    names = [n for n in node.run(["docker", "ps", "--format", "{{.Names}}"], check=False).stdout.split() if n != own]
    busy = []
    for n in names:
        if n.startswith("buildx_buildkit_"):
            continue
        out = node.run(["docker", "inspect", "-f", "{{json .HostConfig.DeviceRequests}} {{json .HostConfig.Devices}}", n],
                       check=False).stdout.lower()
        if "gpu" in out or "nvidia" in out or "infiniband" in out:
            busy.append(n)
    return busy
