#!/usr/bin/env python3
"""Convert Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound (compressed-tensors, pack-quantized) into the layout
TensorFold's Flash Next CUDA loader reads as ``affine-experts``:

- routed experts: the stored int4 (symmetric, groups of 128) as MLX affine 4-bit words in groups of 64. The packed
  words are bit-identical (both pack eight nibbles little-endian per int32); each 128-group scale becomes two
  64-group scales (fp16 -> bf16) with bias -8 x scale, so a weight is scale x (u - 8) as the source defines it.
- every other quantized linear (int6 groups of 64: DeltaNet, attention, shared expert; int8: hyper-connection
  mixers, lm_head, indexer and PLE projections): the stored integers as int8 bytes (`<name>.qweight`, int6 widened
  losslessly) with the group scales as fp32 (`<name>.qscale`, [n, k / group]); exact, half the bytes of bf16.
  The embedding table is dequantized to bf16 (a gather reads one row a token).
- the n-gram table (bf16 rows): e4m3 with one table scale (the form NVIDIA's NVFP4 export uses), half the bytes,
  so it fits beside the weights on a 128 GB node.
- MTP experts (bf16): MLX affine 4-bit in groups of 64 (min/max per group); they only draft, verification is exact.
- everything else (norms, gates, conv taps, the vision tower, A_log, dt_bias): copied as stored.

usage: convert_mixed.py SRC DST [--device cuda|cpu] [--check N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GS_OUT = 64          # MLX affine group the loader's expert kernel takes
FORMAT = 2           # 1: dense linears bf16; 2: dense linears int8 + group scales
PASS_FILES = ("chat_template.jinja", "generation_config.json", "merges.txt", "vocab.json", "tokenizer.json",
              "tokenizer_config.json", "preprocessor_config.json", "processor_config.json",
              "video_preprocessor_config.json", "special_tokens_map.json", "added_tokens.json")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# compressed-tensors pack-quantized: element i of a row sits at bit i * bits of a dense little-endian bitstream,
# packed 32 elements -> ``bits`` int32 words; values are stored unsigned (q + 2^(bits-1)).

def unpack(packed: torch.Tensor, bits: int, k: int) -> torch.Tensor:
    """packed [N, ceil(K*bits/32)] int32 -> signed int values [N, K] (int16)."""
    n, pc = packed.shape
    groups = -(-k // 32)
    words = packed.to(torch.int64) & 0xFFFFFFFF
    if pc < groups * bits:
        words = torch.nn.functional.pad(words, (0, groups * bits - pc))
    words = words.reshape(n, groups, bits)
    mask = (1 << bits) - 1
    i = torch.arange(32, device=packed.device, dtype=torch.int64)
    start = i * bits
    w0 = start // 32
    off = start % 32
    lo = (torch.gather(words, 2, w0.expand(n, groups, 32)) >> off) & mask
    spill = (off + bits - 32).clamp(min=0)                                    # bits in the next word
    w1 = (w0 + 1).clamp(max=bits - 1)
    hi = torch.gather(words, 2, w1.expand(n, groups, 32)) << (bits - spill)
    hi = torch.where(spill > 0, hi & mask, torch.zeros_like(hi))
    vals = (lo | hi).reshape(n, groups * 32)[:, :k]
    return (vals - (1 << (bits - 1))).to(torch.int16)


def dequant_bf16(packed: torch.Tensor, scale: torch.Tensor, bits: int, shape: tuple[int, int]) -> torch.Tensor:
    n, k = shape
    q = unpack(packed, bits, k).float()
    g = k // scale.shape[1]
    w = q.reshape(n, -1, g) * scale.float()[:, :, None]
    return w.reshape(n, k).to(torch.bfloat16)


def dense_int8(packed: torch.Tensor, scale: torch.Tensor, bits: int, shape: tuple[int, int]):
    """int6 / int8 symmetric -> (int8 [n, k], fp32 scales [n, k / group]): the same integers, the same scales."""
    n, k = shape
    q = unpack(packed, bits, k)
    if bits > 8:
        raise ValueError(f"{bits}-bit values do not fit int8")
    return q.to(torch.int8).contiguous(), scale.float().contiguous()


def experts_affine(packed: torch.Tensor, scale: torch.Tensor, shape: tuple[int, int]):
    """int4 g128 -> (MLX words [N, K/8] int32, scales [N, K/64] bf16, biases [N, K/64] bf16)."""
    n, k = shape
    if packed.shape[1] * 8 != k or k % (2 * GS_OUT):
        raise ValueError(f"experts: {tuple(packed.shape)} does not hold {k} nibbles in 64-groups")
    gsrc = k // scale.shape[1]
    if gsrc % GS_OUT:
        raise ValueError(f"experts: source group {gsrc} is not a multiple of {GS_OUT}")
    s = scale.to(torch.bfloat16).repeat_interleave(gsrc // GS_OUT, dim=1).contiguous()
    b = (-8.0 * s.float()).to(torch.bfloat16).contiguous()
    return packed.contiguous(), s, b


def quantize_affine(w: torch.Tensor):
    """bf16 [N, K] -> MLX affine 4-bit, groups of 64 (min/max), for weights that only draft."""
    n, k = w.shape
    g = w.float().reshape(n, k // GS_OUT, GS_OUT)
    mn = g.min(dim=-1, keepdim=True).values
    mx = g.max(dim=-1, keepdim=True).values
    s = ((mx - mn) / 15.0).clamp(min=1e-8).to(torch.bfloat16).float()
    b = mn.to(torch.bfloat16).float()
    q = ((g - b) / s).round().clamp(0, 15).to(torch.int64).reshape(n, k)
    shifts = (torch.arange(8, device=w.device, dtype=torch.int64) * 4)
    words = (q.reshape(n, k // 8, 8) << shifts).sum(-1)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return words.contiguous(), s.reshape(n, k // GS_OUT).to(torch.bfloat16).contiguous(), \
        b.reshape(n, k // GS_OUT).to(torch.bfloat16).contiguous()


# ---------------------------------------------------------------------------------------------------------------

class Source:
    def __init__(self, src: Path, device: str) -> None:
        self.src, self.device = src, device
        self.index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
        self._open: dict[str, object] = {}

    def file(self, shard: str):
        if shard not in self._open:
            if len(self._open) > 4:
                self._open.pop(next(iter(self._open)))
            self._open[shard] = safe_open(str(self.src / shard), framework="pt", device="cpu")
        return self._open[shard]

    def get(self, name: str, device: str | None = None) -> torch.Tensor:
        t = self.file(self.index[name]).get_tensor(name)
        return t.to(self.device if device is None else device)

    def has(self, name: str) -> bool:
        return name in self.index


class Sink:
    """Collects tensors into named output shards and writes the index."""

    def __init__(self, dst: Path) -> None:
        self.dst = dst
        self.weight_map: dict[str, str] = {}
        self.total = 0
        dst.mkdir(parents=True, exist_ok=True)

    def write(self, shard: str, tensors: dict[str, torch.Tensor]) -> None:
        out = {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}
        for k, v in out.items():
            if k in self.weight_map:
                raise ValueError(f"duplicate output tensor {k}")
            self.weight_map[k] = shard
            self.total += v.numel() * v.element_size()
        save_file(out, str(self.dst / shard), metadata={"format": "pt"})
        log(f"wrote {shard}: {len(out)} tensors, {sum(v.numel() * v.element_size() for v in out.values()) / 2**30:.2f} GiB")

    def finish(self) -> None:
        (self.dst / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": self.total}, "weight_map": dict(sorted(self.weight_map.items()))},
                       indent=1))


def classify(src: Source):
    """Group source tensor names: per decoder layer, mtp, visual, ngram shards, head/embed/mixer (top)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for name in src.index:
        m = re.match(r"(.*\.layers\.)(\d+)\.", name)
        if ".ngram_embedding.shard_" in name:
            groups["ngram"].append(name)
        elif name.startswith("mtp."):
            groups["mtp"].append(name)
        elif ".visual." in name:
            groups["visual"].append(name)
        elif m and not name.startswith("mtp."):
            groups[f"layer-{int(m.group(2)):02d}"].append(name)
        else:
            groups["top"].append(name)
    return groups


def is_expert_part(base: str) -> bool:
    return re.search(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$", base) is not None


def convert_group(src: Source, names: list[str], *, device: str) -> dict[str, torch.Tensor]:
    """Every tensor of one output shard: packed linears unpacked, experts stacked, the rest copied."""
    out: dict[str, torch.Tensor] = {}
    experts: dict[str, dict[int, tuple]] = defaultdict(dict)          # "<layer>.mlp.switch_mlp.gate_proj" -> {e: triple}
    mtp_bf16_experts: dict[str, torch.Tensor] = {}
    seen = set(names)
    for name in names:
        if name.endswith(".weight_scale") or name.endswith(".weight_shape"):
            continue
        if name.endswith(".weight_packed"):
            base = name[:-len(".weight_packed")]
            packed = src.get(name)
            scale = src.get(base + ".weight_scale")
            shape = tuple(int(x) for x in src.get(base + ".weight_shape", "cpu").tolist())
            n, k = shape
            bits = packed.shape[1] * 32 // k
            if packed.shape[1] != math.ceil(k * bits / 32) or bits not in (4, 6, 8):
                raise ValueError(f"{name}: cannot infer bits from {tuple(packed.shape)} and shape {shape}")
            if is_expert_part(base):
                if bits != 4:
                    raise ValueError(f"{base}: routed experts are expected int4, got {bits}")
                m = re.match(r"(.*\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$", base)
                key = f"{m.group(1)}.switch_mlp.{m.group(3)}"
                experts[key][int(m.group(2))] = experts_affine(packed, scale, shape)
            elif base.endswith("embed_tokens"):
                out[base + ".weight"] = dequant_bf16(packed, scale, bits, shape)
            else:
                q, sc = dense_int8(packed, scale, bits, shape)
                out[base + ".qweight"], out[base + ".qscale"] = q, sc
            continue
        t = src.get(name)
        if name.startswith("mtp.") and ".mlp.experts." in name and t.dtype in (torch.bfloat16, torch.float16):
            mtp_bf16_experts[name] = t
            continue
        out[name] = t
    for key, parts in experts.items():
        count = len(parts)
        if sorted(parts) != list(range(count)):
            raise ValueError(f"{key}: experts {sorted(parts)[:5]}... are not 0..{count - 1}")
        for field, idx in (("weight", 0), ("scales", 1), ("biases", 2)):
            out[f"{key}.{field}"] = torch.stack([parts[e][idx] for e in range(count)]).contiguous()
    if mtp_bf16_experts:
        out.update(convert_mtp_experts(mtp_bf16_experts))
    return out


def convert_mtp_experts(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The MTP layer's bf16 experts -> stacked MLX affine 4-bit (groups of 64): drafts only."""
    out: dict[str, torch.Tensor] = {}
    per: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    for name, t in tensors.items():
        m = re.match(r"(mtp\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", name)
        if m:
            per[f"{m.group(1)}.switch_mlp.{m.group(3)}"][int(m.group(2))] = t
            continue
        m = re.match(r"(mtp\.layers\.\d+\.mlp)\.experts\.(gate_up_proj|down_proj)$", name)
        if m:                                                   # fused [E, 2NI, D] / [E, D, NI]
            base = m.group(1)
            if m.group(2) == "gate_up_proj":
                ni = t.shape[1] // 2
                per[f"{base}.switch_mlp.gate_proj"] = {e: t[e, :ni] for e in range(t.shape[0])}
                per[f"{base}.switch_mlp.up_proj"] = {e: t[e, ni:] for e in range(t.shape[0])}
            else:
                per[f"{base}.switch_mlp.down_proj"] = {e: t[e] for e in range(t.shape[0])}
            continue
        raise ValueError(f"unexpected MTP expert tensor {name} {tuple(t.shape)}")
    for key, parts in per.items():
        count = len(parts)
        trip = [quantize_affine(parts[e].to(torch.bfloat16)) for e in range(count)]
        for field, idx in (("weight", 0), ("scales", 1), ("biases", 2)):
            out[f"{key}.{field}"] = torch.stack([tr[idx] for tr in trip]).contiguous()
    return out


def convert_ngram(src: Source, sink: Sink, names: list[str], *, device: str, rows_chunk: int = 2_000_000) -> None:
    """bf16 n-gram shards -> e4m3 with one table scale (absmax / 448), written shard file by shard file."""
    by_file: dict[str, list[str]] = defaultdict(list)
    for n in names:
        by_file[src.index[n]].append(n)
    absmax = torch.zeros((), dtype=torch.float32, device=device)
    t0 = time.time()
    for shard, keys in sorted(by_file.items()):
        f = src.file(shard)
        for k in keys:
            t = f.get_slice(k)
            rows = t.get_shape()[0]
            for r in range(0, rows, rows_chunk):
                absmax = torch.maximum(absmax, t[r:r + rows_chunk].to(device).float().abs().amax())
    scale = float(absmax) / 448.0
    log(f"n-gram absmax {float(absmax):.4f} -> table scale {scale:.6g} ({time.time() - t0:.0f}s)")
    base = re.sub(r"\.shard_\d+\.weight$", "", names[0])
    first = True
    for shard, keys in sorted(by_file.items()):
        f = src.file(shard)
        out: dict[str, torch.Tensor] = {}
        for k in keys:
            t = f.get_slice(k)
            rows, width = t.get_shape()
            dst = torch.empty((rows, width), dtype=torch.float8_e4m3fn)
            for r in range(0, rows, rows_chunk):
                dst[r:r + rows_chunk] = (t[r:r + rows_chunk].to(device).float() / scale).to(torch.float8_e4m3fn).cpu()
            out[k] = dst
        if first:
            out[base + ".weight_scale"] = torch.tensor(scale, dtype=torch.float32)
            first = False
        sink.write(shard, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default="", help="comma list of groups to convert (debug), e.g. layer-00,top")
    ap.add_argument("--skip-ngram", action="store_true")
    a = ap.parse_args()
    src = Source(a.src, a.device)
    sink = Sink(a.dst)
    groups = classify(src)
    log(f"{len(src.index)} source tensors in {len(groups)} groups on {a.device}")
    only = set(a.only.split(",")) if a.only else None
    order = sorted(g for g in groups if g not in ("ngram",))
    for g in order:
        if only and g not in only:
            continue
        shard = f"model-{g}.safetensors"
        t0 = time.time()
        out = convert_group(src, groups[g], device=a.device)
        sink.write(shard, out)
        del out
        if a.device == "cuda":
            torch.cuda.empty_cache()
        log(f"{g}: {time.time() - t0:.1f}s")
    if not a.skip_ngram and (not only or "ngram" in only):
        convert_ngram(src, sink, sorted(groups["ngram"]), device=a.device)
    sink.finish()
    cfg = json.loads((a.src / "config.json").read_text())
    source_q = cfg.pop("quantization_config", None)
    for sub in ("text_config",):
        if isinstance(cfg.get(sub), dict):
            cfg[sub].pop("quantization_config", None)
    cfg["quantization_config"] = {
        "quant_method": "affine-experts", "bits": 4, "group_size": GS_OUT, "format": FORMAT,
        "dense": "int8-group-scales", "ngram": "fp8-e4m3-table-scale", "mtp_experts": "affine-4bit-g64-minmax",
        "source": {"repo": "Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound",
                   "format": (source_q or {}).get("format"), "config_groups": (source_q or {}).get("config_groups")},
    }
    (a.dst / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in PASS_FILES:
        if (a.src / f).exists():
            shutil.copy2(a.src / f, a.dst / f)
    log(f"done: {sink.total / 2**30:.1f} GiB in {a.dst}")


if __name__ == "__main__":
    main()
