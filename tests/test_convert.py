"""The converter's arithmetic, offline (needs torch; skipped where it is missing, run inside the image otherwise):
pack-quantized unpacking against a bitstream reference, the expert scale/bias fold, the MTP affine quantizer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spark"))

try:
    import torch
except ModuleNotFoundError:          # the host has no torch; CI and the image do
    torch = None


def pack_reference(values: "torch.Tensor", bits: int) -> "torch.Tensor":
    """compressed-tensors' layout: per row, element i at bit i*bits of a dense little-endian stream, 32 values -> ``bits`` words."""
    n, k = values.shape
    offset = 1 << (bits - 1)
    out = []
    for r in range(n):
        acc, words = 0, []
        nbits = 0
        for v in values[r].tolist():
            acc |= (v + offset) << nbits
            nbits += bits
            while nbits >= 32:
                words.append(acc & 0xFFFFFFFF)
                acc >>= 32
                nbits -= 32
        if nbits:
            words.append(acc & 0xFFFFFFFF)
        out.append(words)
    t = torch.tensor(out, dtype=torch.int64)
    return torch.where(t >= 2 ** 31, t - 2 ** 32, t).to(torch.int32)


@unittest.skipIf(torch is None, "torch is not installed here")
class Unpacking(unittest.TestCase):
    def test_4_6_8_bits_round_trip(self) -> None:
        from convert_mixed import unpack

        g = torch.Generator().manual_seed(1)
        for bits in (4, 6, 8):
            lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
            vals = torch.randint(lo, hi + 1, (5, 2560), generator=g)
            packed = pack_reference(vals, bits)
            self.assertEqual(packed.shape[1], (2560 * bits + 31) // 32)
            got = unpack(packed, bits, 2560)
            self.assertTrue(torch.equal(got.to(torch.int64), vals), f"{bits} bits")

    def test_four_bit_words_are_mlx_words(self) -> None:
        from convert_mixed import experts_affine, unpack

        g = torch.Generator().manual_seed(2)
        vals = torch.randint(-8, 8, (64, 256), generator=g)
        packed = pack_reference(vals, 4)
        scale = (torch.rand(64, 2, generator=g) * 0.05 + 0.01).to(torch.float16)        # groups of 128
        words, s, b = experts_affine(packed, scale, (64, 256))
        self.assertTrue(torch.equal(words, packed))
        self.assertEqual(s.shape, (64, 4))
        # MLX decode: nibble i of word j is input 8j+i; value s*u + b with u = q + 8
        w = words.to(torch.int64) & 0xFFFFFFFF
        u = torch.stack([(w >> (4 * i)) & 0xF for i in range(8)], -1).reshape(64, 256).float()
        mlx = (u.reshape(64, 4, 64) * s.float()[:, :, None] + b.float()[:, :, None]).reshape(64, 256)
        src = (vals.float().reshape(64, 2, 128) * scale.float()[:, :, None]).reshape(64, 256)
        self.assertTrue(torch.equal(u - 8, vals.float()))
        self.assertLess(((mlx - src).abs().max() / src.abs().max()).item(), 0.01)   # bf16 rounding of the scale only

    def test_dense_dequant_and_mtp_quantizer(self) -> None:
        from convert_mixed import dequant_bf16, quantize_affine

        g = torch.Generator().manual_seed(3)
        vals = torch.randint(-32, 32, (8, 640), generator=g)
        scale = (torch.rand(8, 10, generator=g) * 0.02 + 0.005).to(torch.float16)
        w = dequant_bf16(pack_reference(vals, 6), scale, 6, (8, 640))
        ref = (vals.float().reshape(8, 10, 64) * scale.float()[:, :, None]).reshape(8, 640)
        self.assertEqual(w.dtype, torch.bfloat16)
        self.assertLess(((w.float() - ref).abs() / ref.abs().clamp(min=1e-6)).max().item(), 0.008)
        src = (torch.randn(16, 256, generator=g) * 0.05).to(torch.bfloat16)
        words, s, b = quantize_affine(src)
        ww = words.to(torch.int64) & 0xFFFFFFFF
        u = torch.stack([(ww >> (4 * i)) & 0xF for i in range(8)], -1).reshape(16, 256).float()
        deq = (u.reshape(16, 4, 64) * s.float()[:, :, None] + b.float()[:, :, None]).reshape(16, 256)
        rel = ((deq - src.float()).pow(2).mean().sqrt() / src.float().pow(2).mean().sqrt()).item()
        self.assertLess(rel, 0.15)


if __name__ == "__main__":
    unittest.main()
