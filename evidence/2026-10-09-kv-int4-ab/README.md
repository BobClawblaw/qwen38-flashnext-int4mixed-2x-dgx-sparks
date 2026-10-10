# int4 against int8 KV cache, Python and native engines (2026-10-09)

Two GB10s, TP=2, the recipe's `concurrent` profile (1M context, 16 streams, MTP 15 @ 0.70, vision on).
Python: image `py0.6-ed78d6f`, `ENGINE=python KV_DTYPE=…`. Native: the recipe's native image with the
TensorFold `kv-int4` branch binary and a kernel set carrying both cache widths (189 variants), `--kv-dtype …`.

## Speed: bench/ruler.py, aggregate tok/s (prose / code / structured / list), 256 tokens, 2 runs

| engine, cache | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|
| Python int8 | 69 / 109 / 92 / 113 | 147 / 232 / 193 / 278 | 216 / 371 / 275 / 408 | 302 / 509 / 383 / 545 |
| Python int4 | 67 / 113 / 72 / 113 | 137 / 223 / 179 / 267 | 213 / 343 / 266 / 398 | 282 / 478 / 379 / 543 |
| native int8 | 67 / 106 / 87 / 108 | 142 / 228 / 176 / 251 | 195 / 334 / 232 / 353 | 258 / 409 / 306 / 432 |
| native int4 | 63 / 105 / 76 / 108 | 144 / 207 / 170 / 234 | 188 / 302 / 226 / 338 | 255 / 397 / 304 / 422 |

int4 runs 0–10% under int8: the replies differ a little (int4 is a different cache), so drafts land differently
(c=1 structured is the biggest swing). Attention reads at most 2,051 selected keys a row (QSA), so the cache's width
barely moves decode time.

## Long prompts: longctx.py (prefill tok/s, then decode tok/s of a 256-token reply)

| engine, cache | 25.7k | 102.7k | 180.6k |
|---|---|---|---|
| Python int8 | 2,737 / 38.3 | 2,449 / 32.8 | 2,219 / 33.0 |
| Python int4 | 2,731 / 39.1 | 2,439 / 35.9 | 2,219 / 34.1 |
| native int8 | 2,735 / 63.9 | 2,443 / 58.9 | 2,219 / 63.4 |
| native int4 | 2,746 / 67.0 | 2,447 / 60.2 | 2,224 / 59.1 |

## Quality: python3 -m quality.suite

| check | Python int8 | Python int4 | native int4 |
|---|---|---|---|
| drafts, concurrent, streaming, multiturn, stop, thinking, copy, max-tokens, json, repetition, vision | all pass | all pass | all pass |
| tools | 60/60 | 58/60 | 58/60 |
| ifeval (loose / strict) | 130 / 125 | 130 / 123 | 130 / 123 |
| gsm8k | 241/250 | 240/250 | 240/250 |
| mgsm | 219/240 | 219/240 | 219/240 |
| mmlu | 283/342 | 282/342 | 282/342 |
| humaneval | 157/164 | 156/164 | 156/164 |
| long-context | 55/55 | **failed (NoRoom, see below)** | 55/55 |

The native int4 forward equals Python's int4 reply token for token (tf-cuda-test fn-native kv4: prefill + 48
tokens with MTP drafts, 23 rounds, 38 drafted, 24 accepted on both; `fn-native-int4.txt`), and the int8 forward is
unchanged (`fn-native-int8.txt`). Native int4 scores what Python int4 scores, check for check.

## Memory

Bytes a position takes per rank: 13 attention caches (12 layers and the MTP head) x (keys + values + their scales
+ index keys + pooled blocks) = 13 x 864 at int8, 13 x 608 at int4 (-30%). The native engine leaves about 9.3 GiB
per rank for the streams' growing caches, so about 0.85M tokens resident across all streams at int8 and about 1.2M at int4.

Capacity probe (six distinct ~192k-token prompts at once, `capacity6.json`):
- native int8: at about 350 s (about 0.77M tokens in) one request was refused ("no memory left"), and the next cache
  growth hit `CUDA_ERROR_ILLEGAL_ADDRESS` (`zero grown cache`). Every request failed with `CudaFailed`; the server needed a restart.
- native int4: at 483 s (about 1.06M tokens in) one request was refused cleanly, but the five others never finished
  (GPU busy, no tokens for an hour, client timeout).
Both are bugs in the native engine's handling of an exhausted cache budget, and they exist at int8 as well; int4 only
moves where it happens.

## Python int4: rank 1's memory grows after long prompts

After the 180k prompt, Python at int4 left rank 1 at 108 GiB used (12 GiB available). Every later request failed with
`NoRoom: the proposed round cache growth does not fit on both ranks`, including the suite's long-context check and
short requests (`python-int4/suite-after-longctx-NoRoom`, `suite-after-long.log`). At int8 the same sequence passes
(long-context 55/55, 0 NoRoom lines). The native engine at int4 does not have this problem.
