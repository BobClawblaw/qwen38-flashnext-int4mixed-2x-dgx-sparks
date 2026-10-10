# Design

What this recipe is made of and why each piece is the way it is. Everything here is this project's own work;
there is no upstream recipe it descends from.

## The checkpoint and what the engine needed

Minachist's INT4-Mixed AutoRound export is the most carefully quantized public form of Qwen3.8-Flash-Next we
found: int4 only where the bulk is (the 512 routed experts, 92% of the body), int6 on the DeltaNet, attention
and shared-expert linears, int8 on the mixers, embeddings, head and indexer, and bf16 on the router, the MTP layer,
the vision tower and the n-gram table. Its card reports IFBench 81.0, GPQA Diamond 90.4 and LiveCodeBench 92.4
against 81.3 / 91.7 / 91.9 for the bf16 release.

It ships as compressed-tensors `pack-quantized`, which TensorFold does not read (its Python engine read MLX affine
4-bit, NVFP4 and EXL3 for this model; its native engine served Flash Next on Metal only). Two routes were open: teach vLLM's two-Spark path to serve it (its author's patches
target 24 GB cards and a specific nightly, int6 is not in stock vLLM, and he reports no gain from tensor
parallelism), or give TensorFold a reader. The TensorFold route keeps everything this pair already has: two-rank
images and video, copy drafts, sixteen streams, the 1M profile, tool-call handling, and an engine whose
per-request lines make every claim checkable.

## Conversion instead of a new kernel

TensorFold's grouped expert kernel already takes affine 4-bit weights in groups of 32 or 64, and its NVFP4 route
already runs every non-expert linear in bf16 and reads an e4m3 n-gram table. So the checkpoint is converted once
per node (`spark/convert_mixed.py`, inside the engine's image, on the GPU, about 25 minutes) rather than served through a
new set of kernels:

| Source | Converted | Exactness |
|---|---|---|
| routed experts, int4 g128, fp16 scales | the same packed words (bit-identical), scales as two bf16 g64 scales, bias = -8 x scale | the int4 values exactly; each scale rounded fp16 -> bf16 (relative rms 1.7e-3) |
| int6 g64 and int8 g64/g128 linears | bf16 | scale x q rounded once to bf16 (relative rms 1.7e-3) |
| n-gram table, bf16, 102 GB | e4m3 with one table scale, 51 GB | 3 mantissa bits per value; the form NVIDIA ships |
| MTP experts, bf16 | affine 4-bit g64, min/max per group | drafts only; verification runs the main weights |
| router, norms, vision tower, conv taps, MTP projections | copied | exact |

The n-gram table is the one place the recipe loses more than a rounding: at bf16 it cannot sit beside the weights
on a 128 GB node (the engine locks it in host memory), and at e4m3 it does. The suite's long-context and GSM8K
checks are where a table loss would show.

## The engine

The recipe first served the converted checkpoint on TensorFold's Python engine, through a loader branch
(`affine-experts`) in a patch to its `python-0.6` line. Upstream froze that engine, and this project ported Flash Next
to TensorFold's native (Zig) engine for two CUDA ranks instead: the Python engine's extension kernels built from
source, its Triton kernels captured and packed (`docker/kernels/sm121`), and a forward that reproduces the Python
engine's tokens exactly, drafts included. On that port the recipe added shared rounds for sixteen streams, sampling,
structured output, kept prompt states, image and video input, an int4 cache option, growth agreed by both ranks, and
decode graphs split at the cross-rank gathers and shared across sequences. The Python engine and its patch were
removed on 2026-10-10; its measurements stay under `evidence/`. Details: `docker/patches/README.md`.

## The orchestrator

`python3 -m spark` is a stdlib-only Python package: settings from `recipe.toml`, a per-cluster `cluster.toml` and
the environment, validated before anything runs (windows per profile, the patch pin, flags the recipe owns);
the image built on the head and copied to the worker by ID; the snapshot downloaded and converted on the head,
the converted folder copied to the worker by rsync; rank 1 started over ssh, then rank 0; a stop stamp so a
restart waits out the worker's memory settle (a start within seconds of a stop fails NCCL's first registration);
a memory watchdog on each node; `/health` and `/v1/models` as readiness. No shell orchestration to maintain, and
every step is a function a test can call.

## The quality suite

`quality/` is the recipe's own, with no harness dependency: GSM8K exact match and IFEval instruction checks from the
original public datasets (pinned), a 60-case tool-call set with exact-argument matching, 30 JSON-schema prompts
with a validator, repeated-4-gram degeneration, drafted-versus-undrafted and concurrent-versus-alone exactness,
needles and spread-fact recall at 8k to 64k tokens, generated-image vision probes, and a reply snapshot to compare
two servers (two ranks against one GPU; this quantization against another). `bench/ruler.py` is the decode ruler.
The same suite runs against any OpenAI-compatible server, which is how the comparisons in the README were made.
