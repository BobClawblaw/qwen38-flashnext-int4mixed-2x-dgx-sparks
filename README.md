# Qwen3.8-Flash-Next INT4-Mixed on two DGX Sparks

Serve [Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound](https://huggingface.co/Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound),
the most carefully quantized public export of Qwen3.8-Flash-Next (int4 routed experts, int6 attention and DeltaNet,
int8 head and mixers, bf16 router, MTP and vision), across two NVIDIA DGX Spark (GB10, 128 GB) nodes with the
[TensorFold](https://github.com/ashhart/TensorFold) engine at tensor-parallel 2: the full 262,144-token window on
both ranks by default with a 1M-token window (static YaRN), sixteen concurrent streams by default, MTP and copy drafts, image and video input, tool calls, an OpenAI-compatible
API, and a systemd unit.

TensorFold does not read this checkpoint's format; this recipe converts it once per node into the engine's own
layout (the int4 experts bit for bit, the rest to bf16 and e4m3) and adds the loader branch that serves it
([`docs/design.md`](docs/design.md), [`docker/patches/README.md`](docker/patches/README.md)). The recipe is built
from scratch: a stdlib-only Python orchestrator, its own quality suite and ruler, no inherited code and no harness
dependencies.

## Measured on this pair

The defaults: sixteen concurrent streams, a 1,048,576-token window (static YaRN x4), int8 KV, MTP drafts up to 15,
vision on, TP=2 over the RoCE link. `bench/ruler.py`, greedy, thinking off, 256-token replies, a distinct prompt per
stream, median of 2 runs. Receipts: [`evidence/2026-10-07-first-boot/`](evidence/2026-10-07-first-boot/).

| Users | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 67.0 | 106.0 | 85.9 | 106.6 | 0.07-0.10 s |
| 4 | 143.3 | 224.6 | 188.4 | 267.7 | 0.12-0.20 s |
| 8 | 211.5 | 360.1 | 270.1 | 395.9 | 0.24-0.31 s |
| 16 | 294.1 | 414.9 | 370.3 | 534.9 | 0.52-0.68 s |

Aggregate tokens a second (per stream at 16 users: 18.5 / 24.7 / 22.9 / 33.7). One user on `PROFILE=serial` (CUDA
graphs, no YaRN): prose 68.9, code 121.5, structured 88.6, list 117.4. Exactness on the defaults: drafted equals
undrafted 12/12, four streams together equal alone 12/12.

How the build got here, one user on the serial profile:

| Build | prose | code | structured | list |
|---|---:|---:|---:|---:|
| v1: dense linears as bf16 | 54.0 | 87.1 | 72.2 | 91.8 |
| v2: dense linears as int8 + group scales (exact) | 69.2 | 97.8 | 84.6 | 110.3 |
| v2 + int8 MTP layer + retuned int8 tiles | 68.9 | 121.5 | 88.6 | 117.4 |

The draft depth and confidence were swept (6 to 15 drafts, 0.55 to 0.80): 15 at 0.70 stays; nothing beat it beyond
noise, and a low threshold or short drafts cost up to 10%.

## Where a decode round goes

A round on rank 0 reads about 3 GB: the routed experts it picks (int4), the int8 dense projections (DeltaNet,
attention, shared expert, hyper-connection mixers replicated on both ranks, its half of the head) and the cache.
The int8 kernel streams those at 190-240 GB/s on an idle GB10 (DRAM-cycled measurement in
`evidence/.../tuning`), against 75-100 GB/s for TensorFold's bf16 kernel at these shapes under load, which is what
the v1 to v2 step bought. The two ranks exchange partial sums twice a layer (96 gathers a forward); measured
between the nodes they cost 15 us eager and 44-71 us inside a CUDA graph.

## Requirements

- Two DGX Sparks on the QSFP RoCE link, Docker and the NVIDIA Container Toolkit on both, key-based ssh from the head
  to the worker, Python 3.11+ on the head (the orchestrator is stdlib only), the `hf` CLI for the download.
- Disk on the head: the snapshot (165 GB, the bf16 n-gram shards are most of it) plus the converted checkpoint
  (about 130 GB); on the worker: the converted checkpoint (copied from the head). Plus 25 GB for the image on each.
- Exclusive GPUs: `spark up` refuses to start beside another GPU container.

## Quick start

```bash
git clone https://github.com/BobClawblaw/qwen38-flashnext-int4mixed-2x-dgx-sparks.git && cd qwen38-flashnext-int4mixed-2x-dgx-sparks
cp cluster.example.toml cluster.toml && $EDITOR cluster.toml   # head_ip, worker (user@host), hca, port
python3 -m spark validate            # the settings and the guards, touches nothing
python3 -m spark up                  # sixteen streams, 1,048,576-token window; waits for the API
python3 -m spark status
PROFILE=serial python3 -m spark up   # one stream on CUDA graphs (and response_format)
YARN=0 CONTEXT=262144 python3 -m spark up   # the trained window on the plain rotary
python3 -m spark down
python3 -m spark image --rebuild   # after a patch change: rebuild on the head; the worker gets a copy on the next up
```

The weights come ready-made from [BobClawblaw/Qwen3.8-Flash-Next-INT4-Mixed-TensorFold](https://huggingface.co/BobClawblaw/Qwen3.8-Flash-Next-INT4-Mixed-TensorFold)
at the revision `recipe.toml` pins (`converted_repo`, `converted_revision`): about 119 GB, the converter's output for
the pinned source snapshot. Set `converted_repo = ""` to convert locally from Minachist's snapshot instead (a 165 GB
download and about 25 GPU minutes); a folder converted locally with the same converter is accepted as the same.

The first `up` builds the image (a few minutes on top of the NGC base, pulled once), downloads the converted checkpoint (or converts the
snapshot), copies it to the worker, starts rank 1 there over
ssh, then rank 0, and waits for `/health` and `/v1/models`. Later starts load in about two minutes.

Autostart: `systemd/qwen38-int4mixed.service` (install lines at its top). A restart waits out `settle_seconds`
(30) after the stop before rank 1 starts; on a GB10 pair a rank started within seconds of a stop fails NCCL's first
memory registration.

## Settings

`recipe.toml` holds the defaults, `cluster.toml` your pair's values, and any setting can be overridden by an
environment variable of the same name in upper case (`PORT=8001 python3 -m spark up`).

| Setting | Default | Notes |
|---|---|---|
| `profile` | `concurrent` | `concurrent`: sixteen streams (`parallel=16`); `serial`: one stream on CUDA graphs, serves `response_format` |
| `yarn` | true | static YaRN x4 over the trained 262,144 (a profile folder's `config.json`; the weights untouched). Every prompt sees the scaled rotary, short ones too (Qwen's card notes a possible cost on short texts); `yarn = false` keeps the plain rotary |
| `context` | 1048576 | up to 1,048,576 with `yarn`, up to 262,144 without |
| `kv_dtype` | `int8` | `bf16`, `int8`, `int4` |
| `mtp_drafts`, `mtp_confidence` | 15, 0.70 | `mtp_drafts=0` serves without drafts |
| `vision` | true | images and video on both ranks: the tower on rank 0, rank 1 receives each request's features |
| `tool_system` | an instruction | added to tool requests without a system message; empty serves none |
| `max_tokens` | 32768 | the reply cap when a request sets none |
| `hca`, `iface`, `master_port` | | NCCL: pin one HCA, GB10 exposes dead ones |
| `extra_args`, `extra_env` | | passed to the engine; flags the recipe owns are refused |

`python3 -m spark validate` refuses, before anything runs: a window above the trained 262,144 with `yarn = false`
or outside (262,144, 1,048,576] with `yarn = true`, zero-padded or non-decimal integers, a patch whose sha256 is not
the pinned one, `extra_args` that re-set a flag the recipe builds (`--tp`, `--context`, `--parallel`, `--vision`,
any prefix of them), an `extra_env` entry that is not `KEY=VALUE`, `tp` other than 1 or 2.

## Quality suite and ruler

```bash
python3 -m quality.suite --url http://127.0.0.1:8000 --model Qwen3.8-Flash-Next-INT4-Mixed --out evidence/my-run
python3 bench/ruler.py --url http://127.0.0.1:8000 --model Qwen3.8-Flash-Next-INT4-Mixed --concurrency 1 4 16
```

Checks, all written for this recipe (datasets pinned by commit or sha256, downloaded at run time):

| Check | What it measures |
|---|---|
| `drafts` | drafted replies equal `"draft": false` ones, 12 prompts (MTP and copy drafts must not change greedy text) |
| `concurrent` | four streams together give the same replies as each alone |
| `behaviour` | six serving checks: streamed equals non-streamed, facts recalled over six turns, stop strings, thinking mode (`reasoning_content` present, no tag leaks, right answers), verbatim copy of a passage and a code block, exact `max_tokens` |
| `tools` | 60 tool calls: exact function and argument set, optional arguments left out unless asked, parallel calls, no call where none is wanted |
| `json` | 30 JSON-schema prompts checked by our validator (`--guided-json` for `response_format` on the serial profile) |
| `ifeval` | 150 IFEval prompts of the 24 instruction types we implement, strict and loose |
| `gsm8k` | 250 grade-school math problems, exact number |
| `mgsm` | the same kind of problems in 8 languages (de, es, fr, ja, zh, ru, sw, bn), 30 each |
| `mmlu` | 6 questions from each of the 57 MMLU subjects (342), fixed sample, exact letter |
| `humaneval` | 164 Python functions, pass@1, the dataset's tests run in a container with no network |
| `repetition` | repeated 4-grams and early stops in eight 600-word replies |
| `long-context` | needles at 10/50/90% depth of 8k, 32k and 64k documents; ten spread facts at 32k and 64k |
| `vision` | 12 generated images: shapes, colours, counts, left/right, above/below, a blank |
| `snapshot` | the reply set; `--compare a.json b.json` diffs two servers (two ranks vs one GPU, one quantization vs another) |

Both run against any OpenAI-compatible server, which is how the comparison table above was made.

## Not supported

- With `parallel` above 1: no `response_format` / `guided_*` grammars and no logprobs on two ranks (the serial
  profile serves `response_format`).
- Each request decodes to `max_tokens` or EOS on both ranks; a client disconnect stops what is sent, not the work.
- A rank that dies mid-request leaves the other in NCCL without a timeout: `python3 -m spark down && up`.
- No `n > 1`, no presence or frequency penalties, the reasoning field is `reasoning_content`.

## Logs and evidence

`python3 -m spark logs [--rank 1]`. The engine prints one `done req` line per request (tokens, tok/s, TTFT, prefill
time, drafts accepted, the reply's token sha). Every number in this README has a file under
[`evidence/`](evidence/), written by the suite and the ruler.

## License

MIT ([LICENSE](LICENSE)); the TensorFold patch is Apache-2.0 like the code it changes ([NOTICE](NOTICE)).
