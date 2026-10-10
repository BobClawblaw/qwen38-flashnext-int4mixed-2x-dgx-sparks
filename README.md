# Qwen3.8-Flash-Next INT4-Mixed on two DGX Sparks

Serve [Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound](https://huggingface.co/Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound),
the most carefully quantized public export of Qwen3.8-Flash-Next (int4 routed experts, int6 attention and DeltaNet,
int8 head and mixers, bf16 router, MTP and vision), across two NVIDIA DGX Spark (GB10, 128 GB) nodes with the
[TensorFold](https://github.com/ashhart/TensorFold) engine at tensor-parallel 2: the full 262,144-token window on
both ranks by default with a 1M-token window (static YaRN), sixteen concurrent streams by default, MTP and copy drafts, image and video input, tool calls, an OpenAI-compatible
API, and a systemd unit.

TensorFold does not read this checkpoint's format, so the recipe serves it converted into the engine's own layout:
the int4 experts bit for bit, the int6 and int8 linears as their exact integers with their group scales, the n-gram
table as e4m3. `spark up` downloads that converted checkpoint from this project's Hugging Face repository,
[BobClawblaw/Qwen3.8-Flash-Next-INT4-Mixed-TensorFold](https://huggingface.co/BobClawblaw/Qwen3.8-Flash-Next-INT4-Mixed-TensorFold)
(public, pinned to a revision), so there is nothing to convert; the converter ships too, for checking or rebuilding it
from Minachist's export. The patch adds the loader branch that serves it
([`docs/design.md`](docs/design.md), [`docker/patches/README.md`](docker/patches/README.md)). The recipe is built
from scratch: a stdlib-only Python orchestrator, its own quality suite and ruler, no inherited code and no harness
dependencies.

## Measured on this pair

The defaults: TensorFold v1.0.4's native engine with this recipe's patch (pin `fcf3092a`), sixteen concurrent
streams, a 1,048,576-token window (static YaRN x4), int8 KV, MTP drafts up to 15, vision on, TP=2 over the RoCE link.
`bench/ruler.py`, greedy, thinking off, 256-token replies, a distinct prompt per stream, median of 3 runs, on a freshly
started server. Receipts: [`evidence/2026-10-10-native-v1.0.4/`](evidence/2026-10-10-native-v1.0.4/).

| Users | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 69.1 | 110.0 | 88.7 | 112.5 | 0.07-0.09 s |
| 4 | 157.1 | 252.3 | 197.4 | 288.6 | 0.15-0.21 s |
| 8 | 221.9 | 382.2 | 279.7 | 412.3 | 0.23-0.32 s |
| 16 | 297.3 | 511.8 | 383.5 | 519.8 | 0.41-0.55 s |

Aggregate tokens a second. The Python engine (`ENGINE=python`, pin `e286b134`) on the same pair:

| Users | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 68.7 | 109.0 | 89.0 | 109.9 | 0.06-0.10 s |
| 4 | 147.4 | 230.0 | 197.3 | 279.9 | 0.12-0.19 s |
| 8 | 214.1 | 366.0 | 273.2 | 403.7 | 0.24-0.30 s |
| 16 | 299.5 | 475.2 | 372.0 | 542.0 | 0.51-0.72 s |

One stream undrafted: 40 tok/s (native), 35 (Python).

Prompt passes on the Python engine (time to the first token, one prompt, needle found in each); the native engine's
long-context check runs its slowest, 225k, at 2,012 tok/s:

| Prompt | Time | Rate |
|---:|---:|---:|
| 7,089 tokens | 2.8 s | 2,565 tok/s |
| 28,442 | 10.8 s | 2,639 tok/s |
| 114,587 | 50.0 s | 2,292 tok/s |
| 224,822 | 110.2 s | 2,040 tok/s |

With both halves of the QSFP port as NCCL rails. On one rail: 2,342 / 2,351 / 2,091 / 1,871 tok/s; before the
prompt-pass fusions (pin `f188bd45`, one rail): 2,124 / 2,096 / 1,888 / 1,741, so 17-26% faster in all
([`evidence/2026-10-08-prefill-fusions/`](evidence/2026-10-08-prefill-fusions/)).

## Quality

The recipe's suite on the defaults (`quality/`, datasets pinned), greedy with thinking off (`enable_thinking: false`;
scores with thinking on will differ, mostly upward on maths and code). The native v1.0.4 image and the Python engine
score the same on every check ([`suite/report.md`](evidence/2026-10-10-native-v1.0.4/suite/report.md)):

| Check | Result |
|---|---|
| drafted equals undrafted; four streams together equal alone | 12/12; 12/12 |
| streaming = non-streaming, multi-turn recall, stop strings, thinking mode, verbatim copy, exact `max_tokens` | all pass |
| tool calls (60: exact function and arguments, optional arguments left out, parallel calls, no-call cases) | 60/60 |
| JSON schemas | 30/30 |
| IFEval (150 prompts, the 24 instruction types implemented) | 130 loose / 125 strict |
| GSM8K (250) | 241 (96.4%) |
| MGSM (8 languages x 30) | 219 (91.3%) |
| MMLU (6 a subject, 342) | 283 (82.7%) |
| HumanEval (164, pass@1, sandboxed) | 157 (95.7%) |
| repetition (eight 600-word replies) | 8/8, 0.40% repeated 4-grams |
| long context: needles at 2/50/98% of 128k and 225k and 10/50/90% of 8-57k; ten spread facts at 28k to 225k | 55/55 |
| vision (12 drawn probes) | 12/12 |

How the speed got here, one user:

| Build | prose | code | structured | list |
|---|---:|---:|---:|---:|
| v1: dense linears as bf16 (serial) | 54.0 | 87.1 | 72.2 | 91.8 |
| v2: dense linears as int8 + group scales, exact (serial) | 69.2 | 97.8 | 84.6 | 110.3 |
| + int8 MTP layer, retuned tiles (serial) | 68.9 | 121.5 | 88.6 | 117.4 |
| defaults (concurrent, 1M), + fused hyper-connection and shared-expert kernels | 68.7 | 109.0 | 89.0 | 109.9 |

The draft depth and confidence were swept (6 to 15 drafts, 0.55 to 0.80): 15 at 0.70 stays.

## The native engine (TensorFold 1.0.4)

`engine = "native"` (the default) serves the converted checkpoint on TensorFold v1.0.4, the Zig engine, with
[`docker/patches/tensorfold-native-v1.0.4.patch`](docker/patches/tensorfold-native-v1.0.4.patch): this project's
Flash Next family for two CUDA ranks (the release serves Flash Next on Metal only; the port is on
[BobClawblaw/TensorFold `native-v1.0.4`](https://github.com/BobClawblaw/TensorFold/tree/native-v1.0.4)).
`ENGINE=python` keeps the Python engine.

- Up to 16 streams in shared rounds (DeltaNet, keeps and attention batched across streams, up to 128 rows a round),
  prompts admitted together, MTP drafts batched across streams; CUDA graphs for a single stream.
- Sampling (each rank's top-64 candidates drawn on the host; the full row when top-p or min-p reach past them),
  structured output (`response_format`, `guided_json`/`regex`/`choice`/`grammar`, through xgrammar 0.2.8), kept
  prompt states (a next turn resumes one token before the last prompt's end: 0.12 s to the first token instead of
  2.1 s at 5.5k tokens), images and video, public HTTPS media URLs, `--tool-system`.
- int8 or int4 KV (`kv_dtype`). int4 holds 30% more context (about 1.2M tokens across streams against 0.85M per rank)
  at 0-10% lower throughput at 16 streams and scores within an item or two
  ([`evidence/2026-10-09-kv-int4-ab/`](evidence/2026-10-09-kv-int4-ab/)); int8 stays the default.
- Caches grow as they are used; both ranks agree on each growth before a round, so running out refuses one request
  cleanly and the server serves on (six ~192k-token prompts at once: four answered, two refused, then a short request
  served).

Exactness: greedy tokens equal the Python engine's (the reference reply token for token with the same drafts and
acceptances, at int8 and int4), shared rounds equal each stream alone, batched prompts equal prompts alone, a resumed
turn equals a fresh one, and 20 of 20 sampled replies equal the Python engine's (checked on the v1.0.2 port; the
sampler is unchanged since). Quality suite above.
`TENSORFOLD_VISION_FP32=1` runs the vision tower in fp32 (within 0.02% of torch's fp32 tower, about 2.5x its time).

## Where a prompt pass goes

A 28k and a 114k prompt with the stage timer (`TENSORFOLD_STAGE_TIMES`, `evidence/2026-10-08-shipped-image/`), shares of
the prompt pass on rank 0: the MoE about a third (the grouped expert kernels 13 s of 76 s), the hyper-connection read-out a
quarter (its elementwise write-back, norm and mix 10 s), the cross-rank gathers a fifth (each 2,048-row chunk moves 20 MB a
rank, 2.1 ms a gather at about 11 GB/s each way: NCCL stages through host memory, GPU-direct RDMA is off on GB10), the
attention a fifth (the sparse indexer's scoring grows with the prompt), the DeltaNet a fifth. The int8 matmuls are no longer
the large part after the prompt-tile change.

Done since: the read-out's write-back and norm run as one pass on int8 prompts (the byte-checked kernel TensorFold already
used for MLX 4-bit), and its int8 up projection and mix are one new kernel (337 us against 695 for a 2,048-row chunk,
bit-identical to the two kernels, with a test). Done since: both halves of the QSFP port as NCCL rails (above). Tried without gain: 4,096-row prompt chunks (also with two rails), NCCL channel, queue-pair and
buffer settings, a 9000-byte MTU on the link (RoCE at 4096-byte packets; the gathers stay at about 11 GB/s), and NCCL's
DMA-BUF and C2C GPU-direct switches (NCCL still reports GPU-direct RDMA off; `nvidia-peermem` does not load on this kernel).
Next in line: the grouped expert prompt kernels and the indexer.

## Where a decode round goes

A round on rank 0 reads about 3 GB: the routed experts it picks (int4), the int8 dense projections (DeltaNet,
attention, shared expert, hyper-connection mixers replicated on both ranks, its half of the head) and the cache.
The int8 kernel streams those at 190-240 GB/s on an idle GB10 (DRAM-cycled measurement in
`evidence/.../tuning`), against 75-100 GB/s for TensorFold's bf16 kernel at these shapes under load, which is what
the v1 to v2 step bought. The two ranks exchange partial sums twice a layer (96 gathers a forward); measured
between the nodes on the native engine they cost 16 us eager and 25 us inside a CUDA graph for one row (37 and 44 us
for six rows).

## Requirements

- Two DGX Sparks on the QSFP RoCE link, Docker and the NVIDIA Container Toolkit on both, key-based ssh from the head
  to the worker, Python 3.11+ on the head (the orchestrator is stdlib only), the `hf` CLI for the download
  (`pip install -U huggingface_hub`; the repository is public, no token needed).
- Disk: the converted checkpoint on each node, 128 GB (119 GiB; downloaded on the head, copied to the worker over
  the link), plus 25 GB for the image on each. Converting locally instead (`converted_repo = ""`) also needs
  Minachist's 165 GB snapshot on the head.
- Exclusive GPUs: `spark up` refuses to start beside another GPU container.

### Both halves of the port

A Spark's QSFP port reaches the GB10 over two PCIe Gen5 x4 links, so one cable shows up as two netdevs and two RoCE
devices (`enp1s0f1np1` / `rocep1s0f1` and `enP2p1s0f1np1` / `roceP2p1s0f1` on the second port). Each half carries about
112 Gb/s of the port's 200; with only one addressed, NCCL uses one, and a 2,048-row prompt gather took 1.9 ms (about
11 GB/s each way). Address the second half on both Sparks in its own subnet, persistently, and list both in `hca`:

```bash
sudo nmcli connection add type ethernet ifname enP2p1s0f1np1 con-name roce-twin ipv4.method manual \
  ipv4.addresses 10.0.1.1/24 ipv4.never-default yes ipv6.method disabled 802-3-ethernet.mtu 9000   # .2 on the worker
```

The same gather then takes 1.1 ms, prompt passes run 9-12% faster, and decode is unchanged (16-row gathers 75 to
57 us). NCCL keeps the first device's address for its own connection setup (`iface`).

## Quick start

```bash
git clone https://github.com/BobClawblaw/qwen38-flashnext-int4mixed-2x-dgx-sparks.git && cd qwen38-flashnext-int4mixed-2x-dgx-sparks
pip install -U huggingface_hub      # the hf CLI that downloads the weights (public, no token)
cp cluster.example.toml cluster.toml && $EDITOR cluster.toml   # head_ip, worker (user@host), hca, port
python3 -m spark validate            # the settings and the guards, touches nothing
python3 -m spark up                  # sixteen streams, 1,048,576-token window; waits for the API
python3 -m spark status
PROFILE=serial python3 -m spark up   # one stream on CUDA graphs
YARN=0 CONTEXT=262144 python3 -m spark up   # the trained window on the plain rotary
python3 -m spark down
python3 -m spark image --rebuild   # after a patch change: rebuild on the head; the worker gets a copy on the next up
```

The weights come ready-made from [BobClawblaw/Qwen3.8-Flash-Next-INT4-Mixed-TensorFold](https://huggingface.co/BobClawblaw/Qwen3.8-Flash-Next-INT4-Mixed-TensorFold)
at the revision `recipe.toml` pins (`converted_repo`, `converted_revision`): 128 GB (119 GiB), the converter's output
for the pinned source snapshot, with the converter's hash in the model card. Set `converted_repo = ""` to convert locally from Minachist's snapshot instead (a 165 GB
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
| `engine` | `native` | TensorFold v1.0.4 (the Zig engine) with this recipe's patch; `python`: TensorFold's Python engine (`python-0.6`), the only one with logprobs (at `tp=1`) |
| `profile` | `concurrent` | `concurrent`: sixteen streams (`parallel=16`); `serial`: one stream on CUDA graphs |
| `yarn` | true | static YaRN x4 over the trained 262,144 (a profile folder's `config.json`; the weights untouched). Every prompt sees the scaled rotary, short ones too (Qwen's card notes a possible cost on short texts); `yarn = false` keeps the plain rotary |
| `context` | 1048576 | up to 1,048,576 with `yarn`, up to 262,144 without |
| `kv_dtype` | `int8` | `int8` or `int4` on native (int4: 30% less cache a position, 0-10% slower at 16 streams, scores within an item or two); `python` also serves `bf16` |
| `mtp_drafts`, `mtp_confidence` | 15, 0.70 | `mtp_drafts=0` serves without drafts |
| `vision` | true | images and video on both ranks: the tower on rank 0, rank 1 receives each request's features |
| `vision_urls` | true | public HTTPS image and video URLs as well as data URLs, fetched by rank 0 as the Python server does: HTTPS on 443, every resolved address public, redirects checked again, declared media types, 10 MB an image and 20 MB a request, 10 s a download. The server then makes outbound requests for whoever can reach its port: keep the port on a trusted network, or `false` for data URLs only |
| `tool_system` | an instruction | added to tool requests without a system message; empty serves none |
| `max_tokens` | 32768 | the reply cap when a request sets none |
| `hca`, `iface`, `master_port` | | NCCL: the RoCE devices to use, comma-separated (both halves of the cabled port); GB10 also exposes dead ones |
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
| `json` | 30 JSON-schema prompts checked by our validator (`--guided-json` sends `response_format`) |
| `ifeval` | 150 IFEval prompts of the 24 instruction types we implement, strict and loose |
| `gsm8k` | 250 grade-school math problems, exact number |
| `mgsm` | the same kind of problems in 8 languages (de, es, fr, ja, zh, ru, sw, bn), 30 each |
| `mmlu` | 6 questions from each of the 57 MMLU subjects (342), fixed sample, exact letter |
| `humaneval` | 164 Python functions, pass@1, the dataset's tests run in a container with no network |
| `repetition` | repeated 4-grams and early stops in eight 600-word replies |
| `long-context` | needles at 10/50/90% depth of 8k, 32k and 64k documents and at the extreme edges (2% and 98%) and middle of 128k and 250k ones; ten spread facts recalled together at 32k, 64k, 128k and 250k (`--long-max` caps the length) |
| `vision` | 12 generated images: shapes, colours, counts, left/right, above/below, a blank |
| `snapshot` | the reply set; `--compare a.json b.json` diffs two servers (two ranks vs one GPU, one quantization vs another) |

Both run against any OpenAI-compatible server, which is how the comparison table above was made.

## Known issues and limits

- Single-stream speed drops 7-10% on a server that has run for a while (cached CUDA graphs pile up across pooled
  sequences, and new requests recapture them). Restarting (`spark down && spark up`) restores it; several streams at
  once are not affected. The fix is verified on a branch and comes in the next update.
- A start can fail when the worker refuses NCCL's first memory registration (rank 0 exits during its warm-up
  gather); `spark up` again after a minute. It happens most right after a stop, which is what `settle_seconds` is for.
- Scores above are with thinking off; the suite does not yet run every check with thinking on.
- No logprobs on two ranks (either engine). Grammars (`response_format`, `guided_*`) and sampling are served on both
  engines and in both profiles.
- When the streams' caches outgrow the GPU memory left for them (about 0.85M tokens across streams at int8, 1.2M at
  int4, per rank), the request that asks for more is refused with "the GPU had no memory left for this request's
  caches: retry once another request ends"; the others carry on and the server keeps serving. Prompts that grow
  together are refused together, so six ~192k-token prompts at once may finish two to four of them.
- Each request decodes to `max_tokens` or EOS on both ranks; a client disconnect stops what is sent, not the work.
- A rank that dies mid-request leaves the other in NCCL without a timeout: `python3 -m spark down && up`.
- No `n > 1`, no presence or frequency penalties, the reasoning field is `reasoning_content`.

## Logs and evidence

`python3 -m spark logs [--rank 1]`. The engine prints one `done req` line per request (tokens, tok/s, TTFT, prefill
time, drafts accepted, the reply's token sha). Every number in this README has a file under
[`evidence/`](evidence/), written by the suite and the ruler.

## License

MIT ([LICENSE](LICENSE)); the TensorFold patch is Apache-2.0 like the code it changes ([NOTICE](NOTICE)).
