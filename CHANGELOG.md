# Changelog

## 2026-10-08

- **Shipped image verified on its own** (no source tree mounted): exactness 12/12 twice, vision 12/12, rulers as the development tree. Fused the hyper-connection down projection's slice sums into its activation and the shared expert's SwiGLU into one kernel (16 users, code: 415 -> 475 tok/s); prompt-sized int8 calls on 64 x 128 tiles (the dense matmuls 2x, a 28k prompt 15.6 -> 13.6 s, the same bits). The ranks' point-to-point link opens at startup: the image exchange had registered memory mid-serving and the worker refused it (ibv_reg_mr), hanging the first image request on the 1M default.
- **Tool calls 60/60:** the matcher accepts only true equivalents (an expression with the same value, a place named with its city), with tests that wrong calls still fail; the tool instruction adds "When a request asks for several things, make one call for each of them" (a dropped second call).
- **Long-context check to 225k**, needles at the 2% and 98% edges, ten spread facts at every length: 55/55.

## 2026-10-07

- **Multi-user and 1M by default.** `profile = "concurrent"` (16 streams) and `yarn = true` with `context = 1048576`; `PROFILE=serial` and `YARN=0 CONTEXT=262144` switch back. Measured 1 / 4 / 8 / 16 users: prose 67 / 143 / 212 / 294 tok/s aggregate, code 106 / 225 / 360 / 415. Two fixes it took: the int8 kernel's prompt-sized calls now keep decode's K slices (a row's bits no longer depend on what shares the round: four streams together equal alone 12/12), and its 128-row tiles fit the GPU's shared memory (16 users had failed on the head). `spark up` retries when the worker's NCCL memory registration fails (GB10, after a rank frees its memory).
- **Weights on Hugging Face (public):** `spark up` downloads the converted checkpoint from BobClawblaw/Qwen3.8-Flash-Next-INT4-Mixed-TensorFold at a pinned revision instead of converting locally; a locally converted folder with the same converter counts as the same. `quantization_config.format` is a string (`affine-experts-v2`) so the Hub parses the config; the card carries the image-text-to-text pipeline and vision/video tags.
- **First boot on the pair and the first speed work.** Two engine fixes the boot needed (the vision config's newer
  `model_type`; fp32 partial sums from bf16 linears on two ranks). Converter format 2: the int6 and int8 linears kept
  as int8 with fp32 group scales (exact, half the bytes of bf16) on a new int8 kernel (`q8.py`); the MTP layer's
  linears int8 at load (drafts only); the int8 tile table retuned on an idle GPU with DRAM-cycled weights. One user,
  prose / code / structured / list: 54.0 / 87.1 / 72.2 / 91.8 tok/s (bf16 dense) to 68.9 / 121.5 / 88.6 / 117.4.
  `spark up` treats a 25 GB image copy to the worker as a stop (the worker's first NCCL memory registration fails
  right after one). Suite extended with MMLU, HumanEval (sandboxed), MGSM and six serving checks.
- First version. A clean-room recipe: `python3 -m spark` orchestrator (stdlib Python; settings, guards, image,
  download, conversion, both ranks, readiness, memory watchdog, stop stamp), the checkpoint converter
  (`spark/convert_mixed.py`), the TensorFold patch with the `affine-experts` loader branch and the plan-kernel skip,
  the quality suite (`quality/`) and the decode ruler (`bench/`), tests for the guards, the converter and the suite's
  checkers, and the first measurements under `evidence/`.
