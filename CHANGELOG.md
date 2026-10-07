# Changelog

## 2026-10-07

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
