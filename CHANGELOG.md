# Changelog

## 2026-10-07

- First version. A clean-room recipe: `python3 -m spark` orchestrator (stdlib Python; settings, guards, image,
  download, conversion, both ranks, readiness, memory watchdog, stop stamp), the checkpoint converter
  (`spark/convert_mixed.py`), the TensorFold patch with the `affine-experts` loader branch and the plan-kernel skip,
  the quality suite (`quality/`) and the decode ruler (`bench/`), tests for the guards, the converter and the suite's
  checkers, and the first measurements under `evidence/`.
