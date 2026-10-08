# Working on this recipe

Serve `Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound` on two DGX Sparks with TensorFold at TP=2. Everything in
this tree is original to this project (no upstream recipe); keep it that way: no code copied from other recipes,
no harness dependencies, stdlib Python only.

- **Entry point:** `python3 -m spark up|down|status|logs|convert|image|validate`. Settings: `recipe.toml` defaults,
  `cluster.toml` per cluster (git-ignored), environment variables in upper case. `python3 -m spark validate` is the
  dry run; CI runs it on every profile and the refusals.
- **The patch** (`docker/patches/flashnext-int4mixed-py0.6-ed78d6f.patch`) is pinned by sha256 in `recipe.toml`. Never edit it
  in place: change a clean TensorFold checkout at the pinned commit, run its tests inside the image, regenerate,
  re-pin, rebuild, rerun the suite.
- **The conversion** (`spark/convert_mixed.py`) is part of the served artefact: a changed converter is a new
  `.source` marker and a reconversion on both nodes (`up` does it).
- **Measure before you claim.** The suite (`python3 -m quality.suite`) and the ruler (`bench/ruler.py`) write JSON
  under the folder you name; commit those under `evidence/` and point the README row at the file.
- **Restarts:** `down` stamps the stop; `up` waits out `settle_seconds` before rank 1 starts. Do not shorten it on
  a GB10 pair: the worker refuses NCCL's first memory registration right after a rank frees its pages.
- **Memory is unified:** read it with `free -h`, not `nvidia-smi`. The n-gram table (51 GB e4m3) is locked in host
  memory on each node beside the rank's weights.
- Prefill checks stop at 64k tokens; the `long` profile exists for prompts beyond the trained window, it is not a
  benchmark.
