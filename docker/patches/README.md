# docker/patches/

The image build (docker/Dockerfile.native) applies `tensorfold-native-v1.0.4.patch` to TensorFold at the release
`recipe.toml` pins (`[engine] tensorfold_sha`, v1.0.4). The patch's sha256 is pinned too (`patch_sha256`); the build
refuses a file that differs, the image carries the hash as a label, and `spark up` refuses an image whose label
differs. A changed patch is a new pin and new evidence.

Until 2026-10-10 the recipe also built TensorFold's Python engine (`python-0.6` at `ed78d6f`) with a patch of its own
(the `affine-experts` loader, int8 dense linears, prompt-pass fusions, structured output on two ranks). Upstream froze
that engine and the recipe dropped it; its receipts stay under `evidence/` (2026-10-07 to 2026-10-09).

## What the patch adds (`tensorfold-native-v1.0.4.patch`)

The image builds TensorFold v1.0.4 (`d4fc196`) with this patch, docker/Dockerfile.native:
`git diff v1.0.4 native-v1.0.4` of this project's port (branch `native-v1.0.4` of github.com/BobClawblaw/TensorFold).
It adds:

- `zig/src/families/flashnext_cuda/`: the weight loader for the converted checkpoint, the forward, CUDA graphs for
  single streams (split at the cross-rank gathers, one captured step shared by every sequence), shared rounds for up to 16 streams (DeltaNet, keeps and attention batched across streams, up to 128
  rows a round), prompts admitted together, MTP drafts batched across streams, caches that grow as they are used
  (int8 or int4, `--kv-dtype`), growth agreed by both ranks before a round so a refusal ends one stream cleanly,
  kept prompt states (a next turn resumes one token before the last prompt's end), sampling (each rank's top-64
  candidates drawn on the host), grammar masks, and the vision tower.
- The extension kernels it launches (`zig/kernels/cuda/flashnext/`): device copies of this recipe's Python extensions,
  plus the sampling, grammar-mask, vision and multi-stream DeltaNet kernels.
- Structured output: `zig/src/grammar/tf_grammar.cc` (libtfgrammar, a C ABI over xgrammar 0.2.8, which the image
  builds from its pinned tag) and `zig/src/core/lanes/grammar.zig`.
- Image and video inputs in the native server: `zig/src/vision/` and `zig/src/server/vision_inputs.zig`, with
  stb_image vendored under `zig/vendor/stb` (public domain/MIT) and libtfvideo against PyAV's FFmpeg. Public HTTPS
  URLs come through `zig/src/server/media_fetch.zig` (`--vision-urls`).
- The two-rank link, and the tools under `tools/zig/` that capture and pack the kernels.

The Triton kernels it launches are captured from the Python engine and shipped beside it (`docker/kernels/sm121`,
217 variants, the int4 and multi-stream attention variants included). Pinned by sha256 in `recipe.toml`
(`native_patch_sha256`). To change it: change the branch, run its tests and the GPU exactness checks, regenerate with
`git diff v1.0.4 <head>`, re-pin, rebuild, rerun the suite.
