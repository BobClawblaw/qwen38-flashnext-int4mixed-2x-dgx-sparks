# docker/patches/

The image build applies `flashnext-int4mixed-0.6.6.patch` to TensorFold at the commit `recipe.toml` pins. The patch's
sha256 is pinned in `recipe.toml` (`[engine] patch_sha256`); the build refuses a file that differs, the image carries the
hash as a label, and `spark up` refuses an image whose label differs. A changed patch is a new pin and new evidence.

## What the patch changes in TensorFold 0.6.6

TensorFold's Python engine is frozen at 0.6.6 (its author moved fast-kernel work to a new engine), so this lives
in the recipe. Every part is this project's own work on TensorFold; nothing is copied from another recipe.

### Serving this checkpoint: `affine-experts` (new in this recipe)

Files: `families/qwen4_exp/cuda/affine_moe.py` (new), `weights.py` (`moe_affine`, bf16 for every non-expert linear),
`forward.py` (`moe_block` branch), `cuda/moe.py` (`MoEBuffers.plan_routed`), `cuda/experts.cu` (the plan kernels),
`families/qwen4_exp/__init__.py` (the method is accepted and checked), `tests/cuda/test_flashnext_affine_experts.py`.

- **The checkpoint** is compressed-tensors `pack-quantized`: int4 routed experts (symmetric, groups of 128), int6 for
  the DeltaNet, attention and shared-expert linears (groups of 64), int8 for the hyper-connection mixers, embeddings,
  head, indexer and PLE projections, bf16 for the router, the MTP layer, the vision tower and the 102 GB n-gram table.
  TensorFold 0.6.6 reads none of that. `spark/convert_mixed.py` turns it, once per node, into what the engine's
  Flash Next loader can take: the routed experts as affine 4-bit words in groups of 64 (the packed words are
  bit-identical, eight little-endian nibbles an int32 either way; a 128-group fp16 scale becomes two bf16 64-group
  scales with bias minus eight scales, so a weight decodes to scale x (q - 8) as the source defines it), the int6 and
  int8 linears dequantized to bf16 (scale x q rounded once), the n-gram table as e4m3 with one table scale (the form
  NVIDIA's export uses; half the bytes, so it fits beside the weights on a 128 GB node), the MTP experts as affine
  4-bit (they only draft; verification uses the main weights).
- **The loader** gets a third format beside MLX 4-bit and NVFP4: routed experts on the grouped affine kernel at group
  64 (the kernel already took 32 and 64), everything else on the bf16 paths the NVFP4 route uses (bf16 matmuls for
  DeltaNet, attention, hyper-connections, head, embedding, MTP projections; the FP8 n-gram reader).
- **The shared expert** is bf16 here and the affine kernel's table holds only the 512 routed experts, so the
  selection's last slot (the shared expert's id, 512) falls past the table. The plan kernels now leave such picks out
  (they used to index them), and the shared expert runs on the bf16 matmul into that slot with the NVFP4 route's
  rounding rules. A unit test checks the plan with picks past the table at small and wide sizes, and the whole
  layer against a plain-torch reference.
- **Two ranks** split every expert's width in half (whole 64-groups: 640 = 10 x 64), the shared expert likewise,
  heads and DeltaNet heads by count, the head's vocabulary in half, and gather fp32 partial sums, as the engine does
  for the MLX checkpoint.
- **Precision.** Routed experts: the stored int4 values exactly, with each scale rounded from fp16 to bf16 (relative
  rms 1.7e-3 against the source's dequantization on a sampled layer). Dense linears: bf16 rounding of scale x q
  (relative rms 1.7e-3). N-gram rows: e4m3 of bf16 with one scale. MTP experts: affine 4-bit (drafts only).
  The converter's unpacking is checked bit for bit against the `compressed-tensors` library on 4-, 6- and 8-bit
  tensors (`tests/test_convert.py`, when torch is present).

### Format 2: int8 dense linears with group scales (`q8.py`, new)

The checkpoint's int6 and int8 linears are kept as their integers in int8 bytes with fp32 group scales
(`.qweight`, `.qscale`), not dequantized to bf16: exact (the kernel multiplies the stored integers and applies the
stored scales, 3e-7 relative to an fp32 reference), half the bytes of bf16. `q8.Q8Face` stands where a bf16 face
stood (`bf16.matmul` dispatches to it), a face can mix int8 rows and bf16 rows (a hyper-connection's down projection
and its bf16 inject rows), and the tile shape is a function of the weight's shape alone (so drafted, undrafted and
concurrent rows agree bit for bit). The MTP layer's bf16 linears are quantized to int8 at load (absmax per row and
64 inputs): they only draft, the main weights verify, so replies are unchanged (drafted equals undrafted 12/12).
Tests: `tests/cuda/test_flashnext_q8.py`.

### Two fixes the first boot found

- `vision/qwen_cuda.py`: accept the vision config's newer `model_type` name, `qwen4_exp_vision` (this export's).
- `forward._mm`: a bf16 linear writes fp32 when the caller's buffer is fp32 (two ranks gather fp32 partial sums of
  the output projections; bf16 linears had only ever run on one GPU).

### A development switch

`TENSORFOLD_STAGE_TIMES=<file>` (with `TENSORFOLD_STAGE_SYNC=1` for isolated timings) records GPU time per forward
stage with CUDA events and turns decode graphs off; unset, it changes nothing.

### Carried from this project's earlier TensorFold work

- **EXL3 packs on two ranks and a sixteen-stream EXL3 decode window** (the EXL3 loader, `exl3_mm.py`,
  `cuda/exl3/experts.py`, the admission estimate): not used by this checkpoint, kept so one patch serves the pair's
  recipes.
- **YaRN from `config.json`** (`weight_types.py` `rotary_inv_freq`, `scale_rotary`): the long profile's rotary,
  computed as transformers computes it, the attention factor folded into the rotated dims of the norm scales.
- **Image and video input on two ranks, on the concurrent decoder and the serial engine** (`vision_ranks.py`,
  `multi_tp.py`, `engine.py`): rank 0 runs the tower and sends each request's features to rank 1 inside the
  admission lockstep. Merged upstream in TensorFold main (#473).
- **Copy drafts** (`draft_depth`, `decode.py`): a reply that repeats earlier text drafts the continuation whole,
  byte-identical output. Merged upstream (#468).
- **`--tool-system`** on the CUDA server and **declared-parameter tool calls** (`tool_parameters.py`, the parsers,
  the streamer).

### Validation

Inside the image, with a GPU: `tests/cuda/test_flashnext_affine_experts.py` (the plan skip, the layer against a
torch reference), and the engine's own `test_flashnext_nvfp4_loader.py`, `test_nvfp4_experts.py`,
`test_qwen4_exp_exl3.py`, `test_flashnext_yarn_host.py` (63 passed, 8 skipped on 2026-10-07). On the pair: the quality
suite (`quality/`) and the ruler (`bench/`), receipts under `evidence/`.

To change the patch: edit a clean TensorFold checkout at the pinned commit, run the tests, `git diff HEAD` to a new
file, pin its sha256 in `recipe.toml`, rebuild the image, rerun the suite. Never edit a patch file in place.
