# Vision on the native engine

- **Tower** (`tf-cuda-test fn-vision`): against torch's bf16 tower on CUDA (the Python engine's), relative error
  2.9-12.7% on three drawn images and the twelve suite probes, median row cosine >= 0.9996 -- the same band as torch's
  own CPU against CUDA runs (2.5-11.2%). With `TENSORFOLD_VISION_FP32=1`: within 0.02% of torch's fp32 tower.
- **Preparation** (`vision-prep`, `video-prep`): image patches and grids equal to the Python frontend's (0 of 2.5M
  values differ); the video path samples the same frames, decodes them to the same RGB (0 of 1.7M bytes, PyAV's own
  FFmpeg 9.0.2) and writes the same expanded prompt text.
- **Language model with images** (`fn-native ... vision=`): six image prompts with the Python tower's features give
  Python's 48-token replies token for token on both ranks, with the same rounds, drafts and acceptances.
- **Suite**: vision 12/12 on both engines (`native/`, `python/`) with the blank probe's corrected rule; 11/12 on the
  native engine under the old one, which forbade "square" although the canvas is a white square. The model reads the
  blank image differently under any perturbation of the features (`decoded-*.json`, the probes decoded with each
  feature set through the native engine):

| Features of the blank probe | Reply |
|---|---|
| torch bf16, CUDA (the Python engine) | "The image is entirely white with no discernible features or content." (pass) |
| torch bf16, CPU | "The image is a solid, uniform white square with no discernible features, text, or objects." (fails: "square") |
| torch fp32 | "The image is predominantly white with a subtle gradient, ..." (pass) |
| native fp32 tower (0.01% from torch fp32) | the turn's end: an empty reply (fails) |
| native bf16 tower (served) | "The image is a solid, uniform white square ..." (fails: "square") |

The other eleven probes give the same answers with every feature set.
