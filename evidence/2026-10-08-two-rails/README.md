# 2026-10-08: both halves of the QSFP port as NCCL rails

Shipped image (pin e286b134), `hca = "rocep1s0f1,roceP2p1s0f1"`, second half addressed in its own /24 at MTU 9000.

- Gather benchmark (two ranks, NCCL all-gather of fp32 rows, 2560 wide): 2,048 rows 1,912 us on one rail, 1,109 us on
  two; 1 / 4 / 16 rows in a CUDA graph 49 / 60 / 75 us on one, 50 / 57 / 57 us on two.
- Prompt passes, one prompt, 1 reply token: 7,089 tokens 2.8 s (2,565 tok/s), 28,442 10.8 s (2,639), 114,587 50.0 s
  (2,292), 224,822 110.2 s (2,040). With 4,096-row chunks: 2,530 / 2,567 / 2,275, no gain.
- Drafted equals undrafted 12/12, four streams together equal alone 12/12, vision 12/12.
- Rulers at 1 and 16 users: `ruler-c1-c16.json` (16 users: prose 300.7, code 519.4, structured 377.9, list 540.2).
