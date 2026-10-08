# 2026-10-08: prompt-pass fusions (patch pin e286b134)

- `bake-and-verify.log`: patch regenerated and pinned, image rebuilt, the shipped image booted alone; drafted equals
  undrafted 12/12, four streams together equal alone 12/12, vision 12/12; rulers at 1 and 16 users; prompt passes at
  7k, 28k, 115k and 225k tokens.
- A second fresh boot measured 16 users again (3 runs): prose 299.9, code 507.2, structured 374.4, list 541.2 tok/s
  aggregate (the 441.7 code figure in the log was a noisy run).
- Development copy, before baking: long-context needles and facts 42/42 to 114k; the engine's int8 and expert tests
  17 passed, including the new bit-exactness test of the fused up projection and mix.
