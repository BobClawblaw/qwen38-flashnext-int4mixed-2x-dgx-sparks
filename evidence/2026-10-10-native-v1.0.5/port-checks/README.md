Checks from the v1.0.5 port run: the native-v1.0.5 binary (the same source as this recipe's pinned patch) served over
the recipe's image before the image was rebuilt. Full quality suite (`suite/`), the exactness tests at int8 and int4
(`m8.log`, `m4.log`), the six cache-exhaustion cases (`exhaust-*.json`) and the capacity probe with six ~192k-token
prompts (`capacity6-int8.json`). The shipping image's own ruler, server checks and suite exactness checks are one
folder up.
