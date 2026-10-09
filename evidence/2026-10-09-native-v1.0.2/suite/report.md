| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 61s | drafted 82 tok/s vs undrafted 41 tok/s |
| concurrent | 12/12 | 100.0% | 32s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 12s | identical |
| multiturn | 8/8 | 100.0% | 13s | recalled 8/8 facts |
| stop | 3/3 | 100.0% | 1s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 13s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 1s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 1s | 3/3 exact |
| tools | 60/60 | 100.0% | 45s | all exact |
| json | 30/30 | 100.0% | 8s | all valid |
| ifeval | 130/150 | 86.7% | 321s | loose 130/150, strict 125/150, 150 covered prompts |
| gsm8k | 241/250 | 96.4% | 344s | misses at [40, 85, 87, 93, 98, 119, 163, 187, 234] |
| mgsm | 219/240 | 91.2% | 490s | de 28/30, es 29/30, fr 26/30, ja 27/30, zh 27/30, ru 30/30, sw 27/30, bn 25/30 |
| mmlu | 283/342 | 82.7% | 637s | 57 subjects x 6; weakest college_mathematics 1/6, high_school_mathematics 2/6, business_ethics 3/6 |
| humaneval | 157/164 | 95.7% | 138s | pass@1 greedy; failed ['HumanEval_116', 'HumanEval_129', 'HumanEval_132', 'HumanEval_140', 'HumanEval_145', 'HumanEval_147'] ... |
| repetition | 8/8 | 100.0% | 39s | repeated 4-grams 0.40% mean, max 1.09%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 802s | needles 15/15 to 224k, facts 10/10@28k, 10/10@57k, 10/10@114k, 10/10@225k, slowest prefill 1998 tok/s |
| vision | 11/12 | 91.7% | 3s | blank: 'The image is a solid, uniform white square with no' |
| snapshot | 12/12 | 100.0% | 21s | replies written to evidence/2026-10-09-native-v1.0.2/suite/replies.json |
