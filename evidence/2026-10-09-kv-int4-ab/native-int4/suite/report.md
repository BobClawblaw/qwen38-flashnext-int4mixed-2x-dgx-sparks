| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 63s | drafted 77 tok/s vs undrafted 40 tok/s |
| concurrent | 12/12 | 100.0% | 33s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 13s | identical |
| multiturn | 8/8 | 100.0% | 11s | recalled 8/8 facts |
| stop | 3/3 | 100.0% | 1s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 15s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 1s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 1s | 3/3 exact |
| tools | 58/60 | 96.7% | 48s | 'Give me the prices of MSFT and TSLA.': got [('stock_price', {'ticker': 'msft'})] |
| json | 30/30 | 100.0% | 8s | all valid |
| ifeval | 130/150 | 86.7% | 427s | loose 130/150, strict 123/150, 150 covered prompts |
| gsm8k | 240/250 | 96.0% | 465s | misses at [12, 85, 87, 93, 119, 163, 184, 187, 205, 234] |
| mgsm | 219/240 | 91.2% | 657s | de 27/30, es 29/30, fr 27/30, ja 28/30, zh 27/30, ru 28/30, sw 27/30, bn 26/30 |
| mmlu | 282/342 | 82.5% | 867s | 57 subjects x 6; weakest high_school_mathematics 2/6, business_ethics 3/6, college_chemistry 3/6 |
| humaneval | 156/164 | 95.1% | 161s | pass@1 greedy; failed ['HumanEval_116', 'HumanEval_129', 'HumanEval_132', 'HumanEval_140', 'HumanEval_145', 'HumanEval_147'] ... |
| repetition | 8/8 | 100.0% | 53s | repeated 4-grams 0.26% mean, max 0.80%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 787s | needles 15/15 to 224k, facts 10/10@28k, 10/10@57k, 10/10@114k, 10/10@225k, slowest prefill 2036 tok/s |
| vision | 12/12 | 100.0% | 3s | all probes answered |
| snapshot | 12/12 | 100.0% | 21s | replies written to <scratch>/native-int4/suite/replies.json |
