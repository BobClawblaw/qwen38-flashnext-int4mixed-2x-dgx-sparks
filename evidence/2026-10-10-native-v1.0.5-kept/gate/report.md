| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| soak | 60/60 | 100.0% | 184s | 3 chats x 20 turns at once beside a decoding stream (5.2k-5.9k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 60/60 concurrent turns resumed a kept state |
| drafts | 12/12 | 100.0% | 60s | drafted 89 tok/s vs undrafted 41 tok/s |
| concurrent | 12/12 | 100.0% | 29s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 11s | identical |
| multiturn | 8/8 | 100.0% | 12s | recalled 8/8 facts |
| stop | 3/3 | 100.0% | 1s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 11s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 1s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 1s | 3/3 exact |
