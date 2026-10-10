One-stream ruler, the previous image (patch 8bea0406) and the kept-state fix (d5eacba9) alternated in one session,
each on a freshly started server, two runs a start (prev, fix, prev, fix; `ab.log`). The ranges overlap: prose
74.3-77.4 vs 75.4-77.5, code 120.7-125.5 vs 122.5-125.2, structured 95.9-99.5 vs 96.6-99.1, list 125.0-127.7 vs
122.3-126.6 tok/s. The lower one-stream numbers in `ruler-fresh.json` beside this folder were the same drift.
