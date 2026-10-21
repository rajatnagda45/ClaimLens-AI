# Run report: v2 / development

- run_id: `badd9a59-ed21-42c5-8fd1-4bc093505357`
- provider: **OpenAIProvider** (live=True)  model: `gpt-4o-mini`
- prompt: `extract-classify-v2`  evaluator: `sentence-spans-and-labels-v4`
- fact: `speaker-catalogue-v4`  dataset: `long-brand-answers-50-v4`
- timestamp: 2026-10-21T08:58:48.214076+00:00
- source_hashes: {"answers.csv": "fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724", "brand_facts.csv": "2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2", "development_gold.csv": "afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850"}

## Metrics
- G=376 P=313 M=296 C=223
- extraction: P=0.946 R=0.787 **F1=0.859**
- joint: P=0.712 R=0.593 **F1=0.647** (primary)
- gold coverage C/G = 0.593

- answers: success=34 empty=0 error=6

## Per-answer status

- A01: success claims=9 latency=8762ms
- A02: success claims=10 latency=9185ms
- A03: success claims=8 latency=8807ms
- A04: success claims=10 latency=9343ms
- A05: success claims=8 latency=8372ms
- A06: success claims=9 latency=9609ms
- A07: success claims=9 latency=10116ms
- A08: success claims=9 latency=6974ms
- A09: success claims=9 latency=9398ms
- A10: success claims=10 latency=10132ms
- A11: success claims=9 latency=8108ms
- A12: success claims=12 latency=11879ms
- A13: error claims=0 latency=9685ms error=validation_failed
- A14: success claims=9 latency=6699ms
- A15: success claims=10 latency=8598ms
- A16: error claims=0 latency=9523ms error=validation_failed
- A17: success claims=9 latency=8604ms
- A18: success claims=8 latency=7680ms
- A19: success claims=10 latency=8704ms
- A20: success claims=8 latency=8398ms
- A21: success claims=11 latency=8260ms
- A22: success claims=10 latency=9003ms
- A23: success claims=7 latency=9231ms
- A24: error claims=0 latency=2520ms error=unknown_brand
- A25: success claims=9 latency=8646ms
- A26: success claims=8 latency=7424ms
- A27: success claims=12 latency=10682ms
- A28: success claims=9 latency=8133ms
- A29: success claims=7 latency=8374ms
- A30: success claims=11 latency=10879ms
- A31: success claims=9 latency=9212ms
- A32: success claims=8 latency=8349ms
- A33: success claims=10 latency=7632ms
- A34: success claims=8 latency=8187ms
- A35: success claims=10 latency=9767ms
- A36: error claims=0 latency=1293ms error=unknown_brand
- A37: success claims=9 latency=9843ms
- A38: success claims=9 latency=7976ms
- A39: error claims=0 latency=8131ms error=validation_failed
- A40: error claims=0 latency=10438ms error=validation_failed
