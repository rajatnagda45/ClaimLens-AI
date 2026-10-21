# Run report: v1 / development

- run_id: `44085fcb-10ec-4b14-911c-3a905ad04adc`
- provider: **OpenAIProvider** (live=True)  model: `gpt-4o-mini`
- prompt: `extract-classify-v1`  evaluator: `sentence-spans-and-labels-v4`
- fact: `speaker-catalogue-v4`  dataset: `long-brand-answers-50-v4`
- timestamp: 2026-10-21T08:53:05.488965+00:00
- source_hashes: {"answers.csv": "fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724", "brand_facts.csv": "2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2", "development_gold.csv": "afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850"}

## Metrics
- G=376 P=332 M=317 C=243
- extraction: P=0.955 R=0.843 **F1=0.895**
- joint: P=0.732 R=0.646 **F1=0.686** (primary)
- gold coverage C/G = 0.646

- answers: success=36 empty=0 error=4

## Per-answer status

- A01: success claims=9 latency=9811ms
- A02: success claims=10 latency=11507ms
- A03: error claims=0 latency=11711ms error=validation_failed
- A04: success claims=11 latency=11739ms
- A05: success claims=8 latency=8079ms
- A06: success claims=9 latency=10679ms
- A07: success claims=8 latency=8498ms
- A08: success claims=9 latency=9346ms
- A09: success claims=9 latency=9990ms
- A10: success claims=9 latency=68263ms
- A11: success claims=10 latency=11267ms
- A12: success claims=10 latency=10188ms
- A13: success claims=10 latency=11723ms
- A14: success claims=9 latency=12712ms
- A15: success claims=10 latency=11492ms
- A16: error claims=0 latency=16341ms error=validation_failed
- A17: success claims=7 latency=10121ms
- A18: success claims=8 latency=9214ms
- A19: success claims=10 latency=9968ms
- A20: success claims=8 latency=9858ms
- A21: success claims=10 latency=9898ms
- A22: success claims=10 latency=71089ms
- A23: success claims=7 latency=6959ms
- A24: success claims=10 latency=9283ms
- A25: success claims=10 latency=11268ms
- A26: success claims=8 latency=8242ms
- A27: success claims=12 latency=9870ms
- A28: success claims=9 latency=10458ms
- A29: success claims=8 latency=8044ms
- A30: success claims=11 latency=9253ms
- A31: success claims=8 latency=6335ms
- A32: success claims=8 latency=7384ms
- A33: success claims=10 latency=8196ms
- A34: success claims=8 latency=9212ms
- A35: success claims=10 latency=9586ms
- A36: error claims=0 latency=2421ms error=unknown_brand
- A37: success claims=10 latency=9425ms
- A38: success claims=10 latency=9194ms
- A39: success claims=9 latency=9413ms
- A40: error claims=0 latency=34213ms error=validation_failed
