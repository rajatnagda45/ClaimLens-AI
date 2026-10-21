# Run report: v1 / check

- run_id: `18c93a49-9be3-4c37-b8a3-e8d5a1be3621`
- provider: **OpenAIProvider** (live=True)  model: `gpt-4o-mini`
- prompt: `extract-classify-v1`  evaluator: `sentence-spans-and-labels-v4`
- fact: `speaker-catalogue-v4`  dataset: `long-brand-answers-50-v4`
- timestamp: 2026-10-21T09:00:45.833585+00:00
- source_hashes: {"answers.csv": "fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724", "brand_facts.csv": "2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2", "development_gold.csv": "afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850"}

## Metrics
_Quality metrics are null for this split (check split)._

- answers: success=9 empty=0 error=1

## Per-answer status

- A41: success claims=7 latency=8684ms
- A42: success claims=10 latency=9591ms
- A43: success claims=10 latency=9290ms
- A44: success claims=10 latency=10026ms
- A45: success claims=9 latency=15811ms
- A46: success claims=11 latency=9114ms
- A47: success claims=8 latency=10751ms
- A48: success claims=10 latency=10855ms
- A49: success claims=7 latency=12905ms
- A50: error claims=0 latency=20480ms error=validation_failed
