# Run report: v2 / check

- run_id: `f81863cc-f9d2-4d70-8359-25eb3ba57de7`
- provider: **OpenAIProvider** (live=True)  model: `gpt-4o-mini`
- prompt: `extract-classify-v2`  evaluator: `sentence-spans-and-labels-v4`
- fact: `speaker-catalogue-v4`  dataset: `long-brand-answers-50-v4`
- timestamp: 2026-10-21T09:02:20.655844+00:00
- source_hashes: {"answers.csv": "fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724", "brand_facts.csv": "2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2", "development_gold.csv": "afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850"}

## Metrics
_Quality metrics are null for this split (check split)._

- answers: success=9 empty=0 error=1

## Per-answer status

- A41: success claims=9 latency=8659ms
- A42: success claims=10 latency=8662ms
- A43: success claims=10 latency=9792ms
- A44: success claims=10 latency=10157ms
- A45: success claims=8 latency=9483ms
- A46: success claims=13 latency=11633ms
- A47: success claims=8 latency=8130ms
- A48: success claims=10 latency=9728ms
- A49: error claims=0 latency=9565ms error=validation_failed
- A50: success claims=9 latency=8971ms
