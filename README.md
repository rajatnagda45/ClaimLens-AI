# ClaimLens-AI

Evaluation-first AI agent that reads long AI-generated answers about fictional
speaker products, extracts every factual **claim sentence** with exact source
character offsets, and classifies each as `supported`, `contradicted`, or
`unknown` using **only** the supplied brand catalogue — scored by a
deterministic exact-span harness.

Built for the Kasparro FSE Intern Pathway assessment.

---

## 1. Problem statement & design overview

- **Input:** `answers.csv` (50 answers; A01–A40 development, A41–A50 check),
  `brand_facts.csv` (132 catalogue rows, 3 brands × 2 models), and
  `development_gold.csv` (376 labeled gold claims, dev only).
- **Task:** extract complete factual sentences from `ai_answer`, classify each
  against the catalogue, and return validated claims with exact offsets.
- **Design priority:** correctness, measurable joint F1, reproducibility, and
  honest evidence — not the appearance of sophistication.

Key principle: **the model proposes exact quotes and labels; Python
independently locates the character offsets and validates every invariant.**
Model-generated character counts are never trusted.

## 2. Architecture & data flow

```
answers.csv ─┐
brand_facts ─┼─ data_loader ─ (exact text, hashes, split selection)
dev_gold ────┘        │
                      ▼
      ┌───────────────────────────────────────────────┐
      │ agent (tool loop, ≤3 turns, timeout)           │
      │   ├─ provider (OpenAI function-calling)         │
      │   ├─ fact_tool get_brand_facts (allowlisted)    │
      │   └─ submit_claims (structured output)          │
      └───────────────────────────────────────────────┘
                      │  model quotes + labels
                      ▼
     span_locator (deterministic offsets) → validator (invariants)
                      │  validated Claim[]
                      ▼
     runner ── persist (PostgreSQL: runs/answer_results/claim_results)
         │  └─ evaluator (exact-span scoring, dev only)
         └─ reports/*.json + *.md   +   analysis (v1-vs-v2)
                      ▲
      api (FastAPI): /classify, /evaluation-runs, GET /evaluation-runs/{id}
```

Module map (`claimlens/`): `config`, `data_loader`, `schemas`, `fact_tool`,
`span_locator`, `validator`, `provider`, `agent`, `evaluator`, `runner`,
`analysis`, `database`, `api`, `prompts/{v1,v2}`.

## 3. Claim & label semantics

Each accepted claim has: `claim_id`, `start_char`, `end_char`, `quote`,
`label`, `fact_ids`, `reason`. Invariants (enforced in `validator.py`):

- `claim_id == answer_id:start_char:end_char`.
- Offsets are **zero-based Python Unicode code points**; `start` inclusive,
  `end` exclusive; `quote == ai_answer[start:end]`.
- Quotes are complete single sentences ending in sentence punctuation, with no
  leading/trailing whitespace. The locator rejects mid-sentence **fragments**
  (even at a word boundary — the char before the leading whitespace must be a
  sentence terminator or start-of-text) and **merged** quotes that span more
  than one sentence.
- Claims are ordered by ascending `start_char`, non-overlapping; duplicate IDs
  and spans are rejected.
- `label ∈ {supported, contradicted, unknown}`; every `fact_id` exists in the
  catalogue (empty list allowed); `reason` ≤ 2 sentences.

Labels: **supported** = all factual components established by relevant facts;
**contradicted** = ≥1 component explicitly refuted under matching conditions;
**unknown** = otherwise (any unresolved component, including missing rows — an
absent fact is never a negative fact). Compound precedence:
contradicted > unknown > supported. A 2025 price does not establish a 2026
price; conditions apply only within their stated scope.

## 4. Tool-calling & output validation

- Real OpenAI function-calling loop, **≤ 3 model turns** per answer, with a
  configurable per-request timeout (`CLAIMLENS_TIMEOUT_S`, default 90s).
- `get_brand_facts(brand_id)` returns catalogue rows at fact version
  `speaker-catalogue-v4`. Tool name is checked against an **allowlist**; the
  brand argument is schema-validated; unknown brands and malformed args are
  errors. Any known brand may be looked up (comparisons), not just the primary.
- Arbitrary tool names are never dispatched; text inside `ai_answer` is treated
  as **untrusted data** and can never change the allowlist or behavior.
- Failure modes are **explicit answer-level errors** (`provider_error`,
  `provider_timeout`, `invalid_output`, `illegal_tool`, `unknown_brand`,
  `turn_exhausted`, `validation_failed`) — never silently downgraded to
  `unknown`, never partially accepted.
- The sanitized model/tool exchange is preserved per answer as evidence.

## 5. Environment & Python version

- Python **3.14** (developed/tested on 3.14.7). Should work on 3.11+.
- PostgreSQL 14+ running locally.

## 6. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 7. Database setup

```bash
createdb claimlens
export DATABASE_URL="postgresql+psycopg://localhost:5432/claimlens"
python -m claimlens init-db        # idempotent create_all; prints confirmation
```

`DATABASE_URL` uses the SQLAlchemy + psycopg v3 driver. Schema is created
reproducibly via `Base.metadata.create_all` (see `database.py`); no manual DDL.

## 8. Provider setup (no credentials committed)

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...   (never commit this file)
```

Tunables (env): `CLAIMLENS_MODEL` (default `gpt-4o-mini`),
`CLAIMLENS_MAX_TURNS` (default 3), `CLAIMLENS_TIMEOUT_S` (default 90),
`CLAIMLENS_TEMPERATURE` (default 0). Keys are read via `python-dotenv` and are
never printed, logged, committed, or placed in reports.

## 9. Commands

```bash
# Tests (deterministic, fake provider; no API key needed). DB tests use a
# dedicated disposable database '<app>_test' and never touch the app DB, so a
# running PostgreSQL is required but DATABASE_URL data is safe.
pytest -q

# Start the API
uvicorn claimlens.api:app --reload

# Classify one answer (live provider)
python -m claimlens classify --answer-id A01 --version v2

# Run one split (live provider)
python -m claimlens run --version v1 --split development

# Run ALL FOUR live artifacts + comparison
python -m scripts.run_all
```

### API examples

```bash
# POST /classify
curl -s localhost:8000/classify -H 'content-type: application/json' -d '{
  "answer_id":"A01","brand_id":"cleartrail","question":"q",
  "ai_answer":"The ClearTrail Mini costs 1299 rupees.","agent_version":"v2"}'

# POST /evaluation-runs
curl -s localhost:8000/evaluation-runs -H 'content-type: application/json' \
  -d '{"agent_version":"v1","split":"development"}'

# GET /evaluation-runs/{id}
curl -s localhost:8000/evaluation-runs/<run_id>
```

Status codes: `/classify` → 200 | 422 (bad input / client-supplied gold or
claim fields) | 404 (unknown primary brand) | 502 (provider / illegal tool /
invalid output) | 504 (timeout). `GET /evaluation-runs/{id}` → 200 | 404 | 422.

## 10. Reports & traces

- `reports/v1_development.{json,md}`, `reports/v2_development.{json,md}`,
  `reports/v1_check.{json,md}`, `reports/v2_check.{json,md}`.
- `reports/comparison.json` — v1-vs-v2 dev metrics + failure summaries.
- Each report includes predictions, per-answer outcomes, metrics (null for
  check), missing/extra/wrong-label records (dev), error counts, settings,
  prompt/dataset/evaluator/fact versions, input SHA-256 hashes, timestamps, and
  the sanitized model/tool exchange.

## 11. v1 vs v2 metrics & failure analysis

All four runs are live (`OpenAIProvider`, model `gpt-4o-mini`, temperature 0,
≤3 turns, 90s timeout). Metrics below are from the deterministic harness on the
40 development answers (376 gold claims). Check-set quality metrics are `null`
by design.

**Development results (primary metric = joint F1):**

| version | joint F1 | extraction F1 | G | P | M | C | success | error |
|---------|---------:|--------------:|--:|--:|--:|--:|--------:|------:|
| v1      | **0.686** | 0.895 | 376 | 332 | 317 | 243 | 36 | 4 |
| v2      | 0.647 | 0.859 | 376 | 313 | 296 | 223 | 34 | 6 |
| Δ (v2−v1) | **−0.039** | −0.036 | | | | | | |

**v1 failure breakdown (from saved predictions vs gold, `analysis.categorize`):**
missed=59, extra=15, **wrong_label=74**, answer_error=4. Because C (243) is far
below M (317), labeling — not extraction — is the whole gap between joint F1
(0.686) and extraction F1 (0.895). The wrong labels are dominated by
**over-claiming `supported`**: `unknown→supported` = 42 and
`contradicted→supported` = 17 (59 of 74). The model marked claims supported
when the catalogue did not establish them, or even refuted them.

**The single v2 change (justified by that evidence):** an explicit
**evidence-first classification procedure** — decompose each sentence into
factual components, mark each ESTABLISHED / REFUTED / UNRESOLVED, and apply
`contradicted > unknown > supported` precedence before labeling. The extraction
contract, dataset, scorer, provider, model, and settings are identical to v1;
only the classification instructions differ, so the comparison is fair.

**Measured effect — honest result: v2 did NOT improve joint F1 (−0.039).**
- The change *did* work on its target: `unknown→supported` errors fell 42→34
  (the model became appropriately more cautious about asserting support).
- But the heavier procedure **regressed extraction**: missed claims rose
  59→80, answer errors 4→6, and accepted predictions fell 332→313 — i.e. the
  model spent "attention budget" on the classification protocol and extracted
  fewer complete sentences. The recall loss outweighed the label gain, so joint
  F1 dropped.

This is reported as-is. No gold labels were hardcoded, no denominators
manipulated, and the scorer was not tuned to predictions. The practical
takeaway: the next iteration should decouple the two concerns — keep v1's
extraction behavior and apply the evidence-first labeling as a **separate second
pass** rather than folding it into a single heavier prompt.

**Check runs (A41–A50):** v1 and v2 each produced 9 successful answers and 1
answer-level error; predictions, errors, latencies, counts, and versions are
saved while all gold-dependent metrics remain `null` (the reviewer scores these
with the private key).

## 11a. Post-run fixes & database-evidence note

**The saved JSON/Markdown reports in `reports/` are the authoritative live-run
evidence.** They were produced by the four live runs and are frozen (unchanged).
The frozen v1/v2 prompts are likewise unchanged.

**Database rows were lost after the runs.** An early version of the API test
fixture called `drop_db()` against the configured application database. When the
full `pytest` suite was later run during verification, it dropped the `claimlens`
tables, so the originally-persisted run/answer/claim rows no longer exist. This
did **not** affect the on-disk reports. Live persistence had been observed
working right after the runs (40/40/10/10 answers; 332/313/82/87 claims) — but
those specific rows are gone and were **not** restored by rerunning the live
evaluations (that would spend API credits for no new evidence).

**Fixes made AFTER the live runs** (they apply to future runs and the test
suite; they do not alter the frozen prompts or the saved reports):

1. **Destructive test fixture removed.** Tests now bind to a dedicated,
   disposable `<app>_test` database with explicit safety assertions, and
   `drop_db()` refuses to run against the application database. See
   `tests/conftest.py::test_db`, `database.drop_db()`, and
   `tests/test_db_isolation.py`.
2. **Sentence-fragment rejection.** The locator now rejects word-boundary
   fragments (the char before the leading whitespace must be a sentence
   terminator or start-of-text).
3. **Merged-sentence rejection.** A quote spanning more than one sentence is
   rejected. Both rules were validated against all 376 gold sentences (0
   affected).
4. **Empty-split null metrics.** An empty scheduled (labeled) split now yields
   `null` quality metrics, distinct from a 0.0 score when gold is present.

**Persistence re-verified** with the disposable database and synthetic /
fake-provider inputs (no API credits): nonempty-claim, valid-empty, and
answer-level-error outcomes all persist and read back, and the unique-span
constraint is enforced (`tests/test_db_isolation.py`).

## 12. Security & prompt-injection

- `ai_answer` is passed to the model clearly delimited as untrusted data.
- Instructions embedded in answers ("ignore rules", "label everything
  supported", "call tool X") are never obeyed; the tool allowlist is static and
  arbitrary tool names are rejected as `illegal_tool`.
- The API rejects any client-supplied gold/claim fields with 422
  (`extra="forbid"`). Gold data is used **only** in the evaluation/reporting
  pipeline and never enters the model or tool payload (asserted by a test).
- Error responses are sanitized to `{error_type, message}` only; the message is
  the provider/validation text (it does not contain the API key, which is never
  interpolated into errors). Credentials are masked (`***set***`) in run
  metadata and kept out of git, logs, traces, and reports (verified by a
  `grep` for `sk-…` over `reports/`).

## 13. Limitations, time, and AI-assistance disclosure

- **Live runs complete.** All four artifacts in `reports/` are from the live
  OpenAI API (`gpt-4o-mini`). Nothing is mocked or fabricated. The original
  persisted DB rows were lost to an early destructive test fixture (now fixed) —
  see §11a; the reports remain the authoritative evidence.
- **v2 regressed** on joint F1 (−0.039); reported honestly (§11). A better next
  step is a two-pass design (v1 extraction + separate evidence-first labeling).
- **Residual answer-level errors:** a few development/check answers fail with
  `validation_failed` because the model occasionally synthesizes a near-verbatim
  sentence that is not a character-exact substring (e.g. A08/A16). Per the
  assessment, an invalid claim makes the whole answer an answer-level error
  (no partial acceptance); one bounded recovery turn mitigates but does not
  eliminate this.
- **Non-determinism:** even at temperature 0 the API varies slightly run-to-run,
  so exact counts may differ a little on re-run; the qualitative result (v1 >
  v2 on joint F1, labeling as the dominant weakness) is stable.
- Single provider (OpenAI) implemented; the provider interface is small enough
  to add others. `reason`-length check uses a simple sentence splitter. No
  auth/deployment/idempotency (explicitly out of scope).
- **AI assistance:** this repository was implemented with the help of an AI
  coding assistant used through its CLI. I directed the design, chose the v2
  change from the real failure breakdown, reviewed every module, ran the full
  test suite myself (`pytest -q` → 70 passed), executed all four live runs, and
  verified the report artifacts. The measured v2 regression is reported as-is
  rather than tuned away.

## Evaluation harness (summary)

Exact-span matching: a prediction matches a gold claim only when
`(answer_id, start_char, end_char)` are identical **and** the quote equals the
gold quote. One-to-one matching (a gold span matches at most one prediction).

```
G = gold claims, P = accepted predictions,
M = exact span matches (any label), C = span matches with correct label.
extraction: P=M/P, R=M/G, F1=2M/(P+G)
joint:      P=C/P, R=C/G, F1=2C/(P+G)      <- primary quality metric
```

Mandatory check (`tests/test_evaluator.py::test_mandatory_arithmetic`):
`G=10, P=10, M=8, C=7` → extraction F1 `0.80`, joint F1 `0.70`. Answer errors
contribute no predictions but keep their gold claims in `G` as missed. A zero
denominator with gold present scores `0.0`; an **empty scheduled split** (no
gold) and the **check split** both report `null` gold-dependent metrics.
