# ClaimLens-AI

ClaimLens extracts factual sentences from long speaker-product answers and
classifies them against the supplied catalogue. Python locates exact Unicode
source spans; a deterministic harness scores extraction and joint classification.
The project implements the Kasparro FSE Intern Pathway assessment.

## Start here

- Original requirements: [Assessment.docx](Assessment.docx).
- Frozen live evidence: [v1 development](reports/v1_development.json),
  [v2 development](reports/v2_development.json), [v1 check](reports/v1_check.json),
  [v2 check](reports/v2_check.json), and [historical comparison](reports/comparison.json).
- Byte-preservation manifest: [evidence/frozen_sha256.json](evidence/frozen_sha256.json).
- Historical test transcript: [test_output_safe.txt](test_output_safe.txt).
  It describes the earlier revision, not the current test count.
- Remediation verification: [evidence/remediation_verification.txt](evidence/remediation_verification.txt)
  records 125 passing tests, fresh installation/clone checks, and API startup.
- Read-only verification: `python -m scripts.verify_evidence`.
- The current runtime contains post-run correctness fixes. Its live extraction
  and classification quality has **not** been measured. Historical scores below
  describe the preserved runs, not a rerun of the current implementation.

No frontend or hosted deployment is required by the assessment.

## Setup

Developed and verified locally on Python 3.14.7 with PostgreSQL. Other Python
versions are not claimed as verified. Exact dependency versions are in
[requirements.txt](requirements.txt).

```bash
git clone https://github.com/rajatnagda45/ClaimLens-AI.git
cd ClaimLens-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY in .env only when running live inference.
```

For the application database, configure `DATABASE_URL` in `.env`, then:

```bash
createdb claimlens
python -m claimlens init-db
uvicorn claimlens.api:app --host 127.0.0.1 --port 8000
```

The default URL is `postgresql+psycopg://localhost:5432/claimlens`.
Use percent-encoded credentials when required by your database. Never commit
`.env`. The service initializes missing application tables at startup; this is
an intentional application setup action, not part of the default unit tests.
Interactive API documentation is at http://127.0.0.1:8000/docs.

Current defaults: `gpt-4o-mini`, temperature 0, at most three model turns,
90 seconds per provider request, and SDK retries disabled. A request timeout
is not a whole-answer wall-clock deadline. The four historical reports record
a **60-second** request timeout; that recorded value is authoritative.

## Verification without paid API calls

```bash
# Unit tests and in-memory API checks; PostgreSQL tests are explicitly skipped.
python -B -m pytest -q -p no:cacheprovider

# Full suite, including PostgreSQL integration.
python -B -m pytest -q -p no:cacheprovider --run-db

# Recompute historical metrics and check all frozen file hashes.
python -B -m scripts.verify_evidence
```

The integration suite requires a local PostgreSQL role with CREATEDB permission.
It uses the host/user/port from DATABASE_URL, creates a new
`claimlens_test_<32 hexadecimal UUID characters>` database for each fixture,
checks `current_database()` before DDL, and removes only the database it created.
It never reuses an existing test database. Remote hosts and URL query overrides
are rejected by these fixtures. Table deletion additionally requires the exact
disposable name and refuses the configured application database.

Tests use scripted providers and mocked SDK responses. They exercise error
handling and persistence, not real model quality. The suite includes malformed
submissions, duplicate/missing outcomes, source mismatch, tool-use bypasses,
Unicode, decimals, common abbreviations, repeated quotes, partial-run evidence,
concurrent report isolation, API statuses, and mixed-outcome database retrieval.

## Inputs and inference boundary

The original CSVs remain separate and unchanged:

| File | Records | Purpose |
|---|---:|---|
| answers.csv | 50 | A01–A40 development; A41–A50 check |
| brand_facts.csv | 132 | Sole factual authority: six models, three brands |
| development_gold.csv | 376 | Development evaluation only |

Versions: `long-brand-answers-50-v4`, `speaker-catalogue-v4`,
`sentence-spans-and-labels-v4`.

CSV loading uses UTF-8 and a proper parser with embedded-newline support.
The answer text is never normalized. Fact values are interpreted by declared
type, while dates and conditions remain attached to their rows.

The inference path receives the answer ID, primary brand, question, full answer,
and model-requested catalogue results. It does not load gold labels, spans,
explanations, or difficulty tags. The runner saves predictions before loading
development gold. Check runs never load gold labels; hashing the supplied gold
file for provenance is separate from loading its contents as labels.

## Agent and validation

`provider → agent → fact_tool → span_locator → validator`

The executor allows only `get_brand_facts` and the structured-output
`submit_claims` tool. It requires a successful fact lookup in an earlier turn
before submission, accepts cross-brand lookups, validates tool arguments,
and rejects duplicate submissions and mixed lookup/submission turns.
Unknown tools and brands are errors.

Model output requires all claim fields and forbids extras. Invalid output is
terminal in the current runtime; no invalid submission is repaired into success.
An explicit empty claims list remains valid after fact retrieval.
Provider failures produce errors with no accepted claims.

Python derives `claim_id = answer_id:start_char:end_char`, validates the exact
source slice and fact-ID existence, then emits nonoverlapping claims in source
order. Offsets count Unicode code points, not UTF-8 bytes. The locator rejects
fragments and merged sentences using a small English sentence heuristic that
handles decimals, common abbreviations, and closing quotes/brackets.
Repeated quotations must account for every source occurrence; otherwise they
fail as ambiguous. This is not a universal sentence parser.

The prompts require catalogue-only reasoning, date/market/model scope,
ordinary unit conversions, and compound precedence:
contradicted if any component is refuted; otherwise unknown if any component is
unresolved; supported only if all components are established.
Fact-ID existence does not prove evidence relevance. Correct reasoning remains
a model-quality question measured by the saved development results.

Answer instructions are delimited as untrusted text and cannot authorize
arbitrary tool execution. Finite fake-provider tests do not prove universal
prompt-injection resistance.

## Evaluation and historical results

For a complete scheduled development split:

```text
G = all gold claims; P = accepted predictions
M = exact span matches; C = exact matches with the correct label
Extraction: precision=M/P, recall=M/G, F1=2M/(P+G)
Joint:      precision=C/P, recall=C/G, F1=2C/(P+G)
Gold coverage=C/G
```

Matching is exact and one-to-one, with quote equality checked first.
Failed answers retain their gold in G. Extras increase P. Missing/duplicate
outcomes, malformed claims, and source mismatches are rejected.
Zero denominators use zero on a scheduled labeled split; an empty schedule has
null quality. Check quality always remains null. Current check reports retain
the independent prediction count P; the historical reports nulled it, so the
read-only verifier derives that count from the preserved predictions.

| Run | Answers | Success / error | P | M | C | Extraction F1 | Joint F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 development | 40 | 36 / 4 | 332 | 317 | 243 | 0.895480 | 0.686441 |
| v2 development | 40 | 34 / 6 | 313 | 296 | 223 | 0.859216 | 0.647315 |
| v1 check | 10 | 9 / 1 | 82 | null | null | null | null |
| v2 check | 10 | 9 / 1 | 87 | null | null | null | null |

Both development runs have G=376. Their recorded provider, model, inference
settings, and input hashes match. All 100 answer traces contain fact lookups.
These are local saved exchanges and metadata, not independent provider billing
attestations. Historical tool responses record brand/count rather than complete
fact bodies; current traces retain the redacted actual tool response.

### Targeted change and result

V1 states classification rules. V2 adds an explicit procedure: decompose a
sentence, find scoped evidence, mark components established/refuted/unresolved,
then apply label precedence. The frozen prompt files are preserved.

V1 has 59 missed claims, 15 unmatched predictions, and 74 matched wrong labels.
Its wrong-label confusion includes 42 unknown→supported and 17
contradicted→supported errors, motivating investigation of over-asserted support.

V2 has 80 missed claims, 17 unmatched predictions, and 73 matched wrong labels.
Unknown→supported falls to 34, but contradicted→supported rises to 20, and fewer
claims are matched. Raw error-count reduction therefore does not establish a
classification improvement on comparable claims. Overall joint F1 regresses by
**0.039126**, and extraction F1 regresses by **0.036264**.

Comparing exact correctly labeled gold spans, v2 gains 18 and loses 38 relative
to v1. There is one saved run per version; these artifacts do not establish
statistical stability or a causal “attention budget” explanation.

The prompt change is identifiable, but the precise pre-check freeze/tuning
history has not been independently established. No check-set labels were
available or used in this remediation. Prior final-check input exposure/tuning
remains **unverified pending the owner's confirmation**.

A separate extraction/labeling pass is a possible next experiment, with unknown
benefit and additional request-budget implications. It was not run.

## API and persistence

| Endpoint | Behavior |
|---|---|
| POST /classify | Validated input and v1/v2 selection; 200 success, 422 bad input, 404 unknown primary brand, 502 provider/output failure, 504 timeout |
| POST /evaluation-runs | Fixed split; 201 after complete processing, including answer errors; 503 with run reference for infrastructure failure |
| GET /evaluation-runs/{id} | Stable answer/span order, metadata, outcomes and claims; 404 missing run, 422 invalid UUID |

`runs`, `answer_results`, and `claim_results` use PostgreSQL, parameterized
SQLAlchemy queries, the required foreign keys, unique run/answer keys, and unique
run/answer/span keys. Current runs checkpoint before gold joins and record
failed/interrupted status where storage remains available. Abrupt process kills
or total storage failure cannot guarantee a final checkpoint.

Reports for new runs are saved under `reports/runs/<run_id>.json` and `.md`.
The historical four-run files and comparison are never overwritten by the runner.
Metadata captures input hashes, source revision when available, prompt hash,
runtime hash, settings, and scheduled IDs. A runtime hash distinguishes dirty
working-tree code from a commit reference.

The earlier application database rows were lost to an old destructive fixture;
the repository already documented that loss. They have not been recreated or
represented as newly verified live persistence. During remediation, the
application tables were absent and left untouched. Current persistence is
demonstrated using synthetic results in disposable PostgreSQL databases.

## New live experiments

These commands make paid provider calls. They are not needed to verify the
historical evidence:

```bash
python -m claimlens classify --answer-id A01 --version v1
python -m scripts.run_all v1 development --confirm-live
```

For a new improvement study, inspect development failures, make one separately
named change, rerun development with fixed settings, and freeze before checks.
The batch form `python -m scripts.run_all --confirm-live` runs the existing
versions sequentially; it does not itself prove an inspection/freeze protocol.
Do not tune using private check labels. No paid experiments were run as part of
the current remediation.

## Security and remaining limitations

Public provider/setup errors use fixed messages, and database connection URLs
are omitted from new metadata. Trace redaction removes configured credentials
and common key/connection-string patterns. It is best effort, not a guarantee
for every arbitrary secret format; review any evidence before sharing.

The current runtime changes include strict schemas, mandatory prior fact use,
typed facts, revised sentence boundaries, terminal validation failures, disabled
SDK retries, stronger schedule checks, and checkpointed persistence.
They can change accepted outputs. Historical predictions and metrics were
neither filtered nor rewritten: an offline boundary check identifies 2/6
historical development quotations and 3/0 check quotations (v1/v2 respectively)
that the current boundary rules would reject. No quality benefit is claimed.

Authentication, hosted deployment, durable recovery, and request idempotency are
outside the assessment scope. Input-size limits and a general-language sentence
parser are not implemented. Model outputs can still be wrong despite valid
spans and existing fact IDs.

## Ownership, time, and assistance

AI coding assistants used through CLI tooling assisted implementation, testing,
and subsequent review/remediation. Automated checks are identified by command
and actual outcomes; this documentation does not claim that the owner personally
performed each check.

The owner reports approximately **2.5–3 hours** of focused original implementation
and evaluation, covering the agent, exact-span harness, v1 baseline, v2 experiment,
four live runs, and FastAPI/PostgreSQL integration with tests. Additional time was
spent afterward on an independent audit and correctness fixes; that later effort
is separate from the original estimate. Frozen prompts and saved live-run reports
were not modified by these fixes.

The owner confirms **no prompt tuning after inspecting final-check answers or
outcomes**: the v2 change was based solely on the v1 development failure breakdown
(A01–A40), particularly `unknown → supported` and `contradicted → supported`
errors. The prompts were frozen before final-check predictions, and the intended
v1/v2 difference was the evidence-first classification procedure. No check-set
gold labels or outcomes were used to tune prompts or select a version.

These time and workflow statements are owner-provided attestations, not facts
independently established by the saved artifacts. Final-check inputs are present
in `answers.csv`, so the benchmark was not fully blind. Check runs retain
predictions and operational results, with gold-dependent quality metrics null.
