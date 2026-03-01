# Hackathon Status

This file is the running English handoff note for the hackathon. It is intended
to be updated as the project evolves, so the current state, decisions, and next
steps remain easy to present.

## Goal

The long-term product goal is:

- accept a free-form French succession description from a user
- transform it into a structured JSON object matching a target schema

For this hackathon, the working “master schema” is already defined and lives at:

- `../w5/glinerExtract/schema/schema.full.json`

We currently represent training targets as **TOON** (sparse, schema-driven),
because it is cheaper to serialize than JSON and reduces “formatting failure”
noise during fine-tuning.

## Current Phase

The current phase is building **training pairs**.

What we are doing now:

- collecting real succession case descriptions already exercised by E2E tests
- generating many additional synthetic cases with controlled diversity
- generating a **sparse structured target** for each case (server-side)
- preparing a training-ready pool of `{ case_text, target_toon }` pairs

What we are **not** doing yet:

- training the final fine-tuned model
- exporting a fine-tuned model to ONNX

## Why This Order

We intentionally separated the work into phases:

1. Import all real E2E cases as seeds.
2. Define/lock a master schema and a single sparse target representation (TOON).
3. Generate many additional **paired** synthetic samples (target-first).
4. Fine-tune the model on `description -> structured target`.
5. Optionally export for browser inference later (ONNX / Transformers.js).

This keeps the current work useful even before the final schema is available.

## Model Direction

The current preferred model direction is:

- base checkpoint for future tuning: `mistralai/Ministral-3-3B-Instruct-2512`
- training style: LoRA fine-tuning
- final deployment possibility: browser inference later, likely through ONNX / Transformers.js

Important architectural decision:

- ONNX is a deployment format, not the training starting point
- fine-tuning should happen first on the normal trainable checkpoint
- ONNX export, if needed, should happen after training

## Dataset Assets Already Built

### Seed Corpus Imported From E2E

We imported all succession descriptions currently passing through the E2E lane of
`../w5`.

Current imported seed volume:

- `161` unique descriptions total
- `94` from `succession97`
- `17` from legacy `cases/succession`
- `22` from `succession-search-bar`
- `28` from inline E2E prompts

Main generated files:

- `data/succession_e2e/e2e_cases.jsonl`
- `data/succession_e2e/e2e_cases_train.jsonl`
- `data/succession_e2e/e2e_cases_train_mistral.jsonl`
- `data/succession_e2e/manifest.json`

The `*_mistral.jsonl` file is the strictest current format for future Mistral
fine-tuning (`messages` only).

### Instruction Server

A local instruction server was added to drive synthetic case generation.

It does the following:

- issues one generation instruction at a time
- balances multiple diversity dimensions
- generates a schema-driven sparse **target TOON** (server-side)
- persists issued instructions and submitted case texts
- updates progress and quota summaries on disk
- exports a merged training file as new cases are submitted

Main files:

- `src/ministral_ft/case_instruction_server.py`
- `scripts/run_case_instruction_server.sh`
- `data/case_instruction_server/config.json`
- `data/case_instruction_server/summary.md`
- `data/case_instruction_server/full_training_cases_mistral.jsonl`

Key pipeline design:

- The server generates `target_toon` first.
- Agents only generate `case_text` from the provided TOON and style constraints.
- Targets are **sparse only**: no `null`, no empty objects/lists, only relevant branches.

## Method Pivot: Target-First (Deterministic) + Text-From-Target (Creative)

Early idea (initial baseline):

- generate a free-form succession description first (LLM creativity)
- then generate / fill the target structured JSON from that text (LLM “labeling”)

What we switched to (current production path):

- generate the structured target **first**, server-side:
  - schema-driven sparse JSON payload (deterministic constraints, enums, types)
  - encode to **TOON** for a cheaper and more robust serialized target format
- then ask LLM agents to generate the **input text** from that target:
  - natural French, varied personas/voices/noise
  - strict guardrails to avoid leaking schema tokens into the narrative

Why this pivot matters:

- We need the target (`JSON` / `TOON`) to be **deterministic, schema-valid, and consistent**.
- We want the text to be **creative, diverse, and realistic**, which is where LLMs shine.
- This flips the usual direction: we synthesize the **output label** first, then generate
  the **input** that matches it. It reduces hallucinated labels and formatting failures.

In other words: we generate the dataset by producing the *output* and asking agents to
produce the matching *input*.

Quality guardrails added after initial samples:

- removed placeholder strings like “Clause ou élément mentionné” / “Information fournie”
- fixed regime/statut coherence (no “PACSE + participation”, etc.)
- reject submissions that leak schema keys in `snake_case` into the free-form text
- reject submissions that leak enum codes like `MAJUSCULES_AVEC_UNDERSCORE` into the free-form text
- harden prompts so agents translate enum codes into natural French (no underscores)

## How Synthetic Cases Are Generated (Code-Accurate)

This section describes the exact generation loop implemented by the instruction
server.

Source of truth:

- `src/ministral_ft/case_instruction_server.py`

### 1) Issue an instruction (`/next-instruction`)

The server picks a set of diversity dimensions (persona, voice, noise, numeric
density, date precision, complexity, topics, etc.) by *balancing toward target
quotas*:

- selection uses `_pick_underrepresented(...)` so underrepresented buckets are
  preferentially chosen
- recent “signatures” are avoided to reduce short-range repetition
- some topic/persona combinations are blocked (example: PACS/concubin persona
  excludes some matrimonial-regime topics) to reduce forced contradictions

### 2) Generate the target (server-side, schema-driven)

The server generates a **sparse** structured payload first, then encodes it as
TOON.

Key steps:

1. Load the master schema (`../w5/glinerExtract/schema/schema.full.json`) and
   build an index of allowed nodes and leaf specs (paths + enums + expected
   scalar types).
2. Choose a set of leaf paths to populate:
   - mandatory identity paths (defunt name, marital status, death date, etc.)
   - additional mandatory paths derived from persona/topic
   - then probabilistic selection of other leaves under topic-related schema
     prefixes (probability depends on complexity)
   - optional extra prefixes are sometimes sampled to add cross-topic diversity
3. Fill leaf values with typed generators:
   - enums are sampled from allowed values
   - numbers/dates use key-aware heuristics (age, amounts, ratios, durations)
   - names use `faker` if available, otherwise synthetic name lists
   - string fallbacks are always concrete (cities / asset labels), never generic
     placeholders
4. Repair / harmonize business invariants:
   - marital status ↔ partner presence/link type
   - drop matrimonial regime blocks when not in a marriage context
   - age/date consistency across person blocks
   - insurance contract insured name matches the defunt
   - donation donor != beneficiary
   - ensure topic blocks exist (e.g. insurance topic -> at least one contract;
     Dutreil topic -> enterprise block present; etc.)
5. Validate and retry if needed (up to 50 attempts):
   - sparse-only validation (no `null`, no empty objects/lists, no empty strings)
   - schema-path + type + enum validation against the master schema index
   - business coherence validation (core sanity checks + topic-specific checks)
   - topic alignment validation: the declared topics must actually be present in
     the payload (required leaf paths / prefixes)
6. Encode the JSON payload to TOON using the official TOON CLI (`npx
   @toon-format/cli --encode`), then decode-validate to ensure the TOON syntax
   round-trips.

Important note:

- even for “hard negative” cases, the **target is always schema-valid and
  coherent**. “Hard negative” is implemented mostly as text-level ambiguity
  (agents are instructed to include a realistic ambiguity/contradiction), not by
  producing invalid targets.

### 3) Ask LLM agents to generate the text (input-from-target)

The server returns:

- `instruction_id`
- `target_toon`
- a prompt augmented with:
  - the TOON block itself
  - explicit rules: every TOON fact must appear in the narrative, but reformulated
    in natural French
  - strict “no-leakage” rules: no `snake_case`, no `MAJUSCULES_AVEC_UNDERSCORE`,
    no JSON/TOON in the output, no invented facts

Agents then write one free-form French description matching the target and
submit it back.

### 4) Submit and validate (`/submit-case`)

On submission the server enforces:

- instruction existence + “not already submitted”
- target TOON decoding validation (rejects JSON-looking payloads; TOON required)
- **name coverage**: every value under `nom` / `*_nom` / `*_noms` in the decoded
  target must appear in the free-form text (normalized matching with partial
  last-name fallback)
- hard rejection if the narrative contains:
  - `snake_case` keys (regex-based)
  - enum-like tokens in `ALL_CAPS_WITH_UNDERSCORES` (regex-based)
- similarity warnings (Jaccard) are computed to detect exact duplicates / near
  duplicates, but do not hard-block the sample

If valid, the server stores:

- the raw case submission (with dimensions + validation metadata)
- per-instruction artifacts (issued + submitted JSON files)
- a merged training export file in Mistral SFT format (`messages`)

## Guardrails We Had To Add (And Why)

This project required several rounds of manual QA to find the right integrity
constraints. The key “hard” guardrails that emerged are:

1. Target-first generation
- Text-first generation caused hallucinated labels and non-deterministic targets.
- Target-first enforces determinism and schema validity; LLM creativity is used
  only for linguistic rendering.

2. Sparse-only targets (omit instead of `null`)
- Mixing “missing fields” as sometimes `null` and sometimes “absent” created
  inconsistent supervision signals.
- We standardized on sparse-only: omit missing branches entirely.

3. Persona/topic coherence constraints
- Without constraints, some persona/topic pairs forced contradictions (e.g. a
  PACS persona paired with heavy matrimonial-regime liquidation).
- We added mandatory leaf paths per persona and blocked a few topic choices for
  certain personas.

4. Schema-token leakage rejection
- Early samples showed LLMs sometimes copied enum tokens (underscored caps) or
  schema keys (snake_case) into the narrative.
- We made this a hard rejection server-side and reinforced it in prompts.

5. Business integrity repair + validation
- The generator must produce coherent legal facts (within the simplified model).
- We added a dedicated repair pass plus a dedicated business-coherence validator
  to catch issues like inconsistent marital status/partner link types, invalid
  ages vs dates, etc.

## Manual QA / Smoke Tests We Repeated

While iterating, the recurring manual checks were:

- sample the last generated submissions and read them (text + TOON) to spot
  contradictions or schema leaks
- scan the exported JSONL for forbidden patterns (`snake_case`, enum tokens)
- ensure line counts match expected “submitted” counts
- verify the server stays stable during long runs (restartability, port reuse)
- confirm the training export is always valid JSONL and follows the `messages`
  format required by Mistral fine-tuning

## Current Production-Oriented Quota Profile (v6)

The active synthetic generation profile is currently tuned for future structured
extraction quality, not for pure linguistic chaos.

Global target:

- total future training cases: `5000`
- existing seed cases: `161`
- synthetic generation target: `4839`

### Style and Difficulty

- `20%` simple
- `40%` intermediate
- `24%` complex
- `16%` hard negatives

### Cleanliness

- `42%` clean
- `22%` light mistakes
- `17%` mistakes plus abbreviations
- `16%` ambiguous
- `3%` very messy

### Numeric Density

- `6%` no amount
- `26%` one amount
- `38%` several amounts
- `30%` amounts plus dates

### Time Precision

- `15%` no date required
- `20%` approximate time references
- `65%` at least one exact date

### Length

- `18%` short
- `42%` medium
- `32%` long
- `8%` very long

This profile is intentionally biased toward cleaner, more extractable inputs:

- fewer fully messy samples
- fewer amount-free cases
- more exact dates
- fewer hard negatives than earlier profiles

## Why These Quotas (And How We Enforce Them)

The quota profile is designed to maximize downstream extraction reliability
while still exposing the model to realistic noise and failure modes.

### Why the ratios look like this

Hard negatives (`16%`) exist to reduce false positives and over-extraction. We
kept them below ~20% because:

- too many hard negatives can teach the model to be overly conservative
- we still want the majority of training samples to be “actionable” and richly
  extractable
- in a fixed-schema extractor, precision failures are costly, but recall still
  matters

We skew hard negatives toward “soft” realistic traps:

- intensity: `80% soft`, `20% hard`
- modes: missing key info / unclear death / contradictions / out-of-scope

Cleanliness is biased toward parseable text:

- `42%` clean, `22%` light mistakes
- only `3%` very messy (because extremely noisy text tends to add label noise)

Numeric content is biased toward information-rich cases:

- `68%` cases contain multiple amounts or amounts+dates
- `6%` cases contain no amount (kept as a minority so the model doesn’t assume
  “every case has money”)

Dates are mostly exact because they are critical for taxation / timing branches:

- `65%` at least one exact date
- `20%` approximate
- `15%` no required dates

Lengths are biased toward medium/long because real legal intakes are rarely one
sentence:

- `74%` medium or long

### How the code enforces quotas

Quotas are **not** sampled as naive random weights. Instead, each dimension is
balanced toward its target share over time:

- targets live as `{ bucket -> share }` dictionaries in
  `src/ministral_ft/case_instruction_server.py` (e.g. `COMPLEXITY_TARGETS`,
  `NOISE_TARGETS`, `NUMERIC_TARGETS`, `TOPIC_TARGETS`, etc.)
- the server keeps running counts of already *issued* instructions per bucket
- at each new instruction, `_pick_underrepresented(...)` picks the bucket with
  the smallest `current_count / target_share` ratio (ties broken randomly)

This makes the distribution converge toward the desired ratios and avoids drift
when generation is interrupted or runs in long waves.

There are also a few dependency rules to avoid incoherent combinations:

- if `numeric_density == montants_et_dates`, the server forces a date precision
  that is not “none”
- certain persona/topic pairs are blocked to reduce forced contradictions
  (example: PACS/concubin persona excludes some matrimonial-regime liquidation
  topics)
- secondary topics are sampled more often for complex/hard-negative cases to
  increase multi-layer situations

## Why The Quotas Look Like This

The generation profile is designed for a future `description -> structured JSON`
task.

That means the corpus should contain:

- enough variability to avoid overfitting to a single phrasing style
- enough noise to make the extractor robust
- but still a strong majority of parseable, information-rich, realistic cases

So the current bias is toward:

- cleaner writing
- more numbers
- more explicit dates
- medium and long descriptions
- softer hard negatives rather than overly chaotic ones

## Evaluation Direction

Evaluation will be a major differentiator for this project.

During research, we looked at NuExtract as a relevant reference point for
generic JSON extraction.

Important takeaways:

- NuExtract uses a custom benchmark built around extraction tasks and manually
  prepared gold outputs
- their public writeups describe a structure-aware comparison, not simple raw
  JSON string equality
- they report aggregate similarity / F-score style metrics across extraction
  problems
- their exact full benchmark and scoring pipeline are not fully published

This matters because our setup is narrower and more favorable:

- one legal domain only: French succession and liberalities
- one output family only: the same target JSON schema every time
- the main input mode is free-form user descriptions

Because of that, our evaluation can be stricter and more useful than a generic
benchmark.

The current working evaluation plan for the later `description -> JSON` phase
is:

- measure valid JSON rate first
- measure exact schema compliance
- measure field-level precision, recall, and F1
- use type-aware normalization for dates, amounts, and percentages
- compare arrays in an order-insensitive way when order is not semantically
  relevant
- track document-level exact match as a high bar

For this legal extraction use case, false positives are especially costly. So
the final scorecard should likely emphasize precision-forward metrics in
addition to standard F1.

In short:

- NuExtract is a useful benchmark reference
- but our final evaluation should be more rigid, because our schema is fixed and
  our domain is much narrower

## Current Runtime Status

At the time of this note:

- the instruction server is configured with the v6 quota profile
- the repo already contains the updated summaries and generated training exports
- current server counters: `issued=3803`, `submitted=3760` (as of 2026-03-01)
- remaining to target: `1240` (generation paused after a completed wave)

Operational detail (generation throughput):

- Agents were initially orchestrated one case at a time, but this created heavy overhead.
- We introduced an internal `BATCH_SIZE` per agent (e.g. 20 → 50 → 100): each agent still
  generates and submits cases *one-by-one*, but it loops locally before replying back to
  the coordinator. This improves throughput without changing the per-case constraints.

## Immediate Next Steps

The next implementation step, if we continue generation now, is:

1. use the instruction server to issue prompts
2. let AI agents generate one case each
3. submit those cases back to the server
4. progressively fill the synthetic target toward `4839`

## Later Phases

Once the JSON schema is available, the next major milestone will be:

- converting this project from “case generation only” into a paired dataset
  pipeline for `description -> exact JSON`

That future phase will likely add:

- schema storage
- schema validation
- gold output generation / annotation
- train / validation / test splitting
- Mistral fine-tuning job orchestration
- W&B tracking for fine-tuning runs

## Presentation Summary

If you need a short verbal summary during the hackathon:

- We are building the input side first.
- We already imported all existing E2E-tested succession descriptions.
- We added a controlled synthetic data generation server.
- The current profile is optimized for future structured extraction quality.
- The final JSON schema and supervised outputs will be added in the next phase.

## Decision Log (Chronological)

This section logs the practical decisions made so far, in order.

1. Scope clarified
- Product target: free-form French succession description -> structured JSON.
- Immediate phase: generate and curate case descriptions only.

2. Seed data consolidated from real E2E coverage
- Imported all case-like prompts passing through E2E in `../w5`.
- Built a deduplicated seed corpus of `161` descriptions.

3. Synthetic generation infrastructure added
- Implemented a local instruction server to emit controlled prompts.
- Added persistence, quota tracking, and merged training export files.

4. Diversity balancing strategy introduced
- Defined multi-axis quotas (difficulty, narrative voice, noise, numeric density,
  date precision, length, themes, personas).
- Added hard-negative controls to avoid overfitting to easy cases.

5. Quotas iteratively refined
- Progressed from early profiles to a production-oriented `v6`.
- Shifted toward cleaner and more extractable data while preserving realism.
- Reduced fully messy samples and amount-free samples.
- Increased exact dates and information-rich numeric content.

6. Evaluation direction formalized
- NuExtract research used as benchmark inspiration.
- Decision: use stricter fixed-schema evaluation for this project (valid JSON,
  schema compliance, field-level PR/F1, normalization, doc-level exact match).

7. Model strategy aligned
- Student target remains small and deployable (`Ministral 3B Instruct`).
- Plan is teacher-student style data annotation later (strong model creates
  labels, small model learns).
- ONNX remains a post-training deployment step, not a training prerequisite.

8. Initial split policy selected for first annotated batch
- Preferred split for first supervised iteration: `700 train / 150 val / 150 test`
  on a 1000-sample labeled set.
- Test set reserved for final evaluation only.

## What Is Logged As "Done" vs "Later"

Done now:
- E2E seed import
- synthetic instruction server
- v6 quotas and tracking
- hackathon documentation and evaluation direction

Later (once schema is available):
- target JSON schema finalization
- `description -> JSON` annotation
- strict schema validation on labels
- supervised fine-tuning runs with W&B
- final model evaluation and deployment packaging

## Research Archive

This section keeps a persistent summary of the external and strategic research
done during the project, so the hackathon restitution can explain not only what
was built, but also why the direction was chosen.

### 1. Model format and deployment research

- The team explored the difference between:
  - base checkpoints
  - instruct checkpoints
  - ONNX deployment checkpoints
  - API-hosted fine-tuned models
- Main conclusion:
  - ONNX is a deployment format, not the correct starting point for training.
  - If browser deployment is needed later, export should happen after training,
    not before.

### 2. Why the project stayed on "case generation first"

- The final product target is structured extraction, but the target JSON schema
  is still under design.
- Because of that, the project intentionally focused first on building a large,
  diverse pool of French succession descriptions.
- This preserves momentum while keeping the generated corpus reusable for the
  later `description -> JSON` phase.

### 3. NuExtract benchmark research

- NuExtract was examined as a benchmark reference for generic JSON extraction.
- Key learning:
  - they rely on custom extraction benchmarks with manually prepared gold data
  - they compare outputs in a structure-aware way, not through raw JSON string
    equality
  - they report aggregate F-score / similarity-style metrics
- Implication for this project:
  - because our domain is narrower and our schema is fixed, we can use a more
    rigid and more meaningful evaluation protocol than a generic extractor
    benchmark.

### 4. Teacher-student strategy research

- A major strategic decision emerged from the research:
  - use a stronger model to generate target JSON labels
  - use those labels to fine-tune a smaller production-oriented model
- This is effectively a teacher-student / distillation-style workflow.
- For this project, the intended student remains a small Mistral-family model,
  while the teacher can be a stronger general model used only for annotation.

### 5. Fine-tuning method research

- Distinctions clarified:
  - fine-tuning = the general adaptation process
  - LoRA = a lightweight fine-tuning method
  - quantization = primarily an inference/deployment optimization
  - distillation = a teacher-student data generation strategy
- Main conclusion:
  - if training is done through Mistral's managed fine-tuning, LoRA/TRL/Unsloth
    are not directly exposed to the user
  - if training were self-hosted later, LoRA would be the practical default

### 6. Mistral managed fine-tuning research

- Managed Mistral fine-tuning was selected as the most realistic first training
  path because no local GPU is available.
- Key conclusions:
  - the first realistic target model is `ministral-3b-latest`
  - `ministral-8b-latest` is the natural fallback if `3b` underperforms
  - managed fine-tuning provides an API-hosted fine-tuned model
  - managed fine-tuning is not the right path if the immediate goal is weight
    export for browser ONNX deployment

### 7. Why `ministral-3b-latest` is the first target

- The task is narrow and repetitive: free-form description to a fixed JSON
  schema in one legal domain.
- That profile favors a smaller specialized model over a larger general model.
- The current recommendation is:
  - first supervised run on `ministral-3b-latest`
  - evaluate
  - only move upward in model size if needed

### 8. Public fine-tuning ecosystem research

- Publicly, the ecosystem of community fine-tunes is much richer around
  `Mistral-7B` than around the newer `Ministral` family.
- This is mainly due to maturity and ecosystem age, not because `Ministral` is
  necessarily weaker for the target use case.
- Resulting interpretation:
  - community examples around `Mistral-7B` are useful as implementation
    references
  - but the product-oriented model choice still favors the smaller `Ministral`
    family for deployment economics

### 9. Dataset preparation research

- The dataset should be fully prepared before the first real supervised run.
- Recommended first milestone:
  - `1000` labeled pairs
  - split as `700 train / 150 val / 150 test`
- Why:
  - reproducibility
  - clearer evaluation
  - clean train/validation separation
  - test set preserved for final measurement only

### 10. Output schema design research

- One of the most important schema design decisions was clarified:
  - avoid "everything optional"
  - prefer a stable structure with nullable values
- Current preferred rule:
  - stable top-level objects
  - unknown scalar values -> `null`
  - collections with no items -> `[]`
- This should make both training and evaluation significantly more stable.

### 11. Instructor / schema-enforced inference research

- `Instructor` was identified as useful, but not as a training method.
- It is useful for:
  - enforcing structured output during label generation
  - validating generated labels against a schema
  - producing cleaner supervised pairs
- It is not itself a replacement for fine-tuning.

### 12. W&B / TRL / Unsloth clarification

- `W&B` is mainly a training tracking and experiment management layer.
- `TRL` and `Unsloth` are training stacks/frameworks for self-hosted fine-tuning.
- For managed Mistral fine-tuning:
  - W&B remains useful for monitoring
  - TRL / Unsloth are not required in the first phase

### 13. Teacher model choice research

- A strong Mistral model can be used in batch mode to generate target labels.
- Current preferred teacher direction:
  - use `mistral-large-latest` to produce high-quality "silver" labels
  - validate and clean them
  - fine-tune `ministral-3b-latest` on the resulting pairs
- This creates a strong first path toward a domain-specialized student model
  without needing local GPU infrastructure.

### 14. Instruction server payload refinement

- The synthetic instruction server was upgraded so each emitted instruction is
  more usable by delegated agents.
- Instructions now include:
  - a compact executable prompt
  - a `dimension_guide` explaining each selected axis
  - the allowed values for every axis
  - a `style_brief`
  - explicit `must_include` and `must_avoid` lists
- This makes agent generation more reliable and easier to audit during the
  hackathon.

### 15. First paired-generation attempt (full-schema instance)

- We ran a first real paired-generation attempt using:
  - the local instruction server in pair mode
  - a delegated agent
  - the full extraction schema at
    `../w5/glinerExtract/schema/schema.full.json`
- The first attempt failed in a predictable way:
  - the schema is large and custom (not standard JSON Schema)
  - the agent spent too much time exploring the schema structure instead of
    generating the pair
  - the interruption happened before a usable `case_text + target_json` object
    was returned
- This failure was informative rather than wasted:
  - it confirmed that directly asking an agent to both parse a large custom
    schema and fill a full instance in one pass is too expensive and too slow
  - it also confirmed that we need a preprocessed representation of the schema
    for generation
- A mitigation was added immediately:
  - we generated a full instance template from the schema where:
    - scalar leaves are replaced by `null`
    - list leaves are replaced by `[]`
  - this template is stored at:
    `data/case_instruction_server/schema_instance_template.full.json`
- Key conclusion from this attempt:
  - the next attempts should not start from the raw descriptive schema
  - they should start from either:
    - a prebuilt instance template, or
    - a sparse-output rule set where only present fields are emitted

### 16. Sparse pair generation adopted and first batch completed

- We switched to a strict sparse-output policy for `target_json`:
  - full schema remains the source of truth for allowed keys
  - only fields supported by the case are emitted
  - no `null`, no empty arrays, no empty objects
- This significantly reduced generation friction for large-schema extraction.
- We then completed and stored four additional paired samples:
  - `INS-0003`
  - `INS-0004`
  - `INS-0005`
  - `INS-0006`
- Current local generation status after this batch:
  - submitted paired samples: `5` total
  - issued instructions: `6`

### 17. Storage contract switched to `case_text + target_toon` only

- We aligned the instruction server storage contract with the TOON-first pipeline:
  - submission now requires `target_toon`
  - legacy `target_json` is no longer accepted as target payload
  - training exports write assistant output as TOON text
- A startup sanitization pass was added to prevent legacy drift:
  - remove any `target_json` key from submitted rows
  - drop legacy submitted rows that do not contain a valid non-empty `target_toon`
  - migrate old issued instruction metadata (`required_keys`, submission contract, prompt token) from `target_json` to `target_toon`
  - delete stale legacy `_last_instruction.json` when it still contains `target_json`
- Net result for ongoing generation:
  - stored pair payload is constrained to `case_text` and `target_toon` (plus metadata)
  - no JSON target object is persisted as training target

### 18. Target-first generation strategy adopted (TOON-first)

- We formally switched from "text-first" to "target-first" generation:
  - server generates a constrained structured target internally
  - target is encoded as TOON and sent to the agent
  - agent generates only `case_text` from that target
  - submission is validated and stored as `{case_text, target_toon}`
- Why this was chosen:
  - lowers label drift risk
  - makes business constraints enforceable before text generation
  - enables iterative tightening of domain rules without changing storage format

### 19. Dedicated endpoint added for TOON-first flow

- Added endpoint:
  - `GET/POST /next-target`
- Behavior:
  - emits standard instruction dimensions
  - includes server-generated `target_toon`
  - returns a text-only generation prompt aligned with that target
- Submission behavior update:
  - `POST /submit-case` now accepts target omission when instruction carries server-generated `target_toon`
  - if a payload target is provided for a target-first instruction, it must exactly match the locked server target
  - mismatch is rejected

### 20. Simplified TOON-first API iteration

- We simplified the API flow after team feedback:
  - keep a single instruction endpoint (`/next-instruction`)
  - do not return `target_toon` in the public response
  - require submit payload to contain only `instruction_id` + `case_text` (plus optional `agent_id`)
- Internal behavior:
  - `/next-instruction` now generates and stores a hidden server-side TOON target (`server_target_toon`) per instruction
  - `/submit-case` resolves that hidden target, validates text/target coherence, and stores `{case_text, target_toon}`
  - client-provided `target_toon` is now rejected in submit payloads
- Rationale:
  - less client complexity
  - lower leak/copy risk on target payloads
  - easier business-constraint iteration while preserving training pair quality

### 21. Schema-driven target generator hardening

- We upgraded the hidden target generator to be anchored to the master schema:
  - master schema path is now configurable (`--master-schema-file`)
  - default points to `../w5/glinerExtract/schema/schema.full.json`
- Generation now follows a constrained pipeline:
  - sample instruction dimensions
  - generate sparse candidate target
  - validate strict sparse rules (no null, no empty object/list)
  - validate type/enum/path compliance against `schema.full.json`
  - apply business coherence checks before TOON encoding
- Submit flow remains text-only:
  - client sends only `instruction_id` + `case_text`
  - server resolves hidden `server_target_toon` and validates text/target coherence

### 22. Step-by-step business-safe generation (no incoherent targets)

- We replaced the previous monolithic target fill with a staged generation flow:
  - Step 1: identity/legal core (`famille.defunt`, partner linkage)
  - Step 2: topic blocks selected from schema prefixes
  - Step 3: business integrity repair pass (date/age/status/link consistency)
  - Step 4: strict validation (sparse + business + full schema)
- The generator is now explicitely **schema-driven** (not heuristic-only):
  - source of truth: `../w5/glinerExtract/schema/schema.full.json`
  - only allowed leaves are emitted
  - output is sparse by contract (`no null`, `no empty object`, `no empty list`)

Quality checks run after the update:

- `python -m py_compile src/ministral_ft/case_instruction_server.py` passed
- End-to-end server smoke (`next_instruction`) passed on multiple cases
- Offline simulation (500 instructions, 50 retries max each):
  - generation failures: `0`
  - unique leaf coverage: `398 / 398` (`100%` of schema leaves)
  - complexity mix remained aligned with configured quotas (`20/40/24/16`)
- Coherence spot checks (defunt age/date consistency) passed on sampled outputs.

Implication for the pipeline:

- TOON targets are now generated step-by-step with business constraints first
- agents only need to produce `case_text`
- stored training pair remains `{ case_text, target_toon }`

### 23. Final API contract (agent receives TOON)

We iterated on the API contract after user feedback and clarified the final rule:

- `/next-instruction` returns a minimal instruction payload for agents:
  - `instruction_id`
  - `target_toon`
  - `prompt` (TOON -> French free-form case text, no JSON)
- `/submit-case` remains text-only:
  - accepts `{ instruction_id, case_text }` (+ optional `agent_id`)
  - rejects any client-provided `target_toon` (server is source of truth)

Important business-side integrity guard:

- Topic quotas are now enforced on the TOON itself:
  - if the drawn topic is `assurance_vie`, the emitted TOON contains an insurance contract block, etc.
  - generation is rejected/retried if topic <-> TOON alignment fails

Smoke tests executed (temporary state dir, HTTP end-to-end):

- `/health` ok
- `/next-instruction` with forced `topic=assurance_vie` returned a TOON that decodes and includes `assurance_vie.contrats`
- `/submit-case` rejected payloads containing `target_toon` and accepted a `case_text` containing all extracted `*_nom(s)`
- `/dashboard` reflected `issued=1` / `submitted=1`

### 24. Preventing schema-token leakage + persona/target coherence

We hit two recurring failure modes in early generations:

- Agents copy enum codes from the TOON into the French text (examples: `PARTENAIRE_PACS`, `NEVEU_NIECE`, `PROPRE_DEFUNT`, `IMPOT_SUCCESSION`).
- Some agents "invent" narrator facts (e.g. "I am the PACS partner") that are not present in the target, creating accidental contradictions in the training set.

Mitigations implemented in the server:

- Prompt hardening: `/next-instruction` prompt now explicitly requires translating any `MAJUSCULES_AVEC_UNDERSCORE` enum codes into natural French (no underscores).
- Submission validation: `/submit-case` now rejects any `case_text` containing tokens matching `\\b[A-Z]{2,}(?:_[A-Z0-9]{2,})+\\b`.
- Persona-target coherence: the schema-driven target builder now ensures the TOON includes the minimal people/blocks implied by the persona, so "I am the spouse / PACS partner / sibling / child" cannot be true in text unless it is also true in the target.

This keeps the training corpus usable for extraction (French input) without leaking internal schema strings into the text and reduces accidental contradictions.
