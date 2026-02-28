# Hackathon Status

This file is the running English handoff note for the hackathon. It is intended
to be updated as the project evolves, so the current state, decisions, and next
steps remain easy to present.

## Goal

The long-term product goal is:

- accept a free-form French succession description from a user
- transform it into a structured JSON object matching a target schema

The target JSON schema is not defined yet. Because of that, the current phase is
focused on collecting and generating high-quality input descriptions only.

## Current Phase

The current phase is input corpus building.

What we are doing now:

- collecting real succession case descriptions already exercised by E2E tests
- generating many additional synthetic descriptions with controlled diversity
- preparing a large training-ready pool of user-style descriptions

What we are **not** doing yet:

- defining the final JSON schema
- generating gold JSON outputs
- training the final `description -> JSON` model
- exporting a fine-tuned model to ONNX

## Why This Order

We intentionally separated the work into phases:

1. Build a strong corpus of varied input descriptions.
2. Define the target JSON schema.
3. Annotate or generate the expected JSON outputs.
4. Fine-tune the model on `description -> JSON`.
5. Optionally export for browser inference later.

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
- persists issued instructions and generated cases
- updates progress and quota summaries on disk
- exports a merged training file as new cases are submitted

Main files:

- `src/ministral_ft/case_instruction_server.py`
- `scripts/run_case_instruction_server.sh`
- `data/case_instruction_server/config.json`
- `data/case_instruction_server/summary.md`
- `data/case_instruction_server/full_training_cases_mistral.jsonl`

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
