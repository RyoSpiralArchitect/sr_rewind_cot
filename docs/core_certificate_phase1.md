# Core Certificate Phase 1

This note marks the point where `sr_rewind_cot` starts moving from pure
observation toward intervention.

Until this phase, the main questions were:

- what survives when a forward trace is rewound
- when rewind collapses into a summary or attractor
- whether `PRM-lite` changes that collapse

Phase 1 adds a stronger claim surface:

```text
If a rewind candidate looks like a core, can we perturb, ablate, and replay it
enough to call it a core candidate rather than only a repeated sentence?
```

## What Phase 1 Adds

### 1. `fixed_point_type`

Rewind fixed points are no longer treated as one undifferentiated failure mode.

Current types:

- `parser_fixed_point`
- `answer_leak_fixed_point`
- `summary_fixed_point`
- `core_fixed_point`
- `no_fixed_point`

This is a lightweight taxonomy, not a final ontology. The goal is to avoid
calling everything "collapse" when different mechanisms are clearly in play.

### 2. `core_certificate_lite`

For a rewind-side core candidate `C`, the harness now computes a small
intervention bundle:

- `necessity`
  How much answer preservation drops when the nearest forward step is removed.
- `sufficiency`
  Whether the candidate alone can reproduce the answer.
- `stability`
  Whether small paraphrase/noise variants still reproduce the answer.
- `bidirectional_attractor`
  Whether the candidate is reachable from both the forward and rewind side.
- `minimality`
  Whether the candidate is short relative to the full trace.

These are combined into `core_certificate_score`.

This is deliberately called **lite** because it does not yet include
cross-backend transfer or splice-style tests.

## Why This Matters

The main conceptual shift is:

```text
"rewind repeated something"
```

becomes

```text
"rewind produced a candidate with a specific fixed-point type and a measurable
intervention profile"
```

That makes the candidate a more serious research object.

## How To Enable It

In YAML:

```yaml
core_certificate_mode: lite
core_certificate_samples: 3
core_certificate_paraphrases: 3
core_certificate_max_new_tokens: 48
```

On the CLI:

```bash
python3 sr_rewind_cot.py quick-hf \
  --model ./model/llama-3.2-3b \
  --device mps \
  --hf-backend auto \
  --dtype float16 \
  --name llama-3.2-it \
  --question "What is the fundamental difference between Pascal's Wager and agnostic theism?" \
  --prompt-version v2 \
  --prompt-family general_reasoning \
  --trace-format json \
  --core-certificate-mode lite
```

## What To Read In The Output

Start with `summary.csv`:

- `rewind_fixed_point_type`
- `rewind_core_candidate_text`
- `rewind_core_candidate_forward_similarity`
- `rewind_core_certificate_necessity`
- `rewind_core_certificate_sufficiency`
- `rewind_core_certificate_stability`
- `rewind_core_certificate_bidirectional_attractor`
- `rewind_core_certificate_minimality`
- `rewind_core_certificate_score`

Then inspect the per-question JSON for context:

- `rewind_trace.novelty_curve`
- `rewind_curve.curve`
- `trace_vs_rewind`
- `step_influence`

## Expected Early Behavior

This phase does **not** guarantee that many questions will immediately produce
high certificate scores.

That is part of the point.

Early runs may show:

- many `summary_fixed_point` cases with decent sufficiency but weak necessity
- some `answer_leak_fixed_point` cases that look strong until typed correctly
- a smaller number of candidates that are short, stable, and genuinely
  disruptive when removed

Those rare candidates are the most promising path toward a stronger notion of
core.

## What Still Comes Later

Phase 1 is only the first intervention layer.

Still missing from a stronger certificate:

- cross-backend transfer
- close-but-false replacement tests
- splice experiments across question families
- comparison-specific factorized rewind
- motif-level certificates for multi-step cores

So the right interpretation is:

```text
Phase 1 does not prove a final core theory.
It upgrades core talk from intuition to measurable candidate status.
```
