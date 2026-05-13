# General Reasoning Text Mode

Text-mode traces are a companion instrument for `general_reasoning` batches. They
are useful when a model has the right reasoning content but JSON formatting makes
the trace or answer hard to interpret.

## When To Use It

Use `general_reasoning_observation_v1_text.yaml` when:

- JSON traces contain schema fragments, partial objects, or answer text that is
  mostly formatting noise.
- You want to inspect local explanatory steps as plain prose.
- You are comparing whether a failure is semantic drift or just output-format
  fragility.

Use the JSON profiles first when you need stricter structure for downstream
automation. Text mode is easier to read, but exact string metrics become harsher
because open-ended explanations paraphrase naturally.

## Current Local Observation

In local MLX probes on the binary-search question, text mode produced a clean
five-step trace and did not show the older `StepGenerationResponse(...)` wrapper
leakage seen in earlier reconstructed artifacts.

The same probe also showed why raw exact match is too strict for open-ended
answers:

- The baseline answers were semantically close but split across paraphrases.
- `auc_match` could be `0.0` even when the forward and rewind answers described
  the same core idea.
- `rewind_loop_closure_mean` was more informative than exact match for judging
  whether rewind preserved the local structure.

For text-mode runs, inspect these fields together:

- `baseline_raw_answers`
- `forward_raw_answers_by_k`
- `rewind_trace.raw_recoveries`
- `trace_vs_rewind_preservation_rate`
- `rewind_loop_closure_mean`

## Runtime Profile

The text profile uses the same fast rewind budget as
`general_reasoning_observation_v1_fast.yaml`:

- `rewind_answer_max_new_tokens: 48`
- `rewind_sample_schedule: "pyramid"`
- `rewind_sample_min_per_depth: 3`

This keeps the six-question batch practical while preserving enough rewind and
oracle-tail signal for exploratory comparison.
