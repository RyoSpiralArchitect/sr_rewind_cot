# General Reasoning Speculative v1

`general_reasoning_speculative_v1` is a small three-question probe set for the
more open-ended side of `general_reasoning`.

It exists because the base observation batches were good at explanatory and
causal questions, but we also wanted a compact set that could stress:

- semantic-core collapse
- institutional vs social acceptance
- norm conflict and override structure

## The Three Questions

### 1. Existence, negation, and speakability

Question:

`If denying the existence of something is one way of marking its possible existence, can we meaningfully speak of something that truly exists but cannot be erased or negated?`

Why it is here:

- This is the most attractor-prone question in the set.
- It encourages the model to move from local reasoning steps into a smaller
  ontological vocabulary: `existence`, `negation`, `proposition`, `reality`.
- It is useful when we want to observe compression into a semantic core rather
  than faithful step recovery.

What to inspect:

- `__trace_vs_rewind_similarity.png`
- `__rewind_compare.png`
- `trace_vs_rewind_preservation_rate`
- `rewind_curve.raw_answers_by_depth`

Working expectation:

- Forward traces may expand into several explanatory steps.
- Rewind often collapses those steps into a smaller existence/negation core.

### 2. Misusage, dictionaries, and social acceptance

Question:

`If an incorrect usage becomes widespread enough to be recorded in dictionaries and is labeled as incorrect, does it then become institutionally recognized as a misusage, or has its meaning already been socially accepted despite that label?`

Why it is here:

- This is the cleanest institutional-vs-social split in the set.
- It often stabilizes around a comparison between dictionary labeling,
  institutional recognition, and social acceptance.
- It is especially useful for seeing whether rewind preserves a distinction or
  compresses it into a single "acceptance beats labeling" slogan.

What to inspect:

- `baseline_confidence`
- `__trace_axes_similarity.png`
- `__rewind_axes_similarity.png`
- `rewind_axis_base_prm_preservation_rate`

Working expectation:

- Answer-space can be more stable than the other two questions.
- Step-space may still drift, especially if the trace contains JSON-like or
  schema-heavy fragments.

### 3. Norm conflict and override

Question:

`When one fact violates one norm but is justified under another, what conditions cause the second norm to override the first?`

Why it is here:

- This is the most structure-preserving prompt in the set.
- It tends to produce explicit relations between `fact`, `norm`,
  `justification`, and `override`.
- It is a good candidate when we want to test whether rewind can retain a small
  but meaningful ordering skeleton instead of collapsing immediately.

What to inspect:

- `rewind_axis_base_prm_similarity_mean`
- `rewind_axis_base_prm_preservation_rate`
- `__rewind_axes_similarity.png`
- `__trace_candidate_scores.png`

Working expectation:

- Forward and rewind still differ a lot.
- Compared with the other speculative questions, rewind is more likely to keep a
  recognizable norm/justification frame.

## Why There Is Also a Text Variant

Open-ended speculative questions are more likely to generate:

- partial JSON fragments
- schema leakage into answers
- traces whose content is good but whose outer formatting is noisy

That is why `general_reasoning_speculative_v1_text.yaml` exists alongside the
JSON batch. The text batch is not "better" by default; it is a complementary
instrument for checking whether a phenomenon is semantic or just a formatting
artifact.

## Practical Use

Use:

- `general_reasoning_speculative_v1_fast.yaml`
  when you want a quick semantic-attractor probe

- `general_reasoning_speculative_v1_text.yaml`
  when the JSON run feels noisy and you want a cleaner reading of the same
  questions

- `general_reasoning_observation_v2_fast.yaml`
  when you want these three speculative prompts to live alongside the six
  original general-reasoning prompts in one larger batch
