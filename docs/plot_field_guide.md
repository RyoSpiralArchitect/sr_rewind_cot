# Plot Field Guide

This is a short reading guide for the most useful output plots in `sr_rewind_cot`.

## 1. `__trace_vs_rewind_similarity.png`

What it shows:

- How similar each rewind-aligned step is to the corresponding forward step.

How to read it:

- Low similarity across all steps usually means semantic drift or compression.
- A shallow decline followed by a floor often suggests a small surviving core.
- One or two higher points in the middle can indicate that a local relation
  survives even when the rest of the trace changes.

## 2. `__rewind_compare.png`

What it shows:

- Forward re-answering, recovered rewind tails, and oracle tails on the same
  answer-reproduction curve.

How to read it:

- If rewind and oracle are both low, the task may be collapsing rather than
  reconstructing.
- If oracle stays above recovered rewind, the reverse operator may be losing
  structure that the original forward tail still contains.
- If recovered rewind approaches oracle, the reverse operator is probably doing
  real work rather than just drifting.

## 3. `__rewind_axes_similarity.png`

What it shows:

- The distance between base rewind and PRM-selected rewind steps.

How to read it:

- High similarity means PRM-selection is mostly choosing the same reverse path.
- Lower similarity means PRM-selection is actively reshaping rewind, which can be
  useful or harmful depending on whether it improves interpretability.

## 4. `__trace_candidate_scores.png`

What it shows:

- Forward trace candidates and their pseudo-PRM scores.

How to read it:

- A large gap between candidates suggests the scoring heuristic has a strong
  preference.
- If the chosen candidate is much shorter than the rest, the pseudo-PRM may be
  rewarding cleanliness or parseability more than reasoning richness.

## 5. `__rewind_novelty.png`

What it shows:

- How novel each recovered rewind step is relative to earlier recovered steps.

How to read it:

- A rapid drop can suggest entry into a semantic attractor.
- Repeated low novelty without exact repetition often means paraphrastic
  fixed-point behavior.
- High novelty throughout does not automatically mean quality; it may also signal
  drift.

## Suggested Reading Order

For a new run, a good default order is:

1. `summary.csv`
2. `__trace_candidate_scores.png`
3. `__trace_vs_rewind_similarity.png`
4. `__rewind_compare.png`
5. `__rewind_axes_similarity.png`
6. question JSON and JSONL files
