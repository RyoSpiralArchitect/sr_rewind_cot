# Research Notes

This folder is for the human-readable trail around `sr_rewind_cot`: why certain
question sets exist, what specific plots are meant to reveal, and how local pilot
runs are shaping the longer-term research direction.

Current notes:

- [core_certificate_phase1.md](./core_certificate_phase1.md)
  The first explicit move from pure observation toward intervention, including
  `fixed_point_type` and `core_certificate_lite`.

- [bad_converse_ablation_v1.md](./bad_converse_ablation_v1.md)
  Phrase ablation for localizing which bad-converse wording creates the robust
  `Yes` attractor in simple syllogisms.

- [general_reasoning_speculative_v1.md](./general_reasoning_speculative_v1.md)
  Why the three speculative questions were added, what each one probes, and what
  the early runs seem to reveal.

- [general_reasoning_text_mode.md](./general_reasoning_text_mode.md)
  Practical notes from text-trace probes: when text helps, what it fixes, and
  why exact-match metrics can be misleading for open-ended explanations.

- [closed_answer_reasoning_v1.md](./closed_answer_reasoning_v1.md)
  Notes from the Carol-trap pilot and the motivation for a closed-answer
  reasoning micro profile.

- [syllogism_trace_interference_v1.md](./syllogism_trace_interference_v1.md)
  Follow-up on a tiny syllogism anomaly where added trace wording can push a
  stable `No` answer toward a transient `Yes` basin.

- [syllogism_interference_matrix_v1.md](./syllogism_interference_matrix_v1.md)
  Controlled matrix for testing whether the syllogism interference effect
  survives across nonce words, trace variants, and answer temperatures.

- [plot_field_guide.md](./plot_field_guide.md)
  A compact guide for reading the most useful plot families and deciding whether a
  run shows structural preservation, semantic drift, or core collapse.

- [roadmap.md](./roadmap.md)
  A living outline for moving from exploratory runs toward a paper-quality study.
