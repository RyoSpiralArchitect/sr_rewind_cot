# Roadmap Toward a Paper-Quality Study

This is a working roadmap, not a locked plan.

## Phase 1. Instrumentation

Goals:

- make runtime and workload costs visible
- distinguish prefill savings from decode costs
- separate forward, rewind, and oracle-tail timing

Current status:

- mostly in place
- metrics can now be compared run-to-run with `sr_rewind_cot_metrics.py`
- lightweight semantic answer preservation and step influence are now available
  beside exact match

## Phase 2. Task Taxonomy

Goals:

- organize prompts by what they stress
- avoid mixing explanatory tasks, logical tasks, and speculative tasks without
  saying so

Current status:

- `general_reasoning_observation_v1_*`: explanatory / causal / comparison tasks
- `general_reasoning_speculative_v1_*`: open-ended conceptual and normative tasks
- `general_reasoning_observation_v2_*`: combined batch for broader sweeps
- `closed_answer_reasoning_v1_text_micro.yaml`: small closed-answer logic pilot
  for named wrong-answer basins and step-influence readouts

## Phase 3. Rewind Phenomenology

Questions:

- When does rewind preserve local prerequisite structure?
- When does rewind collapse into a semantic core?
- When does PRM-lite stabilize rewind, and when does it distort it?

Operational signs:

- `trace_vs_rewind_preservation_rate`
- `rewind_axis_base_prm_preservation_rate`
- `step_influence_max_necessity_semantic_similarity`
- `step_influence_mean_recovered_substitution_delta_semantic_similarity`
- `rewind_curve.raw_answers_by_depth`
- `__rewind_novelty.png`

## Phase 4. Cleaner Comparison Protocols

Next improvements that would help a paper:

- paired `json` vs `text` runs for the same speculative prompts
- paired `json` vs `text` runs for the six-question observation batch using
  `general_reasoning_observation_v1_fast.yaml` and
  `general_reasoning_observation_v1_text.yaml`
- fixed random seeds and repeated runs per question family
- small curated benchmark subsets for each failure mode
- run manifests that explicitly record which profile was used and why
- closed-answer micro sweeps before larger batches, so exact-match behavior is
  calibrated against tasks where answer surfaces are intentionally constrained

## Phase 5. Stronger Process Models

Near-term:

- continue using pseudo-PRM heuristics as an interpretable control surface

Later:

- test learned or judge-backed PRM variants
- compare `Base CoT`, `PRM-CoT`, `Base Rewind`, and `PRM-Rewind` more formally

## Phase 6. Writing Shape

A plausible paper arc could be:

1. Motivation: reasoning reversal as a probe, not only as a solver
2. Method: forward trace, rewind operator, bridge / oracle comparisons
3. Metrics: preservation, novelty, fixed-point behavior, workload
4. Task families: explanatory vs speculative vs logical
5. Findings: compression, drift, partial preservation, PRM effects
6. Limitations: formatting artifacts, model dependence, prompt dependence

The main goal is not to prove rewind "works" in a single sense. The stronger and
more interesting claim is likely about the shapes of failure, compression, and
surviving structure.

## Phase 7. Conference-Facing Probe

The current strongest wedge is the closed-answer syllogism interference probe:

- direct premise wording keeps the answer stable;
- class/subset wording can create a deterministic wrong-answer basin at an
  intermediate prefix;
- an explicit compatible conclusion can repair that basin;
- rewind-like converse errors create high-entropy answer regions under sampling.

This points toward a focused workshop or short-paper story: use rewind and prefix
interventions not as solvers, but as instruments for locating answer-space phase
transitions inside generated reasoning traces.

Before scaling, prioritize controlled replications over large benchmark breadth:

- vary nonce words and surface wording;
- repeat across multiple local or hosted models;
- keep fixed-trace variants so runs are comparable;
- report both greedy and sampled re-answer distributions.
