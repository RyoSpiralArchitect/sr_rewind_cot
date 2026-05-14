# Syllogism Trace Interference Pilot

This note follows up on a small anomaly from
`closed_answer_reasoning_v1_text_micro.yaml`: the syllogism prompt was mostly
answered correctly as `No`, but one full-prefix sample flipped to `Yes`.

Prompt:

```text
All dax are wugs. No wugs are mips. Can any dax be a mip?
Reply with exactly one line: Answer: <yes/no>.
```

The logical answer is `No`.

## Hypothesis

The first observation suggested three possible explanations:

- A single-sample artifact.
- A wording-sensitive interference effect caused by the trace conclusion.
- A rewind/converse-fallacy effect where reverse reconstruction invents a
  plausible but invalid relation.

To separate these, the pilot used fixed trace files under
`sr_rewind_cot_assets/traces/syllogism_interference_v1/` and re-answered from
each prefix with `n=12`.

Output directory:

```text
results/run_20260513_110646
```

The most useful output file is:

```text
results/run_20260513_110646/trace_interference_rates.csv
```

## Trace Variants

| Variant | Purpose |
| --- | --- |
| `premises_only` | Check whether the two premises are already enough to stabilize `No`. |
| `generated_property_conclusion` | Replay the model-generated conclusion wording that first showed the flip. |
| `direct_conclusion` | Replace the conclusion with a shorter direct `no dax can be a mip` wording. |
| `set_inclusion_conclusion` | Use class/subset language instead of property language. |
| `negative_property_conclusion` | Preserve property language but make the exclusion relation explicit. |
| `bad_converse_rewind_like` | Inject the kind of converse-like error seen in rewind recovery. |

## Results

| Variant / prefix | Yes rate | No rate | Read |
| --- | ---: | ---: | --- |
| `premises_only`, full prefix | `0/12` | `12/12` | Premises alone are stable. |
| `generated_property_conclusion`, full prefix | `2/12` | `10/12` | The original conclusion wording weakly increases `Yes`. |
| `direct_conclusion`, full prefix | `0/12` | `12/12` | Direct conclusion wording removes the flip in the prefix curve. |
| `negative_property_conclusion`, full prefix | `0/12` | `12/12` | Explicit exclusion wording is stable. |
| `set_inclusion_conclusion`, prefix after two class-language premises | `9/12` | `3/12` | Class/subset wording creates a strong transient `Yes` basin. |
| `set_inclusion_conclusion`, full prefix | `4/12` | `8/12` | The explicit conclusion partially repairs the transient basin. |
| `bad_converse_rewind_like`, before bad final step | `1/12` | `11/12` | The converse-like setup alone is mostly not enough. |
| `bad_converse_rewind_like`, full prefix | `7/12` | `5/12` | Adding the invalid conclusion pushes the answer into `Yes`. |

## Interpretation

The flip was not just a one-off sample, but it also was not caused by every
correct conclusion. The effect is wording-sensitive.

Three patterns are worth keeping:

- Direct premise wording is robust: `All dax are wugs` plus `No wugs are mips`
  strongly supports `No`.
- Abstract relation wording can destabilize the answer, especially class/subset
  phrasing before the final conclusion is supplied.
- Rewind-like converse errors are dangerous because they can stay mostly dormant
  until a final invalid conclusion turns them into a `Yes` attractor.

This makes the case a compact probe for trace interference: extra reasoning text
can change the answer distribution even when the underlying problem is trivial.

## Next Checks

Good follow-up runs:

- Increase `n` to `30` or `50` for the three most diagnostic variants:
  `generated_property_conclusion`, `set_inclusion_conclusion`, and
  `bad_converse_rewind_like`.
- Repeat with `temperature_reanswer=0.0` to separate sampling instability from
  deterministic prompt sensitivity.
- Add logically equivalent prompts with different nonce words to test whether the
  effect is lexical or structural.
- Run the same fixed traces with another model to see whether the class-language
  transient `Yes` basin is model-specific.

## Follow-Up: `n=50` And Greedy Re-Answering

The next pass ran the three diagnostic variants with `n=50`, first at
`temperature_reanswer=0.4` and then at `temperature_reanswer=0.0`.

Output directories:

```text
results/run_20260513_112724_syllogism_n50_t04
results/run_20260513_114245_syllogism_n50_t00
```

At `temperature_reanswer=0.4`:

| Variant / prefix | Yes rate | No rate | Read |
| --- | ---: | ---: | --- |
| `generated_property_conclusion`, full prefix | `5/50` | `45/50` | Weak sampling-sensitive drift. |
| `set_inclusion_conclusion`, prefix after two class-language premises | `30/50` | `20/50` | Strong transient `Yes` basin. |
| `set_inclusion_conclusion`, full prefix | `14/50` | `36/50` | Final conclusion partly repairs the basin. |
| `bad_converse_rewind_like`, before bad final step | `6/50` | `44/50` | Small drift before the invalid conclusion. |
| `bad_converse_rewind_like`, full prefix | `25/50` | `25/50` | Invalid conclusion makes the answer maximally unstable. |

At `temperature_reanswer=0.0`:

| Variant / prefix | Yes rate | No rate | Read |
| --- | ---: | ---: | --- |
| `generated_property_conclusion`, full prefix | `0/50` | `50/50` | The weak drift disappears under greedy decoding. |
| `set_inclusion_conclusion`, prefix after two class-language premises | `50/50` | `0/50` | The transient basin is deterministic, not just sampling noise. |
| `set_inclusion_conclusion`, full prefix | `0/50` | `50/50` | The explicit final class-overlap conclusion repairs the deterministic error. |
| `bad_converse_rewind_like`, full prefix | `0/50` | `50/50` | The `temp=0.4` instability is sampling-driven for this model/backend. |

This sharpens the interpretation:

- The original generated-property flip is mostly a sampling tail.
- The bad-converse trace creates a high-entropy answer region at `temp=0.4`, but
  greedy decoding still chooses `No`.
- The class/subset wording is the strongest trace-interference probe: after the
  two class-language premises, greedy decoding deterministically answers `Yes`,
  and the final explicit conclusion deterministically restores `No`.

That last pattern is the most paper-useful miniature example so far. It shows a
local prefix state whose answer is confidently wrong even though adding one more
logically compatible conclusion flips the same model back to the correct answer.

## Paper-Facing Claim Seed

This case suggests a compact claim worth developing:

```text
Reasoning traces are not monotonic evidence for the final answer.
Even logically compatible intermediate text can create transient answer
attractors that are confidently wrong, and a later trace step can repair rather
than merely add information.
```

The contribution would not be "the model fails at a syllogism." The stronger
angle is that a rewind / prefix intervention harness can locate where answer
space changes, then distinguish:

- sampling-tail drift, as in `generated_property_conclusion`;
- high-entropy instability, as in `bad_converse_rewind_like` at `temp=0.4`;
- deterministic trace interference, as in the `set_inclusion_conclusion` prefix.

The deterministic class-language case is especially useful because it is small,
reproducible, and mechanically interpretable:

| Trace state | Greedy answer |
| --- | --- |
| Original prompt only | `No` |
| `Every dax is inside the class of wugs.` | `No` |
| plus `The class of wugs has no overlap with the class of mips.` | `Yes` |
| plus `Therefore, the class of dax has no overlap with the class of mips.` | `No` |

That makes it a minimal "answer-space phase transition" example: the same model
moves from correct, to confidently wrong, back to correct under controlled trace
prefix edits.

## Conference Direction

A plausible short-paper framing:

1. Introduce rewind/prefix interventions as an instrument for reasoning-trace
   phenomenology.
2. Show that open-ended tasks require semantic metrics because exact match
   undercounts paraphrased preservation.
3. Use closed-answer tasks to make answer-space basins directly visible.
4. Present the syllogism interference probe as a minimal controlled case where
   trace wording, not task difficulty, determines whether the answer is stable.
5. Connect this back to rewind: reverse reconstruction can generate plausible
   but invalid intermediate states, and those states can be tested by the same
   prefix intervention machinery.

Candidate venues would depend on how far the evidence grows:

- workshop paper if the contribution stays as a measurement harness plus pilot
  phenomena;
- short paper if the fixed-trace protocol is repeated across several models and
  task families;
- full paper only after adding broader statistics, model comparisons, and a
  cleaner taxonomy of interference modes.

The next highest-value run is therefore not a larger random benchmark. It is a
small controlled matrix:

- 3 to 5 logically equivalent syllogisms with different nonce words;
- 3 trace wordings per syllogism: direct premise, class-language, and explicit
  repaired conclusion;
- at least 2 models;
- both `temperature_reanswer=0.0` and a moderate sampling temperature.

If the deterministic class-language flip survives that matrix, it becomes a
real result rather than a colorful anecdote.
