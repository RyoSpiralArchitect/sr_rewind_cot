# Closed-Answer Reasoning Pilot

This note captures the first local pilot for a closed-answer reasoning profile.
It is intentionally small: the goal is to make answer-space transitions easier to
read before scaling to larger batches.

## Why This Family Exists

Open-ended explanatory prompts are useful for studying semantic compression, but
their answers paraphrase naturally. That makes exact-match curves pessimistic
even when the model is preserving the right idea.

Closed-answer prompts add a complementary axis:

- The answer surface is small enough for exact match to become meaningful again.
- Wrong attractors can be named directly, such as `Carol` instead of `Bob`.
- Step influence can identify whether a final answer-like step is carrying the
  result or whether earlier constraints are doing the work.
- Rewind failures become easier to interpret because recovered tails often fall
  into a specific wrong answer basin.

## Pilot Run: Carol Trap

Local run:

```bash
python3 sr_rewind_cot.py quick-hf \
  --model ./model/llama-3.2-3b \
  --device mps \
  --hf-backend mlx \
  --dtype float16 \
  --name llama-3.2-it \
  --question "A crime was committed by exactly one of Alice, Bob, and Carol. Alice says, 'Bob did it.' Bob says, 'Alice did it.' Carol says, 'Bob is telling the truth.' Exactly one statement is true. Who did it? Reply with exactly one line: Answer: <name>." \
  --trace-mode masked \
  --trace-format text \
  --temp-trace 0.0 \
  --process-reward-mode off \
  --trace-candidates 1 \
  --temp-reanswer 0.4 \
  --n-baseline 1 \
  --n-per-k 1 \
  --max-steps 8 \
  --max-new-tokens 192 \
  --prompt-version v2 \
  --prompt-family general_reasoning \
  --rewind-step-temp 0.5 \
  --rewind-samples-per-depth 1 \
  --rewind-answer-max-new-tokens 16 \
  --rewind-sample-schedule flat \
  --rewind-process-reward-mode off \
  --rewind-step-candidates 1 \
  --rewind-novelty-retries 0 \
  --no-oracle-tail \
  --no-bridge \
  --semantic-match-threshold 0.72 \
  --step-influence-mode lite \
  --step-influence-samples 1 \
  --step-influence-max-new-tokens 16 \
  --save-raw-generations \
  --save-generation-log-jsonl
```

Output directory:

```text
results/run_20260513_082718
```

Key summary values:

| Field | Value |
| --- | --- |
| `trace_format` | `text` |
| `L` | `8` |
| `A_star` | `bob.<|eot_id|>` |
| `auc_match` | `0.1111111111111111` |
| `auc_semantic_match` | `0.1111111111111111` |
| `rewind_auc_match` | `0.0` |
| `rewind_auc_semantic_match` | `0.0` |
| `step_influence_full_semantic_match_rate` | `1.0` |
| `step_influence_max_necessity_step` | `7` |
| `step_influence_max_necessity_semantic_similarity` | `1.0` |
| `step_influence_max_sufficiency_step` | `3` |
| `step_influence_mean_recovered_substitution_delta_semantic_similarity` | `-0.234375` |

The forward prefix curve was especially legible:

| Prefix depth | Top answer |
| --- | --- |
| `k=0..6` | `carol.<|eot_id|>` |
| `k=7` | `alice.<|eot_id|>` |
| `k=8` | `bob.<|eot_id|>` |

The recovered rewind curve stayed in the Carol basin:

| Rewind depth | Typical top answer |
| --- | --- |
| `k=0` | `carol did it` |
| `k=1..8` | `carol committed the crime` |

## Reading

The strongest observation is not "the model solved the proof faithfully." The
raw trace contains local reasoning that looks suspect, while the final answer
still lands on Bob. This makes the case useful as a phenomenological probe:

- The closed answer space exposes a crisp transition from `Carol` to `Alice` to
  `Bob`.
- The last forward step behaves like a decisive answer lock.
- The reverse operator reconstructs plausible steps, but those steps pull the
  answer back toward the wrong Carol attractor.
- Negative recovered-step substitution suggests that reverse plausibility is not
  the same thing as faithful forward structure.

This is exactly the kind of failure shape that is hard to see in open-ended
explanatory prompts because the wrong basin may only be a paraphrase away.

## Follow-Up Protocol

Use `sr_rewind_cot_assets/question_sets/closed_answer_reasoning_v1_text_micro.yaml`
for quick follow-up sweeps.

Initial profile smoke run:

```bash
python3 sr_rewind_cot.py run --config sr_rewind_cot_assets/question_sets/closed_answer_reasoning_v1_text_micro.yaml
```

Output directory:

```text
results/run_20260513_085151
```

This run used four low-sample text-mode questions:

| Question | `A_star` | Forward shape |
| --- | --- | --- |
| `carol_trap` | `bob.<|eot_id|>` | wrong basin shifts through `Alice` / `Carol`, then locks on `Bob` at full prefix |
| `mislabeled_boxes` | `apples.<|eot_id|>` | stable from the empty prefix; useful as an easy control |
| `knight_knave_pair` | `<knight><|eot_id|>` | semantically stable, with surface-format variants around `<knight>` |
| `syllogism_exclusion` | `no<|eot_id|>` | mostly `No`, but the full prefix flipped to `Yes`, making it a compact trace-interference case |

The `selection_rule` / Wason-style card prompt was intentionally left out of the
micro profile after a local smoke run: it was interesting, but it generated a
longer incomplete trace and behaved more like a hard fallacy probe than a quick
closed-answer sanity check.

The `syllogism_exclusion` flip is tracked separately in
`docs/syllogism_trace_interference_v1.md`.

Recommended reading order:

1. `summary.csv`
2. `__match.png`
3. `__step_influence.png`
4. `<backend>__<qid>.json`
5. `<backend>__<qid>__generations.jsonl`

Things to watch:

- Does the forward curve show a named wrong-answer basin before the correct
  answer appears?
- Does rewind recover a prerequisite, or does it stabilize around a plausible
  but wrong explanation?
- Does a single final answer-like step dominate necessity?
- Do recovered-step substitutions improve or damage answer preservation?
