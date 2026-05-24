# Bad-Converse Ablation v1

This note follows the strongest signal from
`syllogism_interference_matrix_v1.md`: the `bad_converse` trace repeatedly
acts as a `Yes` attractor even though the correct answer is `No`.

The goal is to localize the attractor. Instead of asking whether trace
interference exists, this matrix asks which phrase family is doing the work.

## Assets

Spec:

```text
sr_rewind_cot_assets/trace_matrices/bad_converse_ablation_v1.json
```

Runner:

```bash
python3 sr_rewind_cot_trace_matrix.py \
  --spec sr_rewind_cot_assets/trace_matrices/bad_converse_ablation_v1.json \
  --model ./model/llama-3.2-3b \
  --device mps \
  --hf-backend mlx \
  --dtype float16 \
  --n 8 \
  --temps 0.0,0.4
```

Outputs have the same shape as the broader syllogism matrix:

- `matrix_rates.csv`: one row per temperature, case, variant, and prefix depth.
- `matrix_summary.csv`: one row per temperature, case, and variant.
- `matrix_results.json`: raw generated answers by prefix.
- `plots/yes_rate_heatmap_t*.png`: heatmaps where each cell is `P(yes)`.

## Variant Logic

The ablation keeps the same three nonce syllogisms and compares eight trace
variants:

- `control_premises`: original premises only.
- `valid_direct_conclusion`: premises plus the correct `No` conclusion.
- `invalid_converse_only`: adds the invalid converse but stops before a
  permissive conclusion.
- `may_still_without_converse`: asks whether `may still be` alone creates the
  `Yes` attractor.
- `possibility_language_only`: softer possibility wording without the exact
  answer-shaped `may still be` phrase.
- `bad_converse_no_therefore`: the full bad trace without `Therefore`.
- `bad_converse_full`: the original bad-converse trace.
- `converse_repaired`: adds a repair step after the invalid converse.

## Reading The Result

The most important cells are prefix depths after the ablated phrase enters the
prompt:

- If `invalid_converse_only` turns bright, the invalid converse itself is enough
  to induce the wrong answer.
- If `may_still_without_converse` turns bright, the answer-shaped modal phrase
  is the stronger cause.
- If `bad_converse_no_therefore` and `bad_converse_full` match, the discourse
  marker `Therefore` is probably not the main driver.
- If `converse_repaired` turns dark at the final prefix, a later valid repair
  can pull the model back out of the wrong-answer basin.

## Research Payoff

This is a more paper-facing probe than the single anomaly. A phrase ablation
lets us make a mechanistic claim about trace interference:

```text
Some trace phrases behave like answer-space control tokens, not neutral
explanatory text.
```

That framing is stronger than saying the model made a reasoning mistake. It
suggests that certain generated reasoning fragments can steer later answers
toward a local attractor, even when the underlying task is a trivial closed
yes/no syllogism.
