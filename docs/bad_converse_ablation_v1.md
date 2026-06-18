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

## Resume Snapshot: 2026-06-18

The previously stalled run at
`results/run_20260515_212337_bad_converse_ablation_v1/` was resumed after the
matrix runner gained per-cell checkpoints. The original metadata showed that the
spec had expanded successfully to 24 planned cells, but the old runner only
wrote final artifacts after every cell completed. A backend abort or interrupted
process could therefore leave only `run_meta.json`.

Resume command:

```bash
PYTHONNOUSERSITE=1 python3 sr_rewind_cot_trace_matrix.py \
  --spec sr_rewind_cot_assets/trace_matrices/bad_converse_ablation_v1.json \
  --model ./model/llama-3.2-3b \
  --name llama-3.2-it \
  --device mps \
  --hf-backend mlx \
  --dtype float16 \
  --max-new-tokens 16 \
  --temps 0.0 \
  --n 4 \
  --prompt-version v2 \
  --prompt-family general_reasoning \
  --seed 12345 \
  --out-dir results/run_20260515_212337_bad_converse_ablation_v1
```

Completion check:

- `run_meta.json`: `status=complete`, `n_completed_results=24`,
  `n_expected_results=24`.
- `checkpoints/`: 24 per-cell checkpoint files.
- `matrix_summary.csv`: 24 rows.
- `matrix_rates.csv`: 129 rows.
- `plots/yes_rate_heatmap_t0p0.png`: generated successfully.

Full-prefix `Yes` rates at `temperature=0.0`, averaged across the three nonce
cases:

| Variant | Mean full-prefix `Yes` rate | Per-case rates |
| --- | ---: | --- |
| `control_premises` | `0.00` | `0.00, 0.00, 0.00` |
| `valid_direct_conclusion` | `0.00` | `0.00, 0.00, 0.00` |
| `invalid_converse_only` | `0.00` | `0.00, 0.00, 0.00` |
| `may_still_without_converse` | `1.00` | `1.00, 1.00, 1.00` |
| `possibility_language_only` | `0.33` | `1.00, 0.00, 0.00` |
| `bad_converse_no_therefore` | `1.00` | `1.00, 1.00, 1.00` |
| `bad_converse_full` | `1.00` | `1.00, 1.00, 1.00` |
| `converse_repaired` | `0.00` | `0.00, 0.00, 0.00` |

The crisp result is that the invalid converse alone did not induce the wrong
answer in this deterministic probe. The answer-shaped modal phrase `may still be`
was sufficient across all nonce cases, and the repaired trace pulled the model
back to `No`.

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
