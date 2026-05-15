# Syllogism Interference Matrix v1

This note upgrades the single-syllogism trace-interference probe into a small
controlled matrix.

The target phenomenon from `syllogism_trace_interference_v1.md` was:

```text
direct prompt -> No
class-language prefix -> Yes
explicit repaired conclusion -> No
```

The matrix asks whether that pattern survives small changes in nonce words and
trace wording.

## Assets

Spec:

```text
sr_rewind_cot_assets/trace_matrices/syllogism_interference_matrix_v1.json
```

Runner:

```bash
python3 sr_rewind_cot_trace_matrix.py \
  --spec sr_rewind_cot_assets/trace_matrices/syllogism_interference_matrix_v1.json \
  --model ./model/llama-3.2-3b \
  --device mps \
  --hf-backend mlx \
  --dtype float16 \
  --n 12 \
  --temps 0.0,0.4
```

Outputs:

- `matrix_rates.csv`: one row per temperature, case, trace variant, and prefix
  depth.
- `matrix_summary.csv`: one row per temperature, case, and trace variant.
- `matrix_results.json`: raw answers by prefix.
- `plots/yes_rate_heatmap_t*.png`: heatmaps where each cell is `P(yes)`.

## Matrix Shape

The first version uses three equivalent syllogisms:

- `dax_wug_mip`
- `blicket_rone_fep`
- `glim_taver_nork`

Each syllogism has four trace variants:

- `direct_premises`: original premise wording only.
- `direct_conclusion`: original premises plus a direct valid conclusion.
- `class_repaired`: class/subset wording with a final valid repair step.
- `bad_converse`: invalid converse-like wording inspired by rewind drift.

The important readout is not only the full-prefix answer. For
`class_repaired`, the expected interference point is the prefix after two steps:

```text
Every dax is inside the class of wugs.
The class of wugs has no overlap with the class of mips.
```

If the original observation generalizes, this prefix should show an elevated
`P(yes)` even though the full repaired trace returns to `No`.

## Reading The Heatmap

Rows are `case / variant`; columns are prefix depth `k`.

Useful patterns:

- A bright cell at `class_repaired, k=2` means class/subset wording creates a
  transient wrong-answer basin.
- A darker cell at `class_repaired, k=3` means the explicit conclusion repairs
  that basin.
- Bright cells in `bad_converse` indicate that rewind-like converse errors can
  become answer attractors.
- Stable dark rows for `direct_premises` and `direct_conclusion` are controls.

## Initial Local Smoke Read

A first local run with `llama-3.2-3b`, `mlx`, `mps`, `n=12`, and
`temps=0.0,0.4` saved:

```text
results/run_20260515_092142_syllogism_interference_matrix_v1
```

This smoke run should be treated as an observation, not yet a claim:

- At `temp=0.0`, `bad_converse` flipped all three nonce cases to `Yes`.
- At `temp=0.4`, `bad_converse` stayed high, with full-prefix yes rates between
  `0.92` and `1.00`.
- The original `class_repaired` transient did not reproduce deterministically in
  this expanded matrix; this makes the effect look more wording- and
  case-sensitive than the single-case pilot suggested.
- One direct valid-conclusion control, `dax_wug_mip / direct_conclusion`, also
  flipped to `Yes` at `temp=0.0`, while the two parallel nonce cases stayed
  dark. That is a useful warning flag: some interference may come from lexical
  or prompt-surface attractors, not only from the trace logic itself.

## Why This Is The Next Conference-Facing Step

The single-case result was vivid, but a conference claim needs controlled
replication. This matrix is intentionally small enough to inspect by hand while
still asking a sharper question:

```text
Is deterministic trace interference tied to one nonce-word example, or does it
survive across equivalent syllogisms and fixed trace variants?
```

If the class-language prefix repeatedly produces a wrong-answer basin, the paper
story becomes much stronger: trace wording can create local answer-space phase
transitions even in trivial closed-answer reasoning tasks.
