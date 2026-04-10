# sr_rewind_cot

`sr_rewind_cot` is a research harness for studying what happens when you try to run chain-of-thought backward.

Instead of treating reasoning only as a path to the correct answer, this project asks a different question:

- What parts of a forward reasoning trace survive reversal?
- Where does a rewind operator collapse into a semantic core or fixed point?
- Which tasks produce stable step structure, and which tasks compress into summary-like attractors?

The goal is not only accuracy. The goal is to observe the shape of reasoning under reversal.

## Core Idea

For a question, the harness can:

- generate a forward trace (`CoT`)
- re-answer from partial forward prefixes
- recover earlier steps from the answer backward (`Rewind-CoT`)
- compare forward and rewind structure
- probe bridge reconstruction between forward prefixes and rewind tails
- log and plot where the process becomes stable, collapses, or converges to a core

This makes it useful for experiments about:

- reverse reasoning
- semantic attractors
- fixed points in generated thought
- trace preservation vs compression
- differences between answer-space stability and step-space stability

## What It Can Do

- Run local Hugging Face models with `torch`, or prefer `MLX` first on Apple Silicon and fall back to `torch`.
- Run OpenAI-compatible HTTP backends such as local vLLM, Ollama, LiteLLM-compatible endpoints, or hosted APIs.
- Use prompt families and prompt versions, including `v2` task families like `general_reasoning`, `math_reasoning`, `logic_reasoning`, `story_reasoning`, and `strict_answer_format`.
- Build traces in JSON or plain text, with `auto` selection depending on whether streaming is enabled.
- Generate multiple forward trace candidates and rerank them with a pseudo Process Reward Model (`PRM-lite`) before running downstream rewind analysis.
- Optionally rerank multiple rewind-step candidates with the same pseudo Process Reward Model idea, so forward and reverse reasoning can be compared under matched heuristics.
- Stream the forward trace first, then stream rewind recovery.
- Save raw generations and prompt/output logs as JSONL for inspection.
- Save atomic `step_records` so steps can be analyzed as structured claims instead of only flat strings.
- Compute forward stability curves, entropy, divergence, rewind novelty, fixed-point depth, and core-strength style metrics.
- Compare `base` vs `PRM-selected` traces on both the forward and rewind side, including a dedicated `base rewind` vs `PRM rewind` axis.
- Record stage timings and rewind workload counts so bottlenecks can be read directly from `summary.csv`.
- Plot forward, rewind, bridge, trace-vs-rewind similarity, rewind novelty, rewind axes, and base-vs-PRM-vs-rewind comparisons.

## Research Framing

This repository is best thought of as an experimental instrument, not a polished benchmark suite.

It is especially good for questions like:

- Does rewind reconstruct an earlier prerequisite, or just restate a compressed summary?
- Does the model preserve causal or logical order, or collapse into a generic explanatory sentence?
- Is a stable rewind result a genuine core concept, or only a repeated paraphrase?

In many runs, "failure" is still informative. A rewind operator that falls into the same sentence over and over may be revealing a semantic fixed point rather than simply breaking.

## Repository Layout

- [sr_rewind_cot.py](./sr_rewind_cot.py): main CLI and research harness
- [sr_rewind_cot_assets/prompts](./sr_rewind_cot_assets/prompts): external prompt templates and task-family guidance
- [tests/test_sr_rewind_cot.py](./tests/test_sr_rewind_cot.py): targeted regression tests for parsing, rewind behavior, streaming, and metrics

## Installation

Minimum dependencies:

- Python 3.10+
- `requests`
- `matplotlib` for plots
- `pyyaml` for config-based runs

For local Hugging Face runs:

- `transformers`
- `torch`

Optional on Apple Silicon:

- `mlx`
- `mlx_lm`

Example:

```bash
pip install requests matplotlib pyyaml transformers torch
```

If you want MLX support on Apple Silicon, install the compatible `mlx` / `mlx_lm` stack separately.

## Quick Start

### 1. Local model, streamed CoT then rewind

```bash
python3 sr_rewind_cot.py quick-hf \
  --model "./model/llama-3.2-3b" \
  --device "mps" \
  --hf-backend "auto" \
  --dtype "float16" \
  --name "llama-3.2-it" \
  --question "What is the fundamental difference between Pascal's Wager and agnostic theism?" \
  --prompt-version v2 \
  --prompt-family general_reasoning \
  --trace-format auto \
  --stream-output \
  --save-generation-log-jsonl \
  --save-raw-generations
```

### 2. Local model, more structured logic run

```bash
python3 sr_rewind_cot.py quick-hf \
  --model "./model/llama-3.2-3b" \
  --device "mps" \
  --hf-backend "auto" \
  --dtype "float16" \
  --name "llama-3.2-it" \
  --question "Three suspects A, B, and C each make one statement. Exactly one statement is false. Determine whose statement is false." \
  --prompt-version v2 \
  --prompt-family logic_reasoning \
  --trace-format json \
  --save-generation-log-jsonl \
  --save-raw-generations
```

### 3. Multi-backend or multi-question run from config

```bash
python3 sr_rewind_cot.py sample-config > experiment.yaml
python3 sr_rewind_cot.py run --config experiment.yaml
```

### 4. Re-plot an existing run

```bash
python3 sr_rewind_cot.py plot --run-dir results/run_YYYYMMDD_HHMMSS
```

## Outputs

Each run writes a timestamped directory under `results/`.

Common outputs include:

- `summary.csv`: one-row-per-question summary metrics
- `<backend>__<qid>.json`: full structured result
- `<backend>__<qid>__generations.jsonl`: prompt/output log when enabled
- `<backend>__<qid>__trace_vs_rewind.jsonl`: pairwise forward-vs-rewind comparison rows
- `<backend>__<qid>__trace_axes.jsonl`: forward `base / PRM / rewind` alignment rows
- `<backend>__<qid>__rewind_axes.jsonl`: `base rewind` vs `PRM rewind` alignment rows
- `plots/`: generated figures

Typical plots include:

- forward prefix match-rate and entropy
- rewind vs oracle-tail comparison
- trace-vs-rewind similarity
- rewind novelty and fixed-point depth
- bridge heatmaps and core prototype plots

## Important Metrics

- `baseline_confidence`: how concentrated the full-trace answer distribution is
- `baseline_entropy_bits`: uncertainty in the baseline answer distribution
- `rewind_fixed_point_depth`: first rewind depth where novelty falls below the threshold
- `rewind_core_strength`: a compact measure of how strongly rewind has collapsed into a recurring core
- `trace_vs_rewind_preservation_rate`: how much of the forward trace survives rewind alignment
- `rewind_axis_base_prm_preservation_rate`: how different the raw rewind path is from the PRM-guided rewind path
- `time_rewind_total_s` and `rewind_total_generation_calls`: where rewind runtime is actually being spent

These metrics are most useful together. A run can have unstable answers but still show a very strong semantic attractor in rewind space.

## Prompt Families

`v2` prompt families let you bias the harness toward different task types:

- `general_reasoning`
- `math_reasoning`
- `logic_reasoning`
- `story_reasoning`
- `strict_answer_format`

Templates live under [sr_rewind_cot_assets/prompts](./sr_rewind_cot_assets/prompts). They are intentionally external so prompt behavior can be edited without changing core code.

## Current Direction

The project is moving toward a clearer split between:

- human-readable streamed reasoning
- structured atomic traces for measurement
- rewind operators with novelty constraints
- better detection of semantic fixed points vs genuine earlier-step recovery

In short: this repo aims to become a practical lab for studying reverse chain-of-thought as a dynamical system.

## Notes

- Accuracy is not the only success condition.
- Some of the most interesting runs are the ones where rewind collapses.
- Comparative, causal, logical, and algorithmic tasks usually produce more informative rewind behavior than one-line factual prompts.
