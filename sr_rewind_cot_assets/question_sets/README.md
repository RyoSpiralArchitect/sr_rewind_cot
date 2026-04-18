`sr_rewind_cot_assets/question_sets/` contains ready-to-run observation configs and reusable question packs.

Current files:

- `general_reasoning_observation_v1.yaml`
  Backward-compatible main batch. It currently tracks the faster tuned profile and is
  kept for convenience if you already have scripts pointing at the old filename.

- `general_reasoning_observation_v1_fast.yaml`
  Fast observation profile for iterative work. It uses a shorter rewind answer budget
  and a pyramid tail/oracle sample schedule to cut decode cost sharply.

- `general_reasoning_observation_v1_full.yaml`
  Higher-fidelity observation profile for slower but more conservative rewind analysis.
  It keeps rewind tail/oracle sampling flat and allows a longer rewind answer budget.

- `general_reasoning_observation_v1.yaml`, `general_reasoning_observation_v1_fast.yaml`, and `general_reasoning_observation_v1_full.yaml`
  all use the same six English general-reasoning questions, chosen to expose:
  - conceptual compression
  - causal explanation structure
  - comparison vs summary collapse
  - forward/rewind asymmetry

- `general_reasoning_observation_v1_text.yaml`
  Text-mode sibling of the main observation batch. It keeps the same six questions
  and nearly the same settings, but switches `trace_format` to `text` and gives
  the backend a slightly larger generation budget so open-ended traces are less
  likely to truncate mid-answer.

- `general_reasoning_pascal_text_reconstructed_20260409.yaml`
  A single-question reconstruction of the earlier text-trace experiment recovered
  from `results/run_20260409_070922/`. This preserves the pre-JSON `general_reasoning`
  setup closely enough to rerun or debug the old behavior.

Design goals for this set:

- Prefer questions that can produce several local explanatory steps instead of one-line factual answers.
- Avoid overly domain-specific prompts that mostly test recall.
- Keep prompts short enough to compare across models and settings.
- Bias toward tasks where a rewind operator may either recover prerequisites or collapse into a semantic core.

Usage:

```bash
python3 sr_rewind_cot.py run --config sr_rewind_cot_assets/question_sets/general_reasoning_observation_v1.yaml
```

```bash
python3 sr_rewind_cot.py run --config sr_rewind_cot_assets/question_sets/general_reasoning_observation_v1_fast.yaml
```

```bash
python3 sr_rewind_cot.py run --config sr_rewind_cot_assets/question_sets/general_reasoning_observation_v1_full.yaml
```

```bash
python3 sr_rewind_cot.py run --config sr_rewind_cot_assets/question_sets/general_reasoning_observation_v1_text.yaml
```

```bash
python3 sr_rewind_cot.py run --config sr_rewind_cot_assets/question_sets/general_reasoning_pascal_text_reconstructed_20260409.yaml
```

You will usually want to edit the backend block first so it matches your local model path or API endpoint.
