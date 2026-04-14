`sr_rewind_cot_assets/question_sets/` contains ready-to-run observation configs and reusable question packs.

Current files:

- `general_reasoning_observation_v1.yaml`
  A small English batch focused on general reasoning tasks that tend to expose:
  - conceptual compression
  - causal explanation structure
  - comparison vs summary collapse
  - forward/rewind asymmetry

Design goals for this set:

- Prefer questions that can produce several local explanatory steps instead of one-line factual answers.
- Avoid overly domain-specific prompts that mostly test recall.
- Keep prompts short enough to compare across models and settings.
- Bias toward tasks where a rewind operator may either recover prerequisites or collapse into a semantic core.

Usage:

```bash
python3 sr_rewind_cot.py run --config sr_rewind_cot_assets/question_sets/general_reasoning_observation_v1.yaml
```

You will usually want to edit the backend block first so it matches your local model path or API endpoint.
