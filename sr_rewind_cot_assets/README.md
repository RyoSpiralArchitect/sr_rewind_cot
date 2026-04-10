`sr_rewind_cot_assets/prompts/` contains the external English prompt assets used by [sr_rewind_cot.py](/Users/ryospiralarchitect/SpiralReality/sr_rewind_cot.py).

Layout:
- `prompts/*.txt`: legacy `v1` flat templates kept for backward compatibility.
- `prompts/v2/general_reasoning/*.txt`: shared v2 templates.
- `prompts/v2/<family>/_guidance.txt`: family-specific guidance injected into the shared v2 templates.

Built-in v2 families:
- `general_reasoning`
- `math_reasoning`
- `logic_reasoning`
- `story_reasoning`
- `strict_answer_format`

Current v2 templates:
- `trace_masked.txt`
- `trace_unmasked.txt`
- `trace_masked_text.txt`
- `trace_unmasked_text.txt`
- `reanswer_from_prefix.txt`
- `rewind_step.txt`
- `rewind_step_text.txt`
- `rewind_answer_forward.txt`
- `rewind_answer_reverse.txt`
- `bridge_answer.txt`
- `bridge_middle.txt`

Selection:
- CLI: `--prompt-version v2 --prompt-family math_reasoning`
- YAML:
  `prompt_version: v2`
  `prompt_family: math_reasoning`

Fallback behavior:
- `v1` always uses the flat legacy templates.
- `v2` first looks for a template inside the requested family folder, then falls back to `v2/general_reasoning/`.
