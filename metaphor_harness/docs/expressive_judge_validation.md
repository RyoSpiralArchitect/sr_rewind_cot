# Expressive Judge Validation

This validation path reuses the 72 existing expressive generations from
`runs.openai.expressive_grid1.sqlite`. The updated judge prompt versions are
part of the audit IDs, so rerunning the harness schedules fresh audits without
regenerating samples.

## Dry Run

```bash
python3 -m metaphor_harness run \
  --cases data/expressive_seeds.jsonl \
  --config config/providers.openai.json \
  --db runs.openai.expressive_grid1.sqlite \
  --samples 3 \
  --temperatures 0.2,0.8 \
  --arms metaphor_with_forbidden,metaphor_without_forbidden,stance_explicit_metaphor \
  --mapping-visibility hidden \
  --no-quality \
  --dry-run
```

Expected result:

```text
planned_new_generations: 0
planned_new_audit_calls: 432
j_meta: judge_metaphor_integrity_v3
j_lit: judge_literary_v2
j_humor: judge_humor_v2
```

## Refresh Audits

```bash
python3 -m metaphor_harness run \
  --cases data/expressive_seeds.jsonl \
  --config config/providers.openai.json \
  --db runs.openai.expressive_grid1.sqlite \
  --samples 3 \
  --temperatures 0.2,0.8 \
  --arms metaphor_with_forbidden,metaphor_without_forbidden,stance_explicit_metaphor \
  --mapping-visibility hidden \
  --no-quality \
  --concurrency 6
```

## Regenerate Human-Gold Report

```bash
python3 -m metaphor_harness report \
  --db runs.openai.expressive_grid1.sqlite \
  --out reports/openai_expressive_grid1_human_v2 \
  --human-labels reports/openai_expressive_grid1/human_gold_filled_clean.csv
```

The report filters stale audit prompt versions. Old `judge_literary` and
`judge_humor` rows can remain in the DB, but current summaries use
`judge_metaphor_integrity_v3`, `judge_literary_v2`, and `judge_humor_v2`.
