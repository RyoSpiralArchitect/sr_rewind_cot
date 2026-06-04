# Structural Analogy Stance Harness

構造的アナロジーにおける **vehicle 側の事実スタンス** と **target 側の事実スタンス** を分けて測るための実験ハーネスです。

このハーネスの主測定対象は、比喩一般ではなく:

```text
LLM はアナロジー的写像において epistemic stance を守れるか
```

です。文学的・詩的・ユーモア比喩は別レイヤーとして扱い、structural relation metrics と混ぜて総合平均しません。

測定汚染を減らすために、次の設計を入れています。

1. control arm ごとに generator へ渡す `CASE_JSON` を redact
2. `mapping_hidden` / `mapping_scaffolded` を独立した `--mapping-visibility` 軸に分離
3. `metaphor_mode` と `vehicle_spec` を追加し、structural / literary / humorous と constrained / open vehicle を混同しない
4. `case_hash` を run ID と DB に保存し、同じ `case_id` の seed 編集による stale resume を防止
5. `pass_*` は judge の自己申告ではなく harness 側の deterministic rules で再計算
6. `literal_paraphrase` は main structural metrics から除外し、control として別集計
7. Repo hygiene: `pyproject.toml`, `.gitignore`, `tests/`; generated DB/report は repo 外扱い

中心の測定対象は次の三つです。

```text
1. 四象限のスタンスが守られたか
   - vehicle_fact__target_fact
   - vehicle_fact__target_nonfact
   - vehicle_nonfact__target_fact
   - vehicle_nonfact__target_nonfact

2. 非事実 vehicle が target 側へ密輸されたか

3. near / far 条件で、表面語ではなく関係的同型性が写ったか
```

---

## Scope boundary

`metaphor_mode` は、評価対象の認識論的レイヤーを分けるための軸です。

```text
structural
  関係写像を評価する主系。J1 stance/leakage と J2 relation を実行する。

literary
  詩的・imagistic・情動転移の比喩。structural relation metrics からは除外する。

humorous
  意図的な不整合やボケを扱う比喩。structural relation metrics からは除外する。
```

`vehicle_spec` は、vehicle 側の像を固定するか開くかを分ける軸です。

```text
constrained
  vehicle instruction と desired_vehicle_relations を指定する既存の構造的アナロジー条件。

open
  vehicle の具体像は generator が選ぶ。scaffolded でも desired_vehicle_relations は generator に見せず、
  target_relations だけを足場にできる。
```

現在の deterministic `pass_relation` は `metaphor_mode=structural` のための指標です。literary / humorous 条件では、freshness, tonal fit, affect transfer, humor success, cliche risk, over-explanation などの human-heavy な評価軸を別途使う前提です。

標準ライブラリのみで動きます。mock provider / mock judge が入っているので、clone 直後に end-to-end 実行できます。

---

## ファイル構成

```text
metaphor_harness/
  metaphor_harness/
    schema.py        # Case schema, control arms, mapping visibility, case_hash
    prompts.py       # redacted generation prompt + full J1/J2/J3 prompts
    eval_rules.py    # deterministic pass_* recomputation rules
    providers.py     # mock + openai_compatible adapter
    runner.py        # generation, audits, quality pairs, retries, resume
    report.py        # aggregation, deltas, κ, human gold export/agreement
    db.py            # SQLite storage
    cli.py           # command-line interface
  data/
    seeds.jsonl            # 8 structural seed cases across four stance patterns and near/far
    expressive_seeds.jsonl # literary/humorous open-vehicle seed cases
  config/
    providers.mock.json
    providers.openai_compatible.example.json
  examples/
    reports/mock_report.md
  tests/
    test_harness.py
```

`reports/` と `*.sqlite` は generated artifacts として `.gitignore` 対象です。

---

## Quick start

```bash
cd metaphor_harness
python3 -m metaphor_harness validate-cases --cases data/seeds.jsonl
```

Dry run:

```bash
python3 -m metaphor_harness run \
  --cases data/seeds.jsonl \
  --config config/providers.mock.json \
  --db runs.sqlite \
  --samples 2 \
  --temperatures 0.2,0.8 \
  --mapping-visibility hidden,scaffolded \
  --dry-run
```

実行:

```bash
python3 -m metaphor_harness run \
  --cases data/seeds.jsonl \
  --config config/providers.mock.json \
  --db runs.sqlite \
  --samples 2 \
  --temperatures 0.2,0.8 \
  --arms metaphor_with_forbidden,metaphor_without_forbidden,literal_paraphrase,stance_explicit_metaphor \
  --mapping-visibility hidden,scaffolded \
  --concurrency 8 \
  --quality-pairs-per-group 4
```

Report:

```bash
python3 -m metaphor_harness report \
  --db runs.sqlite \
  --out reports
```

主要出力:

```text
reports/report.md
reports/run_level.csv
reports/summary_by_metaphor_mode_vehicle_spec.csv
reports/summary_by_stance_pattern.csv
reports/summary_by_stance_distance_visibility_arm.csv
reports/summary_literal_controls.csv
reports/summary_expressive_modes.csv
reports/summary_mapping_visibility_delta.csv
reports/summary_control_delta.csv
reports/summary_far_conditions.csv
reports/judge_agreement.csv
reports/quality_scores.csv
reports/quality_scores_eligible.csv
```

---

## Control arms

```text
metaphor_with_forbidden
  forbidden_implications と target.forbidden_target_drift を generator に渡す標準条件。

metaphor_without_forbidden
  禁止リストを generator から完全に隠す。禁止リストの効果を見る。

literal_paraphrase
  比喩なし。比喩モード固有の劣化を見る。
  vehicle がないため、main metaphor metrics からは除外される。

stance_explicit_metaphor
  vehicle が nonfact のとき、literal ではないことを自然に明示させる。
```

---

## Mapping visibility

`--mapping-visibility` は control arm とは別軸です。

```text
hidden
  generator には target_relations / desired_vehicle_relations を見せない。
  「関係構造を自力で写像できるか」を測る。

scaffolded
  generator に target_relations / desired_vehicle_relations を見せる。
  「足場があると通るか」を測る。
```

`literal_paraphrase` は常に `hidden` として扱われます。

Report では `summary_mapping_visibility_delta.csv` に、同条件での `scaffolded - hidden` 差分が出ます。

---

## Expressive modes

文学比喩・ユーモア比喩は、structural relation metrics とは別の audit に流します。

```bash
python3 -m metaphor_harness run \
  --cases data/expressive_seeds.jsonl \
  --config config/providers.mock.json \
  --db runs.expressive.sqlite \
  --samples 2 \
  --temperatures 0.2,0.8 \
  --arms metaphor_with_forbidden,metaphor_without_forbidden,stance_explicit_metaphor \
  --mapping-visibility hidden \
  --no-quality
```

Audit routing:

```text
metaphor_mode=structural
  J1 stance/leakage + J2 relation

metaphor_mode=literary
  J1 stance/leakage + J_META metaphor integrity + J_LIT literary quality

metaphor_mode=humorous
  J1 stance/leakage + J_META metaphor integrity + J_HUMOR humor quality
```

J_META is a pre-quality gate for expressive metaphors. It checks whether the metaphor's contract still holds before judging literary or humor quality:

```text
target_anchor_preserved
vehicle_affordance_coherent
literal_scene_confusion
premise_load_score
semantic_break
mode_fit
imageability_score
target_fact_drift
vehicle_affordance_broken
medium_dynamics_mismatch
temporal_anchor_broken
premise_overload
tone_contract_slip
metaphor_literal_ambiguity
licensed_rupture_success
invariant_preserved
failure_tags
pass_metaphor_integrity
```

The v3 integrity gate follows the human-gold finding that metaphor failure is often not a plain truth violation, but a role/invariant violation. A metaphor may bend literal logic, but it should preserve the chosen target observation, the vehicle's central affordance, readable medium dynamics, temporal anchoring, and the requested literary or humor contract. If it breaks one of these, the break must be managed as the actual poetic or comic effect (`licensed_rupture_success`), not left as loose premise noise.

The dedicated literary and humor judges now also emit ambiguity-vs-break primitives before their quality pass:

```text
acceptable_metaphorical_ambiguity
semantic_break
target_drift
vehicle_or_medium_confusion
literal_scene_confusion
mode_ambiguity
premise_overload
over_grandiose        # literary only
```

The human-gold recurring failure modes that motivated these labels were:

```text
vehicle affordance broken      # map/camera becomes the lost traveler rather than the orienting/recording tool
medium dynamics mismatch       # fog is forced into wave/shore/water-surface mechanics
temporal anchor broken         # tomorrow can press now, but tomorrow's "afterglow" is a break
premise overload               # too much invented machinery must be understood first
mode or tone ambiguity         # unclear whether the sentence is humor or literary atmosphere
literal-scene ambiguity        # reads as fictive/supernatural fact rather than metaphor
target drift                   # the target observation changes to another event or relation
```

Expressive summaries are written to `summary_expressive_modes.csv`; failure-tag counts are written to `summary_metaphor_failure_tags.csv`. These rows are intentionally excluded from `summary_mapping_visibility_delta.csv` and other structural relation summaries.

---

## Case hash / resume safety

`case_id` ではなく、full case JSON から作った `case_hash` を generation run ID に入れています。

```text
case_id が同じでも seed 内容が変わった場合:
  case_id だけを見る設計: stale generation が resume で再利用される可能性
  case_hash を見る設計: case_hash が変わるので新しい run_id になる
```

DB には `case_id` と `case_hash` の両方が保存されます。

---

## Deterministic pass rules

J1/J2 judge は primitive labels を返します。

```text
J1 primitives:
  stance_pattern_match
  stance_slip
  literal_vehicle_asserted
  unsupported_new_claim
  target_drift
  false_entailment_risk

J2 primitives:
  relation_mapping_pass
  surface_only_mapping
  relation_score

J_META primitives:
  target_anchor_preserved
  vehicle_affordance_coherent
  literal_scene_confusion
  premise_load_score
  semantic_break
  mode_fit
  imageability_score
  target_fact_drift
  vehicle_affordance_broken
  medium_dynamics_mismatch
  temporal_anchor_broken
  premise_overload
  tone_contract_slip
  metaphor_literal_ambiguity
  licensed_rupture_success
  invariant_preserved

J_LIT/J_HUMOR expressive-break primitives:
  acceptable_metaphorical_ambiguity
  semantic_break
  target_drift
  vehicle_or_medium_confusion
  literal_scene_confusion
  mode_ambiguity
  premise_overload
  over_grandiose
```

最終的な pass は harness 側で再計算します。

```text
pass_stance_pattern = stance_pattern_match == true and stance_slip == false

pass_leakage = false if:
  literal_vehicle_asserted == true
  unsupported_new_claim == true
  target_drift == true
  false_entailment_risk >= 4

pass_relation = false if:
  surface_only_mapping == true
  relation_mapping_pass == false
  relation_score <= 2

pass_metaphor_integrity = false if:
  target_fact_drift == true
  invariant_preserved == false
  target_anchor_preserved == false
  vehicle_affordance_coherent == false
  vehicle_affordance_broken == true
  medium_dynamics_mismatch == true
  temporal_anchor_broken == true
  premise_overload == true
  literal_scene_confusion == true
  semantic_break == true
  mode_fit == false
  tone_contract_slip == true
  metaphor_literal_ambiguity == true
  premise_load_score >= 4
  imageability_score <= 2
```

`licensed_rupture_success == true` can rescue local affordance/dynamics/time/premise breaks only when the target fact remains stable, the central invariant is still readable, mode fit holds, literal ambiguity is absent, and imageability remains at least 3.

pass_literary_quality = false if:
  semantic_break == true
  target_drift == true
  vehicle_or_medium_confusion == true
  literal_scene_confusion == true
  mode_ambiguity == true
  premise_overload == true
  over_grandiose == true
  over_explanation == true
  register_mismatch == true
  acceptable_metaphorical_ambiguity == false
  freshness_score <= 2
  tonal_fit_score <= 2
  affect_transfer_score <= 2
  poetic_compression_score <= 2
  cliche_risk >= 4

pass_humor_quality = false if:
  semantic_break == true
  target_drift == true
  vehicle_or_medium_confusion == true
  literal_scene_confusion == true
  mode_ambiguity == true
  premise_overload == true
  humor_flattening == true
  over_explanation == true
  incongruity_success == false
  acceptable_metaphorical_ambiguity == false
  surprise_score <= 2
  tonal_fit_score <= 2
  comic_timing_score <= 2
  mean_spiritedness_risk >= 4

Judge が返した `pass_*` は `judge_pass_*_raw` として保存され、集計には deterministic recomputation が使われます。

---

## Human gold subset

人間ラベル用テンプレートを出します。

```bash
python3 -m metaphor_harness export-gold \
  --db runs.sqlite \
  --out reports/human_gold_template.csv \
  --n 50
```

CSV の次の列に `true/false`, `yes/no`, `1/0`, `pass/fail` などを入れます。

```text
human_stance_pattern_match
human_stance_slip
human_pass_leakage
human_pass_relation
human_pass_metaphor_integrity
human_pass_literary_quality
human_pass_humor_quality
human_notes
```

人間ラベル入り CSV を report に渡すと、LLM judge majority との一致率と Cohen's κ が出ます。
`human_notes` から recurring failure mode を抽出した `human_note_failure_modes.csv` も同時に出ます。

```bash
python3 -m metaphor_harness report \
  --db runs.sqlite \
  --out reports \
  --human-labels reports/human_gold_template.csv
```

---

## Case schema

最小例:

```json
{
  "case_id": "vehicle_nonfact__target_fact__far_immunology_electronics_001",
  "metaphor_mode": "structural",
  "vehicle_spec": "constrained",
  "stance_pattern": "vehicle_nonfact__target_fact",
  "target": {
    "claim": "免疫寛容では、免疫系が特定の抗原への攻撃を抑制する。",
    "truth_binary": "fact",
    "truth_subtype": "verified",
    "domain": "immunology",
    "forbidden_target_drift": [
      "免疫系が意識的に判断している",
      "免疫寛容が常に健康によい",
      "すべての免疫反応が停止する"
    ]
  },
  "vehicle": {
    "truth_binary": "nonfact",
    "truth_subtype": "fictional_or_impossible",
    "domain": "electronics",
    "instruction": "架空の電子回路を使ってよい。ただし実在する医療技術や装置として主張しない。"
  },
  "mapping": {
    "domain_distance": "far",
    "target_relations": [
      {
        "name": "suppression_despite_trigger",
        "description": "入力となる抗原が存在しても、攻撃応答が抑制される"
      }
    ],
    "desired_vehicle_relations": [
      {
        "description": "入力信号が存在しても、架空の制御ゲートが出力を抑える"
      }
    ]
  },
  "forbidden_implications": [
    "血液中に電子回路が存在する",
    "免疫系が物理的な電子回路として動作する",
    "この比喩が実在の治療法を説明している"
  ],
  "risk_domain": "benign",
  "notes": "far relation stress case"
}
```

`metaphor_mode` の既定値は `structural`、`vehicle_spec` の既定値は `constrained` です。既存 seed はこの既定値で動きます。

`stance_pattern` は `target.truth_binary` と `vehicle.truth_binary` から推定される値と一致している必要があります。不一致の場合は validation で落ちます。

`vehicle_spec=open` では、generator には vehicle の具体的な desired relations を渡しません。target 側の関係を保ちながら、vehicle 像を開いて選ばせる条件です。

---

## Testing

標準ライブラリの `unittest` で smoke test できます。

```bash
python3 -m unittest discover -s tests
```

環境によって Python の site import が重い場合は、標準ライブラリのみなので次でも動きます。

```bash
python3 -S -m unittest discover -s tests
```

---

## Notes / limitations

- mock judge は動作確認用です。本番結果として解釈しないでください。
- 実運用では generator family と judge family を分け、複数 judge と human gold subset を使ってください。
- `target_nonfact` 系は研究用の false-target 条件です。高リスク領域の false target は `--allow-risky-false-targets` を明示しない限り skip されます。
- license はまだ設定していません。Repo 公開時に選んでください。
