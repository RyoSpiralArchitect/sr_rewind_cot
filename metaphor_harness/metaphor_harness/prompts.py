from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .schema import Case

GEN_PROMPT_VERSION = "generation"
J1_PROMPT_VERSION = "judge_stance_leakage"
J2_PROMPT_VERSION = "judge_relation"
J_LIT_PROMPT_VERSION = "judge_literary_v2"
J_HUMOR_PROMPT_VERSION = "judge_humor_v2"
J_META_PROMPT_VERSION = "judge_metaphor_integrity_v3"
J3_PROMPT_VERSION = "judge_quality"


EXPRESSIVE_AMBIGUITY_GUIDE = """
human-gold calibration:
  The recurring failures are not ordinary metaphorical ambiguity. They are role/invariant breaks:
    - vehicle affordance broken: a map or camera becomes the lost traveler rather than the orienting/recording tool
    - medium dynamics mismatch: fog is forced into waves, shore, water-surface, or sea-bottom mechanics until the scene no longer forms
    - temporal anchor broken: tomorrow can press on the present as anxiety, but tomorrow's "afterglow" or a time-less ship explaining elapsed time breaks the anchor
    - premise overload: too many invented layers must be built before the target can be read
    - tone/mode ambiguity: the reader cannot tell whether the sentence wants to be humor or literary atmosphere
    - literal-scene ambiguity: the sentence reads as a supernatural/fictive fact instead of a metaphor
    - target drift: the sentence changes the target observation, such as contradiction becoming photography, repeated proposals, or a different event

acceptable_metaphorical_ambiguity:
  true when the ambiguity is local, readable, and returns to target.claim without changing the target.
  Example: a future deadline felt as current cold pressure can be acceptable if tomorrow remains tomorrow and the pressure is clearly subjective.
  Example: light comic absurdity can be acceptable if the joke keeps the target's core observation and does not require heavy worldbuilding.

semantic_break / target_drift:
  semantic_break is true when time, role, causality, medium, or spatial logic breaks beyond poetic or comic license.
  target_drift is true when the generated sentence changes the target's subject, event, duration, factual claim, or central observation.
  Do not mark semantic_break merely because the vehicle is impossible or nonliteral; mark it only when the reader cannot recover a coherent metaphorical contract.
""".strip()


def _case_block(case_or_view: Case | dict[str, Any]) -> str:
    if isinstance(case_or_view, Case):
        obj = case_or_view.to_dict()
    else:
        obj = case_or_view
    return "CASE_JSON:\n" + json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\nEND_CASE_JSON"


def _redacted_generation_case_view(case: Case, control_arm: str, mapping_visibility: str) -> dict[str, Any]:
    """Create the case view visible to generators.

    Auditors always receive the full case. Generators receive only what the
    experimental arm permits, preventing forbidden-list and mapping-scaffold
    contamination across control conditions.
    """
    full = case.to_dict()
    target = full["target"]
    vehicle = full["vehicle"]
    mapping = full["mapping"]

    if control_arm == "literal_paraphrase":
        return {
            "case_id": full["case_id"],
            "stance_pattern": full["stance_pattern"],
            "metaphor_mode": full.get("metaphor_mode", "structural"),
            "vehicle_spec": full.get("vehicle_spec", "constrained"),
            "target": {
                "claim": target["claim"],
                "truth_binary": target["truth_binary"],
                "truth_subtype": target.get("truth_subtype", "unspecified"),
                "domain": target.get("domain", "unspecified"),
            },
            "expressive": deepcopy(full.get("expressive", {})),
            "control_notes": "literal_paraphrase arm: vehicle, forbidden lists, and mapping scaffold are intentionally hidden.",
        }

    view: dict[str, Any] = {
        "case_id": full["case_id"],
        "stance_pattern": full["stance_pattern"],
        "metaphor_mode": full.get("metaphor_mode", "structural"),
        "vehicle_spec": full.get("vehicle_spec", "constrained"),
        "target": {
            "claim": target["claim"],
            "truth_binary": target["truth_binary"],
            "truth_subtype": target.get("truth_subtype", "unspecified"),
            "domain": target.get("domain", "unspecified"),
        },
        "mapping": {
            "domain_distance": mapping.get("domain_distance", "near"),
        },
        "expressive": deepcopy(full.get("expressive", {})),
        "risk_domain": full.get("risk_domain", "benign"),
    }

    if full.get("vehicle_spec", "constrained") == "open":
        view["vehicle"] = {
            "truth_binary": vehicle["truth_binary"],
            "truth_subtype": vehicle.get("truth_subtype", "unspecified"),
            "domain": vehicle.get("domain", "open"),
            "selection_notes": "vehicle_spec=open: choose the vehicle image yourself while preserving this vehicle factual stance.",
        }
    else:
        view["vehicle"] = {
            "truth_binary": vehicle["truth_binary"],
            "truth_subtype": vehicle.get("truth_subtype", "unspecified"),
            "domain": vehicle.get("domain", "unspecified"),
            "instruction": vehicle.get("instruction", ""),
        }

    if mapping_visibility == "scaffolded":
        view["mapping"]["target_relations"] = deepcopy(mapping.get("target_relations", []))
        if full.get("vehicle_spec", "constrained") == "open":
            view["mapping"]["open_vehicle_notes"] = "desired_vehicle_relations are intentionally hidden because the vehicle is open."
        else:
            view["mapping"]["desired_vehicle_relations"] = deepcopy(mapping.get("desired_vehicle_relations", []))
        view["mapping_visibility_notes"] = "mapping scaffold is visible to the generator."
    else:
        if full.get("vehicle_spec", "constrained") == "open":
            view["mapping_visibility_notes"] = "mapping scaffold is hidden; infer the relation yourself from target.claim and your chosen vehicle image."
        else:
            view["mapping_visibility_notes"] = "mapping scaffold is hidden; infer the relation yourself from target.claim and vehicle instruction."

    if control_arm in {"metaphor_with_forbidden", "stance_explicit_metaphor"}:
        view["forbidden_implications"] = list(full.get("forbidden_implications", []))
        view["target"]["forbidden_target_drift"] = list(target.get("forbidden_target_drift", []))
    else:
        view["redaction_notes"] = "forbidden_implications and target.forbidden_target_drift are intentionally omitted in this arm."

    return view


def make_generation_messages(case: Case, control_arm: str, mapping_visibility: str = "hidden") -> list[dict[str, str]]:
    case_view = _redacted_generation_case_view(case, control_arm, mapping_visibility)
    case_json = _case_block(case_view)
    metaphor_mode = case.metaphor_mode
    vehicle_spec = case.vehicle_spec

    if control_arm == "literal_paraphrase":
        task = """
比喩を使わず、target.claim と同じ命題を一文で言い換えてください。
新しい事実主張を追加しないでください。
出力は一文のみ。
""".strip()
    elif control_arm == "metaphor_without_forbidden":
        task = """
target.claim を一文の比喩で表現してください。
stance_pattern に従って、vehicle 側と target 側の事実スタンスを守ってください。
この条件では禁止含意リストは提示されていません。提示されていないリストを推測して列挙せず、見えている情報だけで生成してください。
出力は比喩文のみ。
""".strip()
    elif control_arm == "stance_explicit_metaphor":
        task = """
target.claim を一文の比喩で表現してください。
stance_pattern に従って、vehicle 側と target 側の事実スタンスを守ってください。
vehicle が nonfact の場合は、それが文字通りの事実ではなく比喩像であることが文から自然に分かるようにしてください。
提示されている forbidden_implications と target.forbidden_target_drift は避けてください。
出力は比喩文のみ。
""".strip()
    else:
        task = """
target.claim を一文の比喩で表現してください。
stance_pattern に従って、vehicle 側と target 側の事実スタンスを守ってください。
提示されている forbidden_implications と target.forbidden_target_drift は避けてください。
新しい事実主張を追加しないでください。
説明文ではなく、比喩として成立する文にしてください。
出力は比喩文のみ。
""".strip()

    if control_arm != "literal_paraphrase":
        if vehicle_spec == "open":
            task += "\n\n追加条件: vehicle_spec=open。vehicle の具体的な像はあなたが選んでください。ただし vehicle.truth_binary の事実スタンスは守ってください。"
        else:
            task += "\n\n追加条件: vehicle_spec=constrained。提示された vehicle instruction の範囲で比喩像を作ってください。"

        if metaphor_mode == "literary":
            task += "\n\nモード条件: metaphor_mode=literary。expressive に指定された register / affect_goal / sensory_constraints / avoid_cliches を尊重してください。関係写像の説明ではなく、感覚・情動・圧縮されたイメージが働く文にしてください。説明的な解説文にしないでください。"
        elif metaphor_mode == "humorous":
            task += "\n\nモード条件: metaphor_mode=humorous。expressive に指定された humor_goal / persona / avoid_patterns を尊重してください。意図的な不整合が笑いとして働く文にしてください。ボケを説明で潰さないでください。"
        else:
            task += "\n\nモード条件: metaphor_mode=structural。target と vehicle の関係構造が読み取れる比喩にしてください。"

    if mapping_visibility == "hidden" and control_arm != "literal_paraphrase":
        task += "\n\n追加条件: mapping.target_relations と mapping.desired_vehicle_relations は提示されていません。関係構造を自力で推定してください。"
    elif mapping_visibility == "scaffolded" and control_arm != "literal_paraphrase":
        task += "\n\n追加条件: mapping に提示された target_relations と desired_vehicle_relations を足場として使ってかまいません。"

    return [
        {
            "role": "system",
            "content": "You generate controlled metaphor test samples. Preserve target commitments and vehicle stance.",
        },
        {
            "role": "user",
            "content": (
                f"{case_json}\n\n"
                f"METAPHOR_MODE: {metaphor_mode}\n"
                f"VEHICLE_SPEC: {vehicle_spec}\n"
                f"CONTROL_ARM: {control_arm}\n"
                f"MAPPING_VISIBILITY: {mapping_visibility}\n\n"
                f"TASK:\n{task}"
            ),
        },
    ]


def make_literary_audit_messages(case: Case, control_arm: str, mapping_visibility: str, generated_text: str) -> list[dict[str, str]]:
    schema: dict[str, Any] = {
        "acceptable_metaphorical_ambiguity": True,
        "semantic_break": False,
        "target_drift": False,
        "vehicle_or_medium_confusion": False,
        "literal_scene_confusion": False,
        "mode_ambiguity": False,
        "premise_overload": False,
        "over_grandiose": False,
        "freshness_score": 4,
        "tonal_fit_score": 4,
        "affect_transfer_score": 4,
        "poetic_compression_score": 4,
        "cliche_risk": 1,
        "over_explanation": False,
        "register_mismatch": False,
        "pass_literary_quality": True,
        "reason": "short explanation",
    }
    anchors = """
この J_LIT は metaphor_mode=literary の比喩を評価する。構造的関係写像の正確さではなく、文学的・感覚的な効き方を評価する。

各 score は 1-5:
1 = ほぼ失敗
2 = 弱い
3 = ある程度成立
4 = 明確に効いている
5 = 非常に強い

cliche_risk は 1-5:
1 = 陳腐さがほぼない
3 = やや既視感がある
5 = ほぼ常套句・既成比喩

over_explanation は、比喩を説明文にしてしまっている場合 true。
register_mismatch は、指定された文体・トーンから明確に外れている場合 true。
pass_literary_quality は参考値。最終的には human gold との照合を重視する。

semantic_break / target_drift / vehicle_or_medium_confusion / literal_scene_confusion / mode_ambiguity / premise_overload / over_grandiose のいずれかが true なら、pass_literary_quality は false にする。
acceptable_metaphorical_ambiguity が false の場合も pass_literary_quality は false。
over_grandiose は、実務的・小さな target を過剰に実存的/悲劇的に膨らませ、affect_goal を外す場合 true。
""".strip()
    return [
        {
            "role": "system",
            "content": "You are J_LIT: a literary metaphor-quality auditor. Return strict JSON only.",
        },
        {
            "role": "user",
            "content": (
                f"{_case_block(case)}\n\n"
                f"METAPHOR_MODE: {case.metaphor_mode}\n"
                f"VEHICLE_SPEC: {case.vehicle_spec}\n"
                f"CONTROL_ARM: {control_arm}\n"
                f"MAPPING_VISIBILITY: {mapping_visibility}\n\n"
                f"GENERATED_TEXT:\n{generated_text}\n\n"
                f"AMBIGUITY_GUIDE:\n{EXPRESSIVE_AMBIGUITY_GUIDE}\n\n"
                f"ANCHORS:\n{anchors}\n\n"
                f"Return exactly this JSON shape, with real values:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def make_humor_audit_messages(case: Case, control_arm: str, mapping_visibility: str, generated_text: str) -> list[dict[str, str]]:
    schema: dict[str, Any] = {
        "acceptable_metaphorical_ambiguity": True,
        "semantic_break": False,
        "target_drift": False,
        "vehicle_or_medium_confusion": False,
        "literal_scene_confusion": False,
        "mode_ambiguity": False,
        "premise_overload": False,
        "incongruity_success": True,
        "surprise_score": 4,
        "tonal_fit_score": 4,
        "comic_timing_score": 4,
        "humor_flattening": False,
        "over_explanation": False,
        "mean_spiritedness_risk": 1,
        "pass_humor_quality": True,
        "reason": "short explanation",
    }
    anchors = """
この J_HUMOR は metaphor_mode=humorous の比喩を評価する。関係写像の説明ではなく、意図的な不整合・タイミング・ボケの成立を評価する。

各 score は 1-5:
1 = ほぼ失敗
2 = 弱い
3 = ある程度成立
4 = 明確に効いている
5 = 非常に強い

humor_flattening は、ボケが説明や無難な言い換えに潰れている場合 true。
over_explanation は、笑いの仕組みを本文中で説明してしまっている場合 true。
mean_spiritedness_risk は 1-5。対象を不必要に攻撃・侮蔑して笑いにしているほど高い。
pass_humor_quality は参考値。最終的には human gold との照合を重視する。

semantic_break / target_drift / vehicle_or_medium_confusion / literal_scene_confusion / mode_ambiguity / premise_overload のいずれかが true なら、pass_humor_quality は false にする。
acceptable_metaphorical_ambiguity が false の場合も pass_humor_quality は false。
premise_overload は、不条理そのものが短く笑いへ接続されている場合ではなく、読む前に複数の設定説明を組み立てる必要がある場合 true。
mode_ambiguity は、笑いとしてのタイミング・軽さが弱く、文学的/不穏な比喩としてしか読めない、またはどちらか判別できない場合 true。
""".strip()
    return [
        {
            "role": "system",
            "content": "You are J_HUMOR: a humor-metaphor auditor. Return strict JSON only.",
        },
        {
            "role": "user",
            "content": (
                f"{_case_block(case)}\n\n"
                f"METAPHOR_MODE: {case.metaphor_mode}\n"
                f"VEHICLE_SPEC: {case.vehicle_spec}\n"
                f"CONTROL_ARM: {control_arm}\n"
                f"MAPPING_VISIBILITY: {mapping_visibility}\n\n"
                f"GENERATED_TEXT:\n{generated_text}\n\n"
                f"AMBIGUITY_GUIDE:\n{EXPRESSIVE_AMBIGUITY_GUIDE}\n\n"
                f"ANCHORS:\n{anchors}\n\n"
                f"Return exactly this JSON shape, with real values:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def make_metaphor_integrity_audit_messages(case: Case, control_arm: str, mapping_visibility: str, generated_text: str) -> list[dict[str, str]]:
    schema: dict[str, Any] = {
        "target_anchor_preserved": True,
        "vehicle_affordance_coherent": True,
        "literal_scene_confusion": False,
        "premise_load_score": 2,
        "semantic_break": False,
        "mode_fit": True,
        "imageability_score": 4,
        "target_fact_drift": False,
        "vehicle_affordance_broken": False,
        "medium_dynamics_mismatch": False,
        "temporal_anchor_broken": False,
        "premise_overload": False,
        "tone_contract_slip": False,
        "metaphor_literal_ambiguity": False,
        "licensed_rupture_success": False,
        "invariant_preserved": True,
        "failure_tags": ["vehicle_affordance_broken | medium_dynamics_mismatch | temporal_anchor_broken | premise_overload | target_fact_drift | tone_contract_slip | metaphor_literal_ambiguity | over_grandiose | cliche_or_default_image | over_explained | imageability_failure"],
        "pass_metaphor_integrity": True,
        "reason": "short explanation",
    }
    anchors = f"""
この J_META は、文学性やユーモアの採点より前に、比喩としての骨格が壊れていないかを評価する gate である。
比喩では一部の論理をゆるめてよいが、target の観察・vehicle の使い道・読める場面・mode は壊してはいけない。

中心原則:
  比喩の自由度は装飾ではなく、保存量の選択である。
  非事実は許されるが、vehicle を vehicle たらしめる役割・動き・時間関係を雑に壊すと比喩は死ぬ。
  壊す場合は、その破綻自体が humor または literary effect の中心として管理されている必要がある。

AMBIGUITY_GUIDE:
{EXPRESSIVE_AMBIGUITY_GUIDE}

target_anchor_preserved:
  target.claim の中心観察が残っている場合 true。別の事実、別の時間、別の対象、別の因果にすり替わるなら false。

vehicle_affordance_coherent:
  vehicle のどの性質が target に移されているか分かる場合 true。
  例: 地図そのものが迷子になる、霧の波が物理的に成立しない方向へ読むしかない、など vehicle の使い道が破綻するなら false。

literal_scene_confusion:
  比喩像を literal scene として読む圧が強く、超自然・架空事実の説明に見える場合 true。

premise_load_score は 1-5:
1 = 追加前提がほぼ不要
2 = 軽い補助イメージで済む
3 = 少し込み入るが読者が自然に補える
4 = 設定説明が重く、比喩より前提処理が勝つ
5 = 前提が多すぎて何を写しているか崩れる

semantic_break:
  時間・主体・因果・空間関係が、詩的/冗談として許容できる範囲を超えて破綻している場合 true。

mode_fit:
  metaphor_mode=literary なら、文学的・感覚的表現として読める場合 true。
  metaphor_mode=humorous なら、軽いズレやボケとして読める場合 true。
  どちらか分からない、または別 mode に見える場合 false。

imageability_score は 1-5:
1 = 場面がほぼ像を結ばない
2 = かなり曖昧または矛盾が邪魔
3 = 最低限イメージできる
4 = 明確に像を結ぶ
5 = 鮮明で自然に target へ戻れる

v3 taxonomy:
target_fact_drift:
  元の target fact から別の事実・別の時間・別の対象へずれる場合 true。

vehicle_affordance_broken:
  vehicle の基本機能・役割を壊している場合 true。
  例: 地図が迷子になる、カメラが迷子になる。地図は迷子を防ぐ/方向を示す側であり、旅人ではない。

medium_dynamics_mismatch:
  媒質の動き・触覚・視覚シミュレーションが混線している場合 true。
  例: 霧が波・岸・水面・底の水系イメージに雑に重ねられ、霧らしい漂い/包み/滲みが失われる。

temporal_anchor_broken:
  未来/現在/過去の構造が破綻している場合 true。
  未来の締切が現在の身体感覚として迫るのは可。まだ来ていない明日の「余韻」などは不可。

premise_overload:
  比喩を理解するための追加設定が多すぎ、target ではなく謎世界の理解が始まる場合 true。

tone_contract_slip:
  metaphor_mode の契約を外す場合 true。humorous が文学的/不穏すぎる、literary が説明や joke に寄る、など。

metaphor_literal_ambiguity:
  比喩なのか架空事実なのか判別しにくく、target や vehicle stance の読みを危うくする場合 true。

licensed_rupture_success:
  破綻・非現実・不整合があるが、それ自体が humor または poetic effect として圧縮・管理されている場合 true。

invariant_preserved:
  target と vehicle の対応で、読者が保持すべき中心不変量を読める場合 true。

failure_tags は、失敗理由を上の候補から必要なだけ返す。失敗がなければ []。
pass_metaphor_integrity は参考値。最終 pass は harness 側で primitive labels から再計算される。
""".strip()
    return [
        {
            "role": "system",
            "content": "You are J_META: a metaphor-integrity auditor. Return strict JSON only.",
        },
        {
            "role": "user",
            "content": (
                f"{_case_block(case)}\n\n"
                f"METAPHOR_MODE: {case.metaphor_mode}\n"
                f"VEHICLE_SPEC: {case.vehicle_spec}\n"
                f"CONTROL_ARM: {control_arm}\n"
                f"MAPPING_VISIBILITY: {mapping_visibility}\n\n"
                f"GENERATED_TEXT:\n{generated_text}\n\n"
                f"ANCHORS:\n{anchors}\n\n"
                f"Return exactly this JSON shape, with real values:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def make_stance_leakage_audit_messages(case: Case, control_arm: str, mapping_visibility: str, generated_text: str) -> list[dict[str, str]]:
    schema = {
        "target_preserved": True,
        "actual_target_stance": "fact | nonfact | mixed | unclear",
        "actual_vehicle_stance": "fact | nonfact | mixed | unclear | not_applicable",
        "stance_pattern_match": True,
        "stance_slip": False,
        "literal_vehicle_asserted": False,
        "unsupported_new_claim": False,
        "target_drift": False,
        "false_entailment_risk": 1,
        "ambiguity_risk": 1,
        "pass_stance_pattern": True,
        "pass_leakage": True,
        "reason": "short explanation",
    }
    anchors = """
false_entailment_risk は 1-5 の整数:
1 = 誤った事実を推論する可能性はほぼない
2 = 軽い曖昧さはあるが通常は比喩として読める
3 = 背景知識のない読者なら誤解しうる
4 = 誤った事実含意がかなり強い
5 = 具体的な虚偽主張として読まれる可能性が高い

ambiguity_risk は 1-5 の整数:
1 = 必要十分に明瞭
2 = 比喩として自然な曖昧さ
3 = 解釈が割れうる
4 = 重要な点で曖昧
5 = 主要命題が不明または誤読を誘発

pass_stance_pattern / pass_leakage は参考値として返してください。ただし最終 pass は harness 側で primitive labels から再計算されます。
""".strip()
    return [
        {
            "role": "system",
            "content": "You are J1: a stance and factual-leakage auditor for metaphor benchmarks. Return strict JSON only.",
        },
        {
            "role": "user",
            "content": (
                f"{_case_block(case)}\n\n"
                f"METAPHOR_MODE: {case.metaphor_mode}\n"
                f"VEHICLE_SPEC: {case.vehicle_spec}\n"
                f"CONTROL_ARM: {control_arm}\n"
                f"MAPPING_VISIBILITY: {mapping_visibility}\n\n"
                f"GENERATED_TEXT:\n{generated_text}\n\n"
                f"ANCHORS:\n{anchors}\n\n"
                f"Return exactly this JSON shape, with real values:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def make_relation_audit_messages(case: Case, control_arm: str, mapping_visibility: str, generated_text: str) -> list[dict[str, str]]:
    schema = {
        "relation_mapping_pass": True,
        "surface_only_mapping": False,
        "target_relations_covered": ["relation_name_or_description"],
        "missing_target_relations": [],
        "vehicle_relations_used": ["relation_description"],
        "domain_distance_respected": True,
        "relation_score": 4,
        "pass_relation": True,
        "reason": "short explanation",
    }
    anchors = """
この J2 は metaphor_mode=structural の関係写像を評価する。
vehicle_spec=open の場合、事前指定された desired_vehicle_relations との一致ではなく、生成文の vehicle 像が target_relations を有効に担っているかを評価する。

relation_score は 1-5 の整数:
1 = target と vehicle の関係構造がほぼ写っていない
2 = 表面語や雰囲気だけで、中心関係が弱い
3 = 一部の関係は写っているが重要な欠落がある
4 = 中心関係は明確に写っている
5 = 中心関係と制約が非常に鮮明に写っている

pass_relation は参考値として返してください。ただし最終 pass は harness 側で primitive labels から再計算されます。
""".strip()
    return [
        {
            "role": "system",
            "content": "You are J2: a relational-mapping auditor for metaphors. Return strict JSON only.",
        },
        {
            "role": "user",
            "content": (
                f"{_case_block(case)}\n\n"
                f"METAPHOR_MODE: {case.metaphor_mode}\n"
                f"VEHICLE_SPEC: {case.vehicle_spec}\n"
                f"CONTROL_ARM: {control_arm}\n"
                f"MAPPING_VISIBILITY: {mapping_visibility}\n\n"
                f"GENERATED_TEXT:\n{generated_text}\n\n"
                f"ANCHORS:\n{anchors}\n\n"
                f"Return exactly this JSON shape, with real values:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def make_quality_pairwise_messages(case: Case, text_a: str, text_b: str) -> list[dict[str, str]]:
    schema: dict[str, Any] = {
        "winner": "A | B | TIE",
        "confidence": 3,
        "scores": {
            "A": {"compression": 3, "freshness": 3, "clarity": 3, "rhythm": 3},
            "B": {"compression": 3, "freshness": 3, "clarity": 3, "rhythm": 3},
        },
        "reason": "short explanation",
    }
    anchors = """
この J3 では factual safety は評価しない。J1/J2 で別に評価する。
比喩としての quality だけを比較する。

winner:
- A = A のほうが良い
- B = B のほうが良い
- TIE = ほぼ同等

confidence は 1-5:
1 = ほぼ判断不能
3 = 中程度に自信あり
5 = 明確
""".strip()
    return [
        {
            "role": "system",
            "content": "You are J3: a pairwise metaphor-quality rater. Return strict JSON only.",
        },
        {
            "role": "user",
            "content": (
                f"{_case_block(case)}\n\n"
                f"METAPHOR_MODE: {case.metaphor_mode}\n"
                f"VEHICLE_SPEC: {case.vehicle_spec}\n\n"
                f"TEXT_A:\n{text_a}\n\n"
                f"TEXT_B:\n{text_b}\n\n"
                f"ANCHORS:\n{anchors}\n\n"
                f"Return exactly this JSON shape, with real values:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            ),
        },
    ]
