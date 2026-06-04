from __future__ import annotations

from typing import Any, Iterable

PASS_RULES_VERSION = "rules_v3"


def bool_or_none(x: Any) -> bool | None:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        text = x.strip().lower()
        if text in {"true", "yes", "y", "1", "pass"}:
            return True
        if text in {"false", "no", "n", "0", "fail"}:
            return False
    if isinstance(x, (int, float)) and x in {0, 1}:
        return bool(x)
    return None


def int_or_none(x: Any) -> int | None:
    try:
        if x is None or isinstance(x, bool):
            return None
        return int(float(x))
    except Exception:
        return None


def number_or_none(x: Any) -> float | None:
    try:
        if x is None or isinstance(x, bool):
            return None
        return float(x)
    except Exception:
        return None


def majority_bool(values: Iterable[Any]) -> bool | None:
    vals = [bool_or_none(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    true_count = sum(1 for v in vals if v)
    false_count = len(vals) - true_count
    if true_count > false_count:
        return True
    if false_count > true_count:
        return False
    return None


def mean_number(values: Iterable[Any]) -> float | None:
    nums: list[float] = []
    for v in values:
        n = number_or_none(v)
        if n is not None:
            nums.append(n)
    if not nums:
        return None
    return sum(nums) / len(nums)


def compute_pass_stance_pattern(labels: dict[str, Any]) -> bool | None:
    stance_pattern_match = bool_or_none(labels.get("stance_pattern_match"))
    stance_slip = bool_or_none(labels.get("stance_slip"))
    if stance_pattern_match is False or stance_slip is True:
        return False
    if stance_pattern_match is True and stance_slip is False:
        return True
    return None


def compute_pass_leakage(labels: dict[str, Any]) -> bool | None:
    literal_vehicle = bool_or_none(labels.get("literal_vehicle_asserted"))
    unsupported = bool_or_none(labels.get("unsupported_new_claim"))
    target_drift = bool_or_none(labels.get("target_drift"))
    risk = number_or_none(labels.get("false_entailment_risk"))
    if risk is None:
        risk = number_or_none(labels.get("false_entailment_risk_mean"))

    fail_flags = [literal_vehicle, unsupported, target_drift]
    if any(v is True for v in fail_flags):
        return False
    if risk is not None and risk >= 4:
        return False
    if all(v is False for v in fail_flags) and risk is not None:
        return True
    return None


def compute_pass_relation(labels: dict[str, Any]) -> bool | None:
    surface = bool_or_none(labels.get("surface_only_mapping"))
    mapping_pass = bool_or_none(labels.get("relation_mapping_pass"))
    score = number_or_none(labels.get("relation_score"))
    if score is None:
        score = number_or_none(labels.get("relation_score_mean"))

    if surface is True:
        return False
    if mapping_pass is False:
        return False
    if score is not None and score <= 2:
        return False
    if surface is False and mapping_pass is True and score is not None and score >= 3:
        return True
    return None


def compute_pass_metaphor_integrity(labels: dict[str, Any]) -> bool | None:
    def get(name: str) -> Any:
        return labels.get(name, labels.get(f"metaphor_{name}"))

    target_anchor = bool_or_none(get("target_anchor_preserved"))
    vehicle_affordance = bool_or_none(get("vehicle_affordance_coherent"))
    literal_scene = bool_or_none(get("literal_scene_confusion"))
    semantic_break = bool_or_none(get("semantic_break"))
    mode_fit = bool_or_none(get("mode_fit"))
    target_fact_drift = bool_or_none(get("target_fact_drift"))
    vehicle_affordance_broken = bool_or_none(get("vehicle_affordance_broken"))
    medium_dynamics_mismatch = bool_or_none(get("medium_dynamics_mismatch"))
    temporal_anchor_broken = bool_or_none(get("temporal_anchor_broken"))
    premise_overload = bool_or_none(get("premise_overload"))
    tone_contract_slip = bool_or_none(get("tone_contract_slip"))
    metaphor_literal_ambiguity = bool_or_none(get("metaphor_literal_ambiguity"))
    licensed_rupture = bool_or_none(get("licensed_rupture_success"))
    invariant_preserved = bool_or_none(get("invariant_preserved"))
    premise_load = number_or_none(labels.get("premise_load_score"))
    if premise_load is None:
        premise_load = number_or_none(labels.get("premise_load_score_mean"))
    imageability = number_or_none(labels.get("imageability_score"))
    if imageability is None:
        imageability = number_or_none(labels.get("imageability_score_mean"))

    rupture_is_licensed = licensed_rupture is True

    if target_fact_drift is True:
        return False
    if invariant_preserved is False:
        return False
    if target_anchor is False:
        return False
    if vehicle_affordance is False and not rupture_is_licensed:
        return False
    if vehicle_affordance_broken is True and not rupture_is_licensed:
        return False
    if medium_dynamics_mismatch is True and not rupture_is_licensed:
        return False
    if temporal_anchor_broken is True and not rupture_is_licensed:
        return False
    if premise_overload is True and not rupture_is_licensed:
        return False
    if literal_scene is True and not rupture_is_licensed:
        return False
    if semantic_break is True and not rupture_is_licensed:
        return False
    if mode_fit is False or tone_contract_slip is True:
        return False
    if metaphor_literal_ambiguity is True:
        return False
    if premise_load is not None and premise_load >= 4 and not rupture_is_licensed:
        return False
    if imageability is not None and imageability <= 2:
        return False

    if (
        target_anchor is True
        and vehicle_affordance is True
        and literal_scene is False
        and semantic_break is False
        and mode_fit is True
        and premise_load is not None
        and premise_load <= 3
        and imageability is not None
        and imageability >= 3
        and target_fact_drift is not True
        and vehicle_affordance_broken is not True
        and medium_dynamics_mismatch is not True
        and temporal_anchor_broken is not True
        and premise_overload is not True
        and tone_contract_slip is not True
        and metaphor_literal_ambiguity is not True
        and invariant_preserved is not False
    ):
        return True
    if (
        rupture_is_licensed
        and target_fact_drift is not True
        and invariant_preserved is not False
        and target_anchor is not False
        and mode_fit is not False
        and tone_contract_slip is not True
        and metaphor_literal_ambiguity is not True
        and imageability is not None
        and imageability >= 3
    ):
        return True
    return None


def compute_pass_literary_quality(labels: dict[str, Any]) -> bool | None:
    def get(name: str) -> Any:
        return labels.get(name, labels.get(f"literary_{name}", labels.get(f"literary_{name}_mean")))

    acceptable_ambiguity = bool_or_none(get("acceptable_metaphorical_ambiguity"))
    semantic_break = bool_or_none(get("semantic_break"))
    target_drift = bool_or_none(get("target_drift"))
    vehicle_or_medium_confusion = bool_or_none(get("vehicle_or_medium_confusion"))
    literal_scene_confusion = bool_or_none(get("literal_scene_confusion"))
    mode_ambiguity = bool_or_none(get("mode_ambiguity"))
    premise_overload = bool_or_none(get("premise_overload"))
    over_grandiose = bool_or_none(get("over_grandiose"))
    over_explanation = bool_or_none(get("over_explanation"))
    register_mismatch = bool_or_none(get("register_mismatch"))

    freshness = number_or_none(get("freshness_score"))
    tonal_fit = number_or_none(get("tonal_fit_score"))
    affect_transfer = number_or_none(get("affect_transfer_score"))
    poetic_compression = number_or_none(get("poetic_compression_score"))
    cliche_risk = number_or_none(get("cliche_risk"))

    fail_flags = [
        semantic_break,
        target_drift,
        vehicle_or_medium_confusion,
        literal_scene_confusion,
        mode_ambiguity,
        premise_overload,
        over_grandiose,
        over_explanation,
        register_mismatch,
    ]
    if any(v is True for v in fail_flags):
        return False
    if acceptable_ambiguity is False:
        return False
    if freshness is not None and freshness <= 2:
        return False
    if tonal_fit is not None and tonal_fit <= 2:
        return False
    if affect_transfer is not None and affect_transfer <= 2:
        return False
    if poetic_compression is not None and poetic_compression <= 2:
        return False
    if cliche_risk is not None and cliche_risk >= 4:
        return False

    score_values = [freshness, tonal_fit, affect_transfer, poetic_compression, cliche_risk]
    if (
        all(v is False for v in fail_flags if v is not None)
        and acceptable_ambiguity is not False
        and all(v is not None for v in score_values)
        and freshness >= 3
        and tonal_fit >= 3
        and affect_transfer >= 3
        and poetic_compression >= 3
        and cliche_risk <= 3
    ):
        return True
    return None


def compute_pass_humor_quality(labels: dict[str, Any]) -> bool | None:
    def get(name: str) -> Any:
        return labels.get(name, labels.get(f"humor_{name}", labels.get(f"humor_{name}_mean")))

    acceptable_ambiguity = bool_or_none(get("acceptable_metaphorical_ambiguity"))
    semantic_break = bool_or_none(get("semantic_break"))
    target_drift = bool_or_none(get("target_drift"))
    vehicle_or_medium_confusion = bool_or_none(get("vehicle_or_medium_confusion"))
    literal_scene_confusion = bool_or_none(get("literal_scene_confusion"))
    mode_ambiguity = bool_or_none(get("mode_ambiguity"))
    premise_overload = bool_or_none(get("premise_overload"))
    incongruity_success = bool_or_none(get("incongruity_success"))
    humor_flattening = bool_or_none(get("humor_flattening"))
    over_explanation = bool_or_none(get("over_explanation"))

    surprise = number_or_none(get("surprise_score"))
    tonal_fit = number_or_none(get("tonal_fit_score"))
    comic_timing = number_or_none(get("comic_timing_score"))
    mean_spiritedness = number_or_none(get("mean_spiritedness_risk"))

    fail_flags = [
        semantic_break,
        target_drift,
        vehicle_or_medium_confusion,
        literal_scene_confusion,
        mode_ambiguity,
        premise_overload,
        humor_flattening,
        over_explanation,
    ]
    if any(v is True for v in fail_flags):
        return False
    if acceptable_ambiguity is False:
        return False
    if incongruity_success is False:
        return False
    if surprise is not None and surprise <= 2:
        return False
    if tonal_fit is not None and tonal_fit <= 2:
        return False
    if comic_timing is not None and comic_timing <= 2:
        return False
    if mean_spiritedness is not None and mean_spiritedness >= 4:
        return False

    score_values = [surprise, tonal_fit, comic_timing, mean_spiritedness]
    if (
        all(v is False for v in fail_flags if v is not None)
        and acceptable_ambiguity is not False
        and incongruity_success is True
        and all(v is not None for v in score_values)
        and surprise >= 3
        and tonal_fit >= 3
        and comic_timing >= 3
        and mean_spiritedness <= 3
    ):
        return True
    return None


def apply_computed_pass_labels(parsed: dict[str, Any], audit_type: str) -> dict[str, Any]:
    """Return a copy with deterministic pass_* labels overriding judge-supplied pass fields.

    The original judge pass fields are preserved as judge_pass_*_raw. Primitive labels
    such as literal_vehicle_asserted and relation_score are left untouched.
    """
    out = dict(parsed)
    out["pass_rules_version"] = PASS_RULES_VERSION

    if audit_type == "stance_leakage":
        if "pass_stance_pattern" in out:
            out["judge_pass_stance_pattern_raw"] = out.get("pass_stance_pattern")
        if "pass_leakage" in out:
            out["judge_pass_leakage_raw"] = out.get("pass_leakage")
        out["pass_stance_pattern"] = compute_pass_stance_pattern(out)
        out["pass_leakage"] = compute_pass_leakage(out)
    elif audit_type == "relation":
        if "pass_relation" in out:
            out["judge_pass_relation_raw"] = out.get("pass_relation")
        out["pass_relation"] = compute_pass_relation(out)
    elif audit_type == "metaphor_integrity":
        if "pass_metaphor_integrity" in out:
            out["judge_pass_metaphor_integrity_raw"] = out.get("pass_metaphor_integrity")
        out["pass_metaphor_integrity"] = compute_pass_metaphor_integrity(out)
    elif audit_type == "literary":
        if "pass_literary_quality" in out:
            out["judge_pass_literary_quality_raw"] = out.get("pass_literary_quality")
        out["pass_literary_quality"] = compute_pass_literary_quality(out)
    elif audit_type == "humor":
        if "pass_humor_quality" in out:
            out["judge_pass_humor_quality_raw"] = out.get("pass_humor_quality")
        out["pass_humor_quality"] = compute_pass_humor_quality(out)
    return out
