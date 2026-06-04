from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class ProviderError(RuntimeError):
    pass


class ProviderClient(Protocol):
    name: str
    model: str

    async def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        ...


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    type: str
    model: str
    behavior: str = "safe"
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    timeout_s: float = 60.0
    max_tokens: int = 700

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "ProviderSpec":
        return cls(
            name=str(obj["name"]),
            type=str(obj.get("type", "mock")),
            model=str(obj.get("model", obj.get("name", "mock"))),
            behavior=str(obj.get("behavior", "safe")),
            base_url=obj.get("base_url"),
            api_key_env=obj.get("api_key_env"),
            api_key=obj.get("api_key"),
            timeout_s=float(obj.get("timeout_s", 60.0)),
            max_tokens=int(obj.get("max_tokens", 700)),
        )


def load_provider_config(path: str) -> tuple[list[ProviderSpec], list[ProviderSpec]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    generators = [ProviderSpec.from_dict(x) for x in obj.get("generators", [])]
    judges = [ProviderSpec.from_dict(x) for x in obj.get("judges", [])]
    if not generators:
        raise ProviderError("Provider config must include at least one generator")
    if not judges:
        raise ProviderError("Provider config must include at least one judge")
    return generators, judges


def build_provider(spec: ProviderSpec) -> ProviderClient:
    if spec.type == "mock":
        return MockProvider(spec)
    if spec.type == "openai_compatible":
        return OpenAICompatibleProvider(spec)
    raise ProviderError(f"Unknown provider type: {spec.type}")


def _extract_case(messages: list[dict[str, str]]) -> dict[str, Any]:
    text = "\n".join(m.get("content", "") for m in messages)
    marker = "CASE_JSON:"
    end_marker = "END_CASE_JSON"
    start = text.find(marker)
    end = text.find(end_marker)
    if start == -1 or end == -1 or end < start:
        return {}
    blob = text[start + len(marker):end].strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return {}


def _extract_generated_text(messages: list[dict[str, str]]) -> str:
    text = "\n".join(m.get("content", "") for m in messages)
    m = re.search(r"GENERATED_TEXT:\s*(.*?)\n\s*(?:ANCHORS:|Return exactly|$)", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return text


def _extract_pair_texts(messages: list[dict[str, str]]) -> tuple[str, str]:
    text = "\n".join(m.get("content", "") for m in messages)
    ma = re.search(r"TEXT_A:\s*(.*?)\n\s*TEXT_B:", text, flags=re.S)
    mb = re.search(r"TEXT_B:\s*(.*?)\n\s*(?:ANCHORS:|Return exactly|$)", text, flags=re.S)
    return (ma.group(1).strip() if ma else "", mb.group(1).strip() if mb else "")


def _extract_control_arm(messages: list[dict[str, str]]) -> str:
    text = "\n".join(m.get("content", "") for m in messages)
    m = re.search(r"CONTROL_ARM:\s*([^\n]+)", text)
    return m.group(1).strip() if m else "metaphor_with_forbidden"


def _extract_metaphor_mode(messages: list[dict[str, str]]) -> str:
    text = "\n".join(m.get("content", "") for m in messages)
    m = re.search(r"METAPHOR_MODE:\s*([^\n]+)", text)
    return m.group(1).strip() if m else "structural"


def _extract_sample_index(messages: list[dict[str, str]]) -> int:
    text = "\n".join(m.get("content", "") for m in messages)
    m = re.search(r"SAMPLE_INDEX:\s*(\d+)", text)
    return int(m.group(1)) if m else 0


def _json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


class MockProvider:
    def __init__(self, spec: ProviderSpec):
        self.name = spec.name
        self.model = spec.model
        self.behavior = spec.behavior
        self.timeout_s = spec.timeout_s

    async def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        await asyncio.sleep(0)
        system = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
        if "J1:" in system:
            return self._mock_j1(messages)
        if "J2:" in system:
            return self._mock_j2(messages)
        if "J_LIT:" in system:
            return self._mock_j_lit(messages)
        if "J_HUMOR:" in system:
            return self._mock_j_humor(messages)
        if "J_META:" in system:
            return self._mock_j_meta(messages)
        if "J3:" in system:
            return self._mock_j3(messages)
        return self._mock_generation(messages, temperature)

    def _mock_generation(self, messages: list[dict[str, str]], temperature: float) -> str:
        case = _extract_case(messages)
        control_arm = _extract_control_arm(messages)
        metaphor_mode = _extract_metaphor_mode(messages)
        sample_index = _extract_sample_index(messages)
        target = case.get("target", {})
        vehicle = case.get("vehicle", {})
        mapping = case.get("mapping", {})
        claim = target.get("claim", "この命題")
        vehicle_truth = vehicle.get("truth_binary", "fact")
        vehicle_domain = vehicle.get("domain", "unknown")
        domain_distance = mapping.get("domain_distance", "near")
        relation = ""
        rels = mapping.get("target_relations", [])
        if rels:
            relation = rels[0].get("description") or rels[0].get("name", "中心関係")

        if control_arm == "literal_paraphrase":
            return f"{claim}"

        if metaphor_mode == "literary":
            return f"{claim}は、濡れた硝子の内側で小さな灯りが息をひそめるように近づく。"
        if metaphor_mode == "humorous":
            return f"{claim}は、会議室に迷い込んだGPSが自信満々に海へ案内するようなものだ。"

        # Deterministic variation from sample index and provider behavior.
        rng = random.Random(f"{self.name}|{self.behavior}|{case.get('case_id')}|{control_arm}|{sample_index}|{temperature}")
        slip = self.behavior == "slippy" and (vehicle_truth == "nonfact") and (rng.random() < min(0.85, 0.35 + temperature / 2))
        surface = self.behavior == "surface" or (domain_distance == "far" and rng.random() < max(0.0, temperature - 0.55))

        if vehicle_truth == "fact":
            image = f"{vehicle_domain}の実在する制御弁が流れを調整する"
        else:
            image = f"架空の{vehicle_domain}の透明な歯車が、存在しない夜の入力を受け流す"

        if control_arm == "stance_explicit_metaphor" and vehicle_truth == "nonfact":
            image = f"文字通りではない比喩として、{image}"

        if slip:
            return f"実際には、{claim}。その背後では{image}仕組みが物理的に働いている。"

        if surface:
            return f"{claim}は、{vehicle_domain}の火花や配線のようにきらめく。"

        if relation:
            return f"{claim}は、{image}ように、{relation}という関係を一息で見せる。"
        return f"{claim}は、{image}ようなものだ。"

    def _mock_j1(self, messages: list[dict[str, str]]) -> str:
        case = _extract_case(messages)
        control_arm = _extract_control_arm(messages)
        generated = _extract_generated_text(messages)
        target = case.get("target", {})
        vehicle = case.get("vehicle", {})
        claim = target.get("claim", "")
        target_truth = target.get("truth_binary", "fact")
        vehicle_truth = vehicle.get("truth_binary", "fact")

        target_preserved = bool(claim and claim in generated)
        literal_vehicle = any(x in generated for x in ["物理的に働いている", "実際には", "背後では"])
        unsupported = any(x in generated for x in ["物理的に働いている", "背後では", "体内に", "実在する装置"])
        target_drift = not target_preserved

        if control_arm == "literal_paraphrase":
            actual_vehicle = "not_applicable"
            stance_pattern_match = True
            stance_slip = False
        else:
            actual_vehicle = "fact" if literal_vehicle and vehicle_truth == "nonfact" else vehicle_truth
            actual_target = target_truth
            stance_pattern_match = actual_vehicle == vehicle_truth and actual_target == target_truth
            stance_slip = not stance_pattern_match

        false_entailment_risk = 4 if literal_vehicle or unsupported else (3 if target_drift else 1)
        ambiguity_risk = 2 if "よう" in generated else 1
        pass_leakage = not (literal_vehicle or unsupported or target_drift or false_entailment_risk >= 4)
        pass_stance_pattern = stance_pattern_match and not stance_slip

        return _json({
            "target_preserved": target_preserved,
            "actual_target_stance": target_truth,
            "actual_vehicle_stance": actual_vehicle,
            "stance_pattern_match": stance_pattern_match,
            "stance_slip": stance_slip,
            "literal_vehicle_asserted": literal_vehicle,
            "unsupported_new_claim": unsupported,
            "target_drift": target_drift,
            "false_entailment_risk": false_entailment_risk,
            "ambiguity_risk": ambiguity_risk,
            "pass_stance_pattern": pass_stance_pattern,
            "pass_leakage": pass_leakage,
            "reason": "mock判定: 明示的な物理化・実在化表現とtarget保持を簡易検査した。",
        })

    def _mock_j2(self, messages: list[dict[str, str]]) -> str:
        case = _extract_case(messages)
        control_arm = _extract_control_arm(messages)
        generated = _extract_generated_text(messages)
        rels = case.get("mapping", {}).get("target_relations", [])
        names = [(r.get("name") or r.get("description") or "relation") for r in rels]

        literal = control_arm == "literal_paraphrase"
        surface = "火花や配線" in generated or "きらめく" in generated
        has_relation_marker = ("関係" in generated) or any((r.get("description") or "")[:8] in generated for r in rels)
        pass_relation = (not literal) and (not surface) and has_relation_marker
        score = 4 if pass_relation else (2 if surface else 1 if literal else 3)
        return _json({
            "relation_mapping_pass": pass_relation,
            "surface_only_mapping": surface,
            "target_relations_covered": names if pass_relation else [],
            "missing_target_relations": [] if pass_relation else names,
            "vehicle_relations_used": ["mock vehicle relation"] if pass_relation else [],
            "domain_distance_respected": True,
            "relation_score": score,
            "pass_relation": pass_relation,
            "reason": "mock判定: 関係マーカーと表面語逃げを簡易検査した。",
        })

    def _mock_j_meta(self, messages: list[dict[str, str]]) -> str:
        case = _extract_case(messages)
        generated = _extract_generated_text(messages)
        metaphor_mode = _extract_metaphor_mode(messages)
        claim = case.get("target", {}).get("claim", "")

        target_anchor = bool(claim and claim in generated)
        semantic_break = any(x in generated for x in ["明日の朝の余韻", "時間のない宇宙船"])
        literal_scene = any(x in generated for x in ["物理的に働いている", "実際に液体", "本当に超自然"])
        weak_affordance = any(x in generated for x in ["地図が迷子", "地図のように、何度も方向を変え"])
        medium_dynamics_mismatch = any(x in generated for x in ["霧の波", "海の底", "水面", "岸を", "岸辺"])
        temporal_anchor_broken = any(x in generated for x in ["明日の朝の余韻", "未来の余韻", "まだ来ていない余韻"])
        premise_load = 4 if any(x in generated for x in ["無限", "時空", "宇宙船"]) else 2
        mode_fit = True
        if metaphor_mode == "humorous":
            mode_fit = any(x in generated for x in ["GPS", "自信満々", "会議", "迷子", "海"])
        elif metaphor_mode == "literary":
            mode_fit = not any(x in generated for x in ["GPS", "笑える", "冗談"])
        imageability = 2 if weak_affordance or semantic_break or medium_dynamics_mismatch else 4
        premise_overload = premise_load >= 4
        target_fact_drift = not target_anchor
        tone_contract_slip = not mode_fit
        metaphor_literal_ambiguity = literal_scene
        licensed_rupture = (
            metaphor_mode == "humorous"
            and target_anchor
            and mode_fit
            and any(x in generated for x in ["無限のサイコロ", "ナンセンス", "わざと"])
        )
        invariant_preserved = (
            target_anchor
            and not weak_affordance
            and not semantic_break
            and not temporal_anchor_broken
        )

        tags = []
        if target_fact_drift:
            tags.append("target_fact_drift")
        if weak_affordance:
            tags.append("vehicle_affordance_broken")
        if medium_dynamics_mismatch:
            tags.append("medium_dynamics_mismatch")
        if temporal_anchor_broken:
            tags.append("temporal_anchor_broken")
        if literal_scene:
            tags.append("metaphor_literal_ambiguity")
        if premise_overload:
            tags.append("premise_overload")
        if semantic_break:
            tags.append("semantic_break")
        if tone_contract_slip:
            tags.append("tone_contract_slip")
        if imageability <= 2:
            tags.append("imageability_failure")
        if licensed_rupture:
            tags.append("licensed_rupture_success")

        passed = (
            target_anchor
            and not weak_affordance
            and not medium_dynamics_mismatch
            and not temporal_anchor_broken
            and not literal_scene
            and not semantic_break
            and mode_fit
            and (premise_load <= 3 or licensed_rupture)
            and imageability >= 3
            and invariant_preserved
        )
        return _json({
            "target_anchor_preserved": target_anchor,
            "vehicle_affordance_coherent": not weak_affordance,
            "literal_scene_confusion": literal_scene,
            "premise_load_score": premise_load,
            "semantic_break": semantic_break,
            "mode_fit": mode_fit,
            "imageability_score": imageability,
            "target_fact_drift": target_fact_drift,
            "vehicle_affordance_broken": weak_affordance,
            "medium_dynamics_mismatch": medium_dynamics_mismatch,
            "temporal_anchor_broken": temporal_anchor_broken,
            "premise_overload": premise_overload,
            "tone_contract_slip": tone_contract_slip,
            "metaphor_literal_ambiguity": metaphor_literal_ambiguity,
            "licensed_rupture_success": licensed_rupture,
            "invariant_preserved": invariant_preserved,
            "failure_tags": tags,
            "pass_metaphor_integrity": passed,
            "reason": "mock判定: target保持、前提負荷、写像の足場を簡易検査した。",
        })

    def _mock_j3(self, messages: list[dict[str, str]]) -> str:
        a, b = _extract_pair_texts(messages)

        def score(t: str) -> int:
            s = 0
            s += 2 if "ように" in t or "ような" in t else 0
            s += 2 if "関係" in t else 0
            s += 1 if "透明な歯車" in t else 0
            s -= 2 if "物理的に働いている" in t else 0
            s -= 1 if len(t) < 15 else 0
            return s

        sa, sb = score(a), score(b)
        if abs(sa - sb) <= 1:
            winner = "TIE"
        else:
            winner = "A" if sa > sb else "B"
        return _json({
            "winner": winner,
            "confidence": 3 if winner != "TIE" else 2,
            "scores": {
                "A": {"compression": 3, "freshness": max(1, min(5, sa + 2)), "clarity": 3, "rhythm": 3},
                "B": {"compression": 3, "freshness": max(1, min(5, sb + 2)), "clarity": 3, "rhythm": 3},
            },
            "reason": "mock判定: 簡易スコアによる比較。",
        })

    def _mock_j_lit(self, messages: list[dict[str, str]]) -> str:
        case = _extract_case(messages)
        generated = _extract_generated_text(messages)
        claim = case.get("target", {}).get("claim", "")
        target_drift = bool(claim and claim not in generated)
        semantic_break = any(x in generated for x in ["明日の朝の余韻", "時間のない宇宙船"])
        vehicle_or_medium_confusion = any(x in generated for x in ["霧の波", "海の底", "水面", "岸を", "地図が迷子", "迷子のカメラ"])
        literal_scene_confusion = any(x in generated for x in ["本当に超自然", "実際に液体", "物理的に働いている"])
        mode_ambiguity = any(x in generated for x in ["GPS", "冗談", "笑える"])
        premise_overload = any(x in generated for x in ["無限", "時空", "宇宙船"])
        over_grandiose = any(x in generated for x in ["心を押し潰す", "実存", "悲劇"])
        over_explanation = any(x in generated for x in ["つまり", "これは", "意味する"])
        cliche = 5 if any(x in generated for x in ["夢のよう", "人生は旅", "光と闇"]) else 1
        freshness = 2 if cliche >= 4 else 4
        tonal_fit = 3 if "ように" in generated else 4
        affect = 4 if any(x in generated for x in ["ひそめる", "濡れた", "灯り", "沈む"]) else 3
        compression = 4 if len(generated) <= 80 else 3
        acceptable_ambiguity = not (
            target_drift
            or semantic_break
            or vehicle_or_medium_confusion
            or literal_scene_confusion
            or premise_overload
        )
        passed = (
            freshness >= 3
            and affect >= 3
            and cliche <= 3
            and not over_explanation
            and acceptable_ambiguity
            and not mode_ambiguity
            and not over_grandiose
        )
        return _json({
            "acceptable_metaphorical_ambiguity": acceptable_ambiguity,
            "semantic_break": semantic_break,
            "target_drift": target_drift,
            "vehicle_or_medium_confusion": vehicle_or_medium_confusion,
            "literal_scene_confusion": literal_scene_confusion,
            "mode_ambiguity": mode_ambiguity,
            "premise_overload": premise_overload,
            "over_grandiose": over_grandiose,
            "freshness_score": freshness,
            "tonal_fit_score": tonal_fit,
            "affect_transfer_score": affect,
            "poetic_compression_score": compression,
            "cliche_risk": cliche,
            "over_explanation": over_explanation,
            "register_mismatch": False,
            "pass_literary_quality": passed,
            "reason": "mock判定: 陳腐語と説明化を簡易検査した。",
        })

    def _mock_j_humor(self, messages: list[dict[str, str]]) -> str:
        case = _extract_case(messages)
        generated = _extract_generated_text(messages)
        claim = case.get("target", {}).get("claim", "")
        target_drift = bool(claim and claim not in generated)
        semantic_break = any(x in generated for x in ["明日の朝の余韻", "時間のない宇宙船"])
        vehicle_or_medium_confusion = any(x in generated for x in ["地図が迷子", "迷子になった自信満々の地図", "迷子のカメラ", "霧の波"])
        literal_scene_confusion = any(x in generated for x in ["本当に超自然", "実際に液体", "物理的に働いている"])
        mode_ambiguity = any(x in generated for x in ["静かな海", "微かな不安", "静寂"])
        premise_overload = any(x in generated for x in ["時間のない宇宙船", "時空", "古代の機械"])
        over_explanation = any(x in generated for x in ["なぜなら", "という冗談", "笑える"])
        flattening = over_explanation or not any(x in generated for x in ["GPS", "自信満々", "海", "会議室"])
        incongruity = not flattening
        surprise = 4 if incongruity else 2
        timing = 4 if "ようなものだ" in generated or "ように" in generated else 3
        acceptable_ambiguity = not (
            target_drift
            or semantic_break
            or vehicle_or_medium_confusion
            or literal_scene_confusion
            or premise_overload
        )
        passed = incongruity and not over_explanation and acceptable_ambiguity and not mode_ambiguity
        return _json({
            "acceptable_metaphorical_ambiguity": acceptable_ambiguity,
            "semantic_break": semantic_break,
            "target_drift": target_drift,
            "vehicle_or_medium_confusion": vehicle_or_medium_confusion,
            "literal_scene_confusion": literal_scene_confusion,
            "mode_ambiguity": mode_ambiguity,
            "premise_overload": premise_overload,
            "incongruity_success": incongruity,
            "surprise_score": surprise,
            "tonal_fit_score": 4,
            "comic_timing_score": timing,
            "humor_flattening": flattening,
            "over_explanation": over_explanation,
            "mean_spiritedness_risk": 1,
            "pass_humor_quality": passed,
            "reason": "mock判定: 不整合と説明化を簡易検査した。",
        })


class OpenAICompatibleProvider:
    """Minimal adapter for chat-completions-compatible HTTP APIs.

    This adapter intentionally uses only the Python standard library. It expects
    a provider endpoint compatible with a POST to {base_url}/chat/completions.
    """

    def __init__(self, spec: ProviderSpec):
        if not spec.base_url:
            raise ProviderError(f"Provider {spec.name} requires base_url")
        self.name = spec.name
        self.model = spec.model
        self.base_url = spec.base_url.rstrip("/")
        self.timeout_s = spec.timeout_s
        self.max_tokens = spec.max_tokens
        self.api_key = spec.api_key or (os.environ.get(spec.api_key_env or "") if spec.api_key_env else None)
        if not self.api_key:
            raise ProviderError(f"Provider {spec.name} requires api_key or api_key_env")

    async def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        return await asyncio.to_thread(self._complete_sync, messages, temperature)

    def _complete_sync(self, messages: list[dict[str, str]], temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"HTTP error from {self.name}: {exc.code}: {body}") from exc
        except Exception as exc:
            raise ProviderError(f"Error from {self.name}: {exc}") from exc
        try:
            return obj["choices"][0]["message"]["content"]
        except Exception as exc:
            raise ProviderError(f"Unexpected response shape from {self.name}: {obj}") from exc
