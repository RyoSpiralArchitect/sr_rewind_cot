from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, List

VALID_TRUTH_BINARY = {"fact", "nonfact"}
VALID_STANCE_PATTERNS = {
    "vehicle_fact__target_fact",
    "vehicle_fact__target_nonfact",
    "vehicle_nonfact__target_fact",
    "vehicle_nonfact__target_nonfact",
}
VALID_DOMAIN_DISTANCE = {"near", "medium", "far"}
VALID_CONTROL_ARMS = {
    "metaphor_with_forbidden",
    "metaphor_without_forbidden",
    "literal_paraphrase",
    "stance_explicit_metaphor",
}
VALID_MAPPING_VISIBILITY = {"hidden", "scaffolded"}
VALID_METAPHOR_MODES = {"structural", "literary", "humorous"}
VALID_VEHICLE_SPEC = {"constrained", "open"}


class SchemaError(ValueError):
    pass


def _require(obj: Dict[str, Any], key: str) -> Any:
    if key not in obj:
        raise SchemaError(f"Missing required key: {key}")
    return obj[key]


def _as_str_list(value: Any, key: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise SchemaError(f"{key} must be a list[str]")
    return value


def _stance_pattern_from_truth(vehicle_truth: str, target_truth: str) -> str:
    return f"vehicle_{vehicle_truth}__target_{target_truth}"


@dataclass(frozen=True)
class TargetSpec:
    claim: str
    truth_binary: str
    truth_subtype: str
    domain: str
    forbidden_target_drift: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "TargetSpec":
        truth_binary = str(_require(obj, "truth_binary"))
        if truth_binary not in VALID_TRUTH_BINARY:
            raise SchemaError(f"target.truth_binary must be one of {sorted(VALID_TRUTH_BINARY)}")
        return cls(
            claim=str(_require(obj, "claim")),
            truth_binary=truth_binary,
            truth_subtype=str(obj.get("truth_subtype", "unspecified")),
            domain=str(obj.get("domain", "unspecified")),
            forbidden_target_drift=_as_str_list(obj.get("forbidden_target_drift", []), "target.forbidden_target_drift"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "truth_binary": self.truth_binary,
            "truth_subtype": self.truth_subtype,
            "domain": self.domain,
            "forbidden_target_drift": list(self.forbidden_target_drift),
        }


@dataclass(frozen=True)
class VehicleSpec:
    truth_binary: str
    truth_subtype: str
    domain: str
    instruction: str

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "VehicleSpec":
        truth_binary = str(_require(obj, "truth_binary"))
        if truth_binary not in VALID_TRUTH_BINARY:
            raise SchemaError(f"vehicle.truth_binary must be one of {sorted(VALID_TRUTH_BINARY)}")
        return cls(
            truth_binary=truth_binary,
            truth_subtype=str(obj.get("truth_subtype", "unspecified")),
            domain=str(obj.get("domain", "unspecified")),
            instruction=str(obj.get("instruction", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truth_binary": self.truth_binary,
            "truth_subtype": self.truth_subtype,
            "domain": self.domain,
            "instruction": self.instruction,
        }


@dataclass(frozen=True)
class MappingSpec:
    domain_distance: str
    target_relations: List[Dict[str, Any]] = field(default_factory=list)
    desired_vehicle_relations: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "MappingSpec":
        domain_distance = str(obj.get("domain_distance", "near"))
        if domain_distance not in VALID_DOMAIN_DISTANCE:
            raise SchemaError(f"mapping.domain_distance must be one of {sorted(VALID_DOMAIN_DISTANCE)}")
        target_relations = obj.get("target_relations", [])
        desired_vehicle_relations = obj.get("desired_vehicle_relations", [])
        if not isinstance(target_relations, list) or not all(isinstance(x, dict) for x in target_relations):
            raise SchemaError("mapping.target_relations must be list[dict]")
        if not isinstance(desired_vehicle_relations, list) or not all(isinstance(x, dict) for x in desired_vehicle_relations):
            raise SchemaError("mapping.desired_vehicle_relations must be list[dict]")
        return cls(
            domain_distance=domain_distance,
            target_relations=target_relations,
            desired_vehicle_relations=desired_vehicle_relations,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_distance": self.domain_distance,
            "target_relations": list(self.target_relations),
            "desired_vehicle_relations": list(self.desired_vehicle_relations),
        }


@dataclass(frozen=True)
class Case:
    case_id: str
    stance_pattern: str
    metaphor_mode: str
    vehicle_spec: str
    target: TargetSpec
    vehicle: VehicleSpec
    mapping: MappingSpec
    expressive: Dict[str, Any] = field(default_factory=dict)
    forbidden_implications: List[str] = field(default_factory=list)
    risk_domain: str = "benign"
    notes: str = ""

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Case":
        target = TargetSpec.from_dict(_require(obj, "target"))
        vehicle = VehicleSpec.from_dict(_require(obj, "vehicle"))
        metaphor_mode = str(obj.get("metaphor_mode", "structural"))
        if metaphor_mode not in VALID_METAPHOR_MODES:
            raise SchemaError(f"metaphor_mode must be one of {sorted(VALID_METAPHOR_MODES)}")
        vehicle_spec = str(obj.get("vehicle_spec", "constrained"))
        if vehicle_spec not in VALID_VEHICLE_SPEC:
            raise SchemaError(f"vehicle_spec must be one of {sorted(VALID_VEHICLE_SPEC)}")
        stance_pattern = str(_require(obj, "stance_pattern"))
        if stance_pattern not in VALID_STANCE_PATTERNS:
            raise SchemaError(f"stance_pattern must be one of {sorted(VALID_STANCE_PATTERNS)}")
        inferred = _stance_pattern_from_truth(vehicle.truth_binary, target.truth_binary)
        if stance_pattern != inferred:
            raise SchemaError(
                f"stance_pattern={stance_pattern} disagrees with vehicle/target truth fields; inferred={inferred}"
            )
        expressive = obj.get("expressive", {})
        if not isinstance(expressive, dict):
            raise SchemaError("expressive must be a dict when provided")
        return cls(
            case_id=str(_require(obj, "case_id")),
            stance_pattern=stance_pattern,
            metaphor_mode=metaphor_mode,
            vehicle_spec=vehicle_spec,
            target=target,
            vehicle=vehicle,
            mapping=MappingSpec.from_dict(obj.get("mapping", {})),
            expressive=dict(expressive),
            forbidden_implications=_as_str_list(obj.get("forbidden_implications", []), "forbidden_implications"),
            risk_domain=str(obj.get("risk_domain", "benign")),
            notes=str(obj.get("notes", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "stance_pattern": self.stance_pattern,
            "metaphor_mode": self.metaphor_mode,
            "vehicle_spec": self.vehicle_spec,
            "target": self.target.to_dict(),
            "vehicle": self.vehicle.to_dict(),
            "mapping": self.mapping.to_dict(),
            "expressive": dict(self.expressive),
            "forbidden_implications": list(self.forbidden_implications),
            "risk_domain": self.risk_domain,
            "notes": self.notes,
        }


def case_content_hash(case_or_dict: Case | Dict[str, Any]) -> str:
    """Stable content hash for the full case definition.

    The hash is intentionally based on full case JSON rather than only case_id.
    Editing a seed while reusing the same case_id therefore creates new run IDs
    and prevents silent reuse of stale generations.
    """
    if isinstance(case_or_dict, Case):
        obj = case_or_dict.to_dict()
    else:
        obj = dict(case_or_dict)
    obj.pop("case_hash", None)
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def validate_control_arm(arm: str) -> str:
    if arm not in VALID_CONTROL_ARMS:
        raise SchemaError(f"control arm must be one of {sorted(VALID_CONTROL_ARMS)}; got {arm}")
    return arm


def validate_mapping_visibility(value: str) -> str:
    if value not in VALID_MAPPING_VISIBILITY:
        raise SchemaError(f"mapping visibility must be one of {sorted(VALID_MAPPING_VISIBILITY)}; got {value}")
    return value
