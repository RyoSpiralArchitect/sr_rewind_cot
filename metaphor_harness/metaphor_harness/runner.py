from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import random
import time
from dataclasses import dataclass
from typing import Any

from .db import HarnessDB
from .eval_rules import apply_computed_pass_labels
from .io_utils import load_cases_jsonl
from .prompts import (
    GEN_PROMPT_VERSION,
    J_HUMOR_PROMPT_VERSION,
    J_META_PROMPT_VERSION,
    J1_PROMPT_VERSION,
    J2_PROMPT_VERSION,
    J_LIT_PROMPT_VERSION,
    J3_PROMPT_VERSION,
    make_generation_messages,
    make_humor_audit_messages,
    make_literary_audit_messages,
    make_metaphor_integrity_audit_messages,
    make_stance_leakage_audit_messages,
    make_quality_pairwise_messages,
    make_relation_audit_messages,
)
from .providers import ProviderClient, build_provider, load_provider_config
from .schema import Case, case_content_hash, validate_control_arm, validate_mapping_visibility


def stable_id(*parts: Any, prefix: str = "") -> str:
    blob = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}{h}" if prefix else h


def parse_json_lenient(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {"parse_error": True, "reason": "top-level JSON was not an object", "raw": raw}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"parse_error": True, "reason": "could not parse JSON", "raw": raw}


async def complete_with_retry(
    client: ProviderClient,
    messages: list[dict[str, str]],
    temperature: float,
    retries: int,
    expect_json: bool = False,
) -> tuple[str, dict[str, Any] | None]:
    last_raw = ""
    for _attempt in range(retries + 1):
        raw = await client.complete(messages, temperature=temperature)
        last_raw = raw
        if not expect_json:
            return raw, None
        parsed = parse_json_lenient(raw)
        if not parsed.get("parse_error"):
            return raw, parsed
        messages = messages + [
            {
                "role": "user",
                "content": "前回の出力は JSON として parse できませんでした。説明なしで、指定された JSON object だけを返してください。",
            }
        ]
    return last_raw, parse_json_lenient(last_raw)


@dataclass
class RunOptions:
    cases_path: str
    config_path: str
    db_path: str
    samples: int
    temperatures: list[float]
    control_arms: list[str]
    mapping_visibility: list[str]
    concurrency: int = 4
    retries: int = 2
    dry_run: bool = False
    allow_risky_false_targets: bool = False
    run_quality_pairs: bool = True
    quality_pairs_per_group: int = 12


class HarnessRunner:
    def __init__(self, options: RunOptions):
        self.options = options
        self.db = HarnessDB(options.db_path)
        gen_specs, judge_specs = load_provider_config(options.config_path)
        self.generator_specs = gen_specs
        self.judge_specs = judge_specs
        self.generators = [build_provider(s) for s in gen_specs]
        self.judges = [build_provider(s) for s in judge_specs]
        self.sem = asyncio.Semaphore(max(1, options.concurrency))

    def close(self) -> None:
        self.db.close()

    def _should_skip_case(self, case: Case) -> bool:
        return (
            case.target.truth_binary == "nonfact"
            and case.risk_domain not in {"benign", "low"}
            and not self.options.allow_risky_false_targets
        )

    def _generation_run_id(
        self,
        case_hash: str,
        provider: ProviderClient,
        temperature: float,
        arm: str,
        mapping_visibility: str,
        sample_index: int,
    ) -> str:
        return stable_id(
            "generation", GEN_PROMPT_VERSION, case_hash, provider.name, provider.model,
            temperature, arm, mapping_visibility, sample_index,
            prefix="gen_",
        )

    def _audit_specs_for_case(self, case: Case) -> list[tuple[str, str]]:
        specs = [("stance_leakage", J1_PROMPT_VERSION)]
        if case.metaphor_mode == "structural":
            specs.append(("relation", J2_PROMPT_VERSION))
        elif case.metaphor_mode == "literary":
            specs.append(("metaphor_integrity", J_META_PROMPT_VERSION))
            specs.append(("literary", J_LIT_PROMPT_VERSION))
        elif case.metaphor_mode == "humorous":
            specs.append(("metaphor_integrity", J_META_PROMPT_VERSION))
            specs.append(("humor", J_HUMOR_PROMPT_VERSION))
        return specs

    async def run(self) -> None:
        cases = load_cases_jsonl(self.options.cases_path)
        for arm in self.options.control_arms:
            validate_control_arm(arm)
        for visibility in self.options.mapping_visibility:
            validate_mapping_visibility(visibility)

        case_hashes: dict[str, str] = {}
        for case in cases:
            ch = case_content_hash(case)
            case_hashes[case.case_id] = ch
            self.db.upsert_case(case.to_dict(), ch)

        generation_jobs = []
        skipped_cases = 0
        for case in cases:
            if self._should_skip_case(case):
                skipped_cases += 1
                continue
            ch = case_hashes[case.case_id]
            for generator in self.generators:
                for temp in self.options.temperatures:
                    for arm in self.options.control_arms:
                        visibilities = ["hidden"] if arm == "literal_paraphrase" else self.options.mapping_visibility
                        for mapping_visibility in visibilities:
                            for sample_idx in range(self.options.samples):
                                run_id = self._generation_run_id(ch, generator, temp, arm, mapping_visibility, sample_idx)
                                if not self.db.generation_exists(run_id):
                                    generation_jobs.append((run_id, ch, case, generator, temp, arm, mapping_visibility, sample_idx))

        if self.options.dry_run:
            planned_generations = len(generation_jobs)
            case_lookup = {ch: Case.from_dict(obj) for ch, obj in self.db.fetch_cases_by_hash().items()}
            planned_audits = 0
            planned_new_audits = 0
            for _run_id, _ch, case, *_rest in generation_jobs:
                audit_count = len(self._audit_specs_for_case(case)) * len(self.judges)
                planned_audits += audit_count
                planned_new_audits += audit_count
            for gen in self.db.fetch_generations():
                case = case_lookup.get(gen["case_hash"])
                if case is not None:
                    for judge_index, judge in enumerate(self.judges):
                        for audit_type, prompt_version in self._audit_specs_for_case(case):
                            planned_audits += 1
                            audit_id = stable_id(
                                "audit", prompt_version, gen["run_id"], audit_type,
                                judge.name, judge.model, judge_index,
                                prefix="aud_",
                            )
                            if not self.db.audit_exists(audit_id):
                                planned_new_audits += 1
            print(json.dumps({
                "mode": "dry_run",
                "cases_loaded": len(cases),
                "cases_skipped_due_to_risk_guard": skipped_cases,
                "planned_new_generations": planned_generations,
                "planned_total_audit_calls_after_generation": planned_audits,
                "planned_new_audit_calls": planned_new_audits,
                "generators": [g.name for g in self.generators],
                "judges": [j.name for j in self.judges],
                "temperatures": self.options.temperatures,
                "control_arms": self.options.control_arms,
                "mapping_visibility": self.options.mapping_visibility,
                "samples": self.options.samples,
                "prompt_versions": {
                    "generation": GEN_PROMPT_VERSION,
                    "j1": J1_PROMPT_VERSION,
                    "j2": J2_PROMPT_VERSION,
                    "j_meta": J_META_PROMPT_VERSION,
                    "j_lit": J_LIT_PROMPT_VERSION,
                    "j_humor": J_HUMOR_PROMPT_VERSION,
                    "j3": J3_PROMPT_VERSION,
                },
                "mode_boundary": "relation audits run only for structural; literary and humorous use metaphor-integrity plus dedicated expressive audits",
            }, ensure_ascii=False, indent=2))
            return

        if generation_jobs:
            print(f"generation jobs: {len(generation_jobs)}")
            await asyncio.gather(*(self._run_generation_job(*job) for job in generation_jobs))
        else:
            print("generation jobs: 0; using existing generations")

        await self._run_audits()
        if self.options.run_quality_pairs:
            await self._run_quality_pairs()

    async def _run_generation_job(
        self,
        run_id: str,
        ch: str,
        case: Case,
        generator: ProviderClient,
        temperature: float,
        arm: str,
        mapping_visibility: str,
        sample_index: int,
    ) -> None:
        async with self.sem:
            messages = make_generation_messages(case, arm, mapping_visibility=mapping_visibility)
            messages[-1]["content"] += (
                f"\n\nRUN_ID: {run_id}\nSAMPLE_INDEX: {sample_index}"
                f"\nCASE_HASH: {ch}\nPROMPT_VERSION: {GEN_PROMPT_VERSION}"
            )
            raw, _ = await complete_with_retry(generator, messages, temperature=temperature, retries=self.options.retries)
            self.db.insert_generation({
                "run_id": run_id,
                "case_id": case.case_id,
                "case_hash": ch,
                "stance_pattern": case.stance_pattern,
                "metaphor_mode": case.metaphor_mode,
                "vehicle_spec": case.vehicle_spec,
                "domain_distance": case.mapping.domain_distance,
                "mapping_visibility": mapping_visibility,
                "control_arm": arm,
                "provider": generator.name,
                "model": generator.model,
                "temperature": temperature,
                "sample_index": sample_index,
                "prompt_version": GEN_PROMPT_VERSION,
                "generated_text": raw.strip(),
                "raw_response": raw,
                "created_at": time.time(),
            })

    async def _run_audits(self) -> None:
        case_lookup = {ch: Case.from_dict(obj) for ch, obj in self.db.fetch_cases_by_hash().items()}
        jobs = []
        generations = self.db.fetch_generations()
        for gen in generations:
            case = case_lookup.get(gen["case_hash"])
            if case is None:
                # Legacy fallback. Current runs should always have case_hash matches.
                by_id = self.db.fetch_cases()
                case = Case.from_dict(by_id[gen["case_id"]])
            for judge_index, judge in enumerate(self.judges):
                for audit_type, prompt_version in self._audit_specs_for_case(case):
                    audit_id = stable_id(
                        "audit", prompt_version, gen["run_id"], audit_type,
                        judge.name, judge.model, judge_index,
                        prefix="aud_",
                    )
                    if not self.db.audit_exists(audit_id):
                        jobs.append((audit_id, audit_type, prompt_version, gen, case, judge, judge_index))
        if jobs:
            print(f"audit jobs: {len(jobs)}")
            await asyncio.gather(*(self._run_audit_job(*job) for job in jobs))
        else:
            print("audit jobs: 0; using existing audits")

    async def _run_audit_job(
        self,
        audit_id: str,
        audit_type: str,
        prompt_version: str,
        gen: Any,
        case: Case,
        judge: ProviderClient,
        judge_index: int,
    ) -> None:
        async with self.sem:
            mapping_visibility = gen["mapping_visibility"]
            if audit_type == "stance_leakage":
                messages = make_stance_leakage_audit_messages(case, gen["control_arm"], mapping_visibility, gen["generated_text"])
            elif audit_type == "relation":
                messages = make_relation_audit_messages(case, gen["control_arm"], mapping_visibility, gen["generated_text"])
            elif audit_type == "literary":
                messages = make_literary_audit_messages(case, gen["control_arm"], mapping_visibility, gen["generated_text"])
            elif audit_type == "humor":
                messages = make_humor_audit_messages(case, gen["control_arm"], mapping_visibility, gen["generated_text"])
            elif audit_type == "metaphor_integrity":
                messages = make_metaphor_integrity_audit_messages(case, gen["control_arm"], mapping_visibility, gen["generated_text"])
            else:
                raise ValueError(f"unknown audit_type: {audit_type}")
            messages[-1]["content"] += f"\n\nAUDIT_ID: {audit_id}\nPROMPT_VERSION: {prompt_version}"
            raw, parsed = await complete_with_retry(judge, messages, temperature=0.0, retries=self.options.retries, expect_json=True)
            parsed = parsed or {"parse_error": True, "raw": raw}
            parsed = apply_computed_pass_labels(parsed, audit_type)
            self.db.insert_audit({
                "audit_id": audit_id,
                "run_id": gen["run_id"],
                "audit_type": audit_type,
                "judge_provider": judge.name,
                "judge_model": judge.model,
                "judge_index": judge_index,
                "prompt_version": prompt_version,
                "raw_response": raw,
                "parsed_json": parsed,
                "created_at": time.time(),
            })

    async def _run_quality_pairs(self) -> None:
        case_lookup = {ch: Case.from_dict(obj) for ch, obj in self.db.fetch_cases_by_hash().items()}
        gens = self.db.fetch_generations()
        groups: dict[tuple[str, str, str], list[Any]] = {}
        for gen in gens:
            if gen["control_arm"] == "literal_paraphrase":
                continue
            groups.setdefault(
                (
                    gen["case_hash"],
                    gen["metaphor_mode"],
                    gen["vehicle_spec"],
                    gen["control_arm"],
                    gen["mapping_visibility"],
                ),
                [],
            ).append(gen)

        jobs = []
        rng = random.Random("quality-pairs")
        for (ch, metaphor_mode, vehicle_spec, control_arm, mapping_visibility), rows in groups.items():
            if len(rows) < 2:
                continue
            pairs = list(itertools.combinations(rows, 2))
            rng.shuffle(pairs)
            pairs = pairs[:max(0, self.options.quality_pairs_per_group)]
            case = case_lookup[ch]
            for a, b in pairs:
                if a["run_id"] > b["run_id"]:
                    a, b = b, a
                for judge_index, judge in enumerate(self.judges):
                    pair_id = stable_id(
                        "quality", J3_PROMPT_VERSION, a["run_id"], b["run_id"],
                        judge.name, judge.model, judge_index,
                        prefix="qual_",
                    )
                    if not self.db.quality_pair_exists(pair_id):
                        jobs.append((pair_id, case, ch, metaphor_mode, vehicle_spec, control_arm, mapping_visibility, a, b, judge, judge_index))
        if jobs:
            print(f"quality pair jobs: {len(jobs)}")
            await asyncio.gather(*(self._run_quality_pair_job(*job) for job in jobs))
        else:
            print("quality pair jobs: 0; using existing quality pairs")

    async def _run_quality_pair_job(
        self,
        pair_id: str,
        case: Case,
        ch: str,
        metaphor_mode: str,
        vehicle_spec: str,
        control_arm: str,
        mapping_visibility: str,
        a: Any,
        b: Any,
        judge: ProviderClient,
        judge_index: int,
    ) -> None:
        async with self.sem:
            messages = make_quality_pairwise_messages(case, a["generated_text"], b["generated_text"])
            messages[-1]["content"] += f"\n\nPAIR_ID: {pair_id}\nPROMPT_VERSION: {J3_PROMPT_VERSION}"
            raw, parsed = await complete_with_retry(judge, messages, temperature=0.0, retries=self.options.retries, expect_json=True)
            self.db.insert_quality_pair({
                "pair_id": pair_id,
                "case_id": case.case_id,
                "case_hash": ch,
                "metaphor_mode": metaphor_mode,
                "vehicle_spec": vehicle_spec,
                "mapping_visibility": mapping_visibility,
                "control_arm": control_arm,
                "text_a_run_id": a["run_id"],
                "text_b_run_id": b["run_id"],
                "judge_provider": judge.name,
                "judge_model": judge.model,
                "judge_index": judge_index,
                "prompt_version": J3_PROMPT_VERSION,
                "raw_response": raw,
                "parsed_json": parsed or {"parse_error": True, "raw": raw},
                "created_at": time.time(),
            })
