from __future__ import annotations

import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path

from metaphor_harness.db import HarnessDB
from metaphor_harness.eval_rules import (
    compute_pass_humor_quality,
    compute_pass_leakage,
    compute_pass_literary_quality,
    compute_pass_metaphor_integrity,
    compute_pass_relation,
    compute_pass_stance_pattern,
    majority_bool,
)
from metaphor_harness.io_utils import load_cases_jsonl
from metaphor_harness.prompts import make_generation_messages
from metaphor_harness.report import build_run_level_rows, write_report
from metaphor_harness.runner import HarnessRunner, RunOptions
from metaphor_harness.schema import Case, case_content_hash


ROOT = Path(__file__).resolve().parents[1]


class HarnessTests(unittest.TestCase):
    def test_generation_prompt_redacts_forbidden_and_mapping_when_hidden(self) -> None:
        case = load_cases_jsonl(ROOT / "data" / "seeds.jsonl")[0]
        self.assertEqual(case.metaphor_mode, "structural")
        self.assertEqual(case.vehicle_spec, "constrained")
        hidden = make_generation_messages(case, "metaphor_without_forbidden", mapping_visibility="hidden")[-1]["content"]
        scaffolded = make_generation_messages(case, "metaphor_with_forbidden", mapping_visibility="scaffolded")[-1]["content"]
        literal = make_generation_messages(case, "literal_paraphrase", mapping_visibility="scaffolded")[-1]["content"]

        for forbidden in case.forbidden_implications + case.target.forbidden_target_drift:
            self.assertNotIn(forbidden, hidden)
            self.assertNotIn(forbidden, literal)
            self.assertIn(forbidden, scaffolded)

        relation_name = case.mapping.target_relations[0]["name"]
        self.assertNotIn(relation_name, hidden)
        self.assertNotIn(relation_name, literal)
        self.assertIn(relation_name, scaffolded)
        self.assertNotIn('"vehicle"', literal)

    def test_open_vehicle_spec_keeps_target_scaffold_but_hides_vehicle_scaffold(self) -> None:
        base = load_cases_jsonl(ROOT / "data" / "seeds.jsonl")[0].to_dict()
        base["vehicle_spec"] = "open"
        case = Case.from_dict(base)
        prompt = make_generation_messages(case, "metaphor_with_forbidden", mapping_visibility="scaffolded")[-1]["content"]

        self.assertIn("VEHICLE_SPEC: open", prompt)
        self.assertIn(base["mapping"]["target_relations"][0]["name"], prompt)
        self.assertNotIn(base["mapping"]["desired_vehicle_relations"][0]["description"], prompt)
        self.assertNotIn(base["vehicle"]["instruction"], prompt)

    def test_expressive_cases_use_dedicated_audits_not_relation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            db_path = tmp / "expressive.sqlite"
            report_dir = tmp / "reports"
            opts = RunOptions(
                cases_path=str(ROOT / "data" / "expressive_seeds.jsonl"),
                config_path=str(ROOT / "config" / "providers.mock.json"),
                db_path=str(db_path),
                samples=1,
                temperatures=[0.2],
                control_arms=["metaphor_with_forbidden"],
                mapping_visibility=["hidden"],
                concurrency=16,
                retries=1,
                run_quality_pairs=False,
            )
            runner = HarnessRunner(opts)
            try:
                asyncio.run(runner.run())
            finally:
                runner.close()

            db = HarnessDB(db_path)
            try:
                rows = build_run_level_rows(db)
                self.assertEqual(len(rows), 4 * 3)
                self.assertTrue(any(r["metaphor_mode"] == "literary" for r in rows))
                self.assertTrue(any(r["metaphor_mode"] == "humorous" for r in rows))
                self.assertTrue(all(r["vehicle_spec"] == "open" for r in rows))
                self.assertTrue(any(r["literary_freshness_score_mean"] is not None for r in rows))
                self.assertTrue(any(r["humor_surprise_score_mean"] is not None for r in rows))
                audits = db.fetch_audits()
                audit_types = {a["audit_type"] for a in audits}
                self.assertIn("metaphor_integrity", audit_types)
                self.assertIn("literary", audit_types)
                self.assertIn("humor", audit_types)
                self.assertNotIn("relation", audit_types)
                self.assertTrue(any(r["pass_metaphor_integrity"] is not None for r in rows))
                self.assertTrue(all("metaphor_vehicle_affordance_broken" in r for r in rows))
                self.assertTrue(all("metaphor_licensed_rupture_success" in r for r in rows))
                self.assertTrue(all("literary_acceptable_metaphorical_ambiguity" in r for r in rows))
                self.assertTrue(all("humor_vehicle_or_medium_confusion" in r for r in rows))
            finally:
                db.close()

            write_report(str(db_path), str(report_dir))
            self.assertTrue((report_dir / "summary_expressive_modes.csv").exists())
            self.assertTrue((report_dir / "summary_metaphor_failure_tags.csv").exists())

    def test_case_hash_changes_when_case_content_changes(self) -> None:
        case = load_cases_jsonl(ROOT / "data" / "seeds.jsonl")[0]
        obj = case.to_dict()
        h1 = case_content_hash(obj)
        obj["target"]["claim"] = obj["target"]["claim"] + " 追加。"
        h2 = case_content_hash(obj)
        self.assertNotEqual(h1, h2)

    def test_pass_rules_are_deterministic(self) -> None:
        self.assertIs(True, compute_pass_stance_pattern({"stance_pattern_match": True, "stance_slip": False}))
        self.assertIs(False, compute_pass_stance_pattern({"stance_pattern_match": False, "stance_slip": False}))
        self.assertIs(False, compute_pass_leakage({
            "literal_vehicle_asserted": False,
            "unsupported_new_claim": False,
            "target_drift": False,
            "false_entailment_risk": 4,
        }))
        self.assertIs(True, compute_pass_leakage({
            "literal_vehicle_asserted": False,
            "unsupported_new_claim": False,
            "target_drift": False,
            "false_entailment_risk": 1,
        }))
        self.assertIs(False, compute_pass_relation({
            "relation_mapping_pass": True,
            "surface_only_mapping": True,
            "relation_score": 5,
        }))
        self.assertIs(True, compute_pass_metaphor_integrity({
            "target_anchor_preserved": True,
            "vehicle_affordance_coherent": True,
            "literal_scene_confusion": False,
            "semantic_break": False,
            "mode_fit": True,
            "premise_load_score": 2,
            "imageability_score": 4,
        }))
        self.assertIs(False, compute_pass_metaphor_integrity({
            "target_anchor_preserved": True,
            "vehicle_affordance_coherent": False,
            "literal_scene_confusion": False,
            "semantic_break": False,
            "mode_fit": True,
            "premise_load_score": 2,
            "imageability_score": 4,
        }))
        self.assertIs(False, compute_pass_metaphor_integrity({
            "target_anchor_preserved": True,
            "vehicle_affordance_coherent": True,
            "literal_scene_confusion": False,
            "semantic_break": False,
            "mode_fit": True,
            "premise_load_score": 2,
            "imageability_score": 4,
            "medium_dynamics_mismatch": True,
            "invariant_preserved": True,
        }))
        self.assertIs(True, compute_pass_metaphor_integrity({
            "target_anchor_preserved": True,
            "vehicle_affordance_coherent": True,
            "literal_scene_confusion": False,
            "semantic_break": False,
            "mode_fit": True,
            "premise_load_score": 4,
            "imageability_score": 4,
            "premise_overload": True,
            "licensed_rupture_success": True,
            "invariant_preserved": True,
        }))
        self.assertIs(False, compute_pass_metaphor_integrity({
            "target_anchor_preserved": True,
            "vehicle_affordance_coherent": True,
            "literal_scene_confusion": False,
            "semantic_break": False,
            "mode_fit": True,
            "premise_load_score": 2,
            "imageability_score": 4,
            "target_fact_drift": True,
            "licensed_rupture_success": True,
            "invariant_preserved": True,
        }))
        self.assertIs(None, majority_bool([True, False]))

    def test_expressive_quality_rules_distinguish_ambiguity_from_breaks(self) -> None:
        self.assertIs(True, compute_pass_literary_quality({
            "acceptable_metaphorical_ambiguity": True,
            "semantic_break": False,
            "target_drift": False,
            "vehicle_or_medium_confusion": False,
            "literal_scene_confusion": False,
            "mode_ambiguity": False,
            "premise_overload": False,
            "over_grandiose": False,
            "over_explanation": False,
            "register_mismatch": False,
            "freshness_score": 3,
            "tonal_fit_score": 4,
            "affect_transfer_score": 4,
            "poetic_compression_score": 4,
            "cliche_risk": 2,
        }))
        self.assertIs(False, compute_pass_literary_quality({
            "acceptable_metaphorical_ambiguity": False,
            "semantic_break": True,
            "target_drift": False,
            "vehicle_or_medium_confusion": False,
            "literal_scene_confusion": False,
            "mode_ambiguity": False,
            "premise_overload": False,
            "over_grandiose": False,
            "over_explanation": False,
            "register_mismatch": False,
            "freshness_score": 4,
            "tonal_fit_score": 4,
            "affect_transfer_score": 4,
            "poetic_compression_score": 4,
            "cliche_risk": 1,
        }))
        self.assertIs(True, compute_pass_humor_quality({
            "acceptable_metaphorical_ambiguity": True,
            "semantic_break": False,
            "target_drift": False,
            "vehicle_or_medium_confusion": False,
            "literal_scene_confusion": False,
            "mode_ambiguity": False,
            "premise_overload": False,
            "incongruity_success": True,
            "humor_flattening": False,
            "over_explanation": False,
            "surprise_score": 3,
            "tonal_fit_score": 3,
            "comic_timing_score": 4,
            "mean_spiritedness_risk": 1,
        }))
        self.assertIs(False, compute_pass_humor_quality({
            "acceptable_metaphorical_ambiguity": False,
            "semantic_break": False,
            "target_drift": True,
            "vehicle_or_medium_confusion": False,
            "literal_scene_confusion": False,
            "mode_ambiguity": False,
            "premise_overload": False,
            "incongruity_success": True,
            "humor_flattening": False,
            "over_explanation": False,
            "surprise_score": 5,
            "tonal_fit_score": 5,
            "comic_timing_score": 5,
            "mean_spiritedness_risk": 1,
        }))
        self.assertIs(False, compute_pass_humor_quality({
            "acceptable_metaphorical_ambiguity": False,
            "semantic_break": False,
            "target_drift": False,
            "vehicle_or_medium_confusion": True,
            "literal_scene_confusion": False,
            "mode_ambiguity": False,
            "premise_overload": False,
            "incongruity_success": True,
            "humor_flattening": False,
            "over_explanation": False,
            "surprise_score": 5,
            "tonal_fit_score": 5,
            "comic_timing_score": 5,
            "mean_spiritedness_risk": 1,
        }))

    def test_mock_runner_smoke_and_report_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            db_path = tmp / "runs.sqlite"
            report_dir = tmp / "reports"
            opts = RunOptions(
                cases_path=str(ROOT / "data" / "seeds.jsonl"),
                config_path=str(ROOT / "config" / "providers.mock.json"),
                db_path=str(db_path),
                samples=1,
                temperatures=[0.2],
                control_arms=["metaphor_with_forbidden", "metaphor_without_forbidden", "literal_paraphrase"],
                mapping_visibility=["hidden", "scaffolded"],
                concurrency=16,
                retries=1,
                run_quality_pairs=False,
            )
            runner = HarnessRunner(opts)
            try:
                asyncio.run(runner.run())
            finally:
                runner.close()

            db = HarnessDB(db_path)
            try:
                generations = db.fetch_generations()
                # literal_paraphrase collapses to hidden only: 8 cases * 3 generators * (2 metaphor arms*2 + 1 literal) = 120
                self.assertEqual(len(generations), 8 * 3 * 5)
                rows = build_run_level_rows(db)
                self.assertTrue(all("case_hash" in r for r in rows))
                self.assertTrue(all(r["metaphor_mode"] == "structural" for r in rows))
                self.assertTrue(all(r["vehicle_spec"] == "constrained" for r in rows))
                self.assertTrue(any(r["mapping_visibility"] == "scaffolded" for r in rows))
                self.assertTrue(any(r["control_arm"] == "literal_paraphrase" for r in rows))
            finally:
                db.close()

            write_report(str(db_path), str(report_dir))
            self.assertTrue((report_dir / "summary_by_stance_pattern.csv").exists())
            self.assertTrue((report_dir / "summary_by_metaphor_mode_vehicle_spec.csv").exists())
            self.assertTrue((report_dir / "summary_expressive_modes.csv").exists())
            self.assertTrue((report_dir / "summary_metaphor_failure_tags.csv").exists())
            self.assertTrue((report_dir / "summary_by_stance_distance_visibility_arm.csv").exists())
            self.assertTrue((report_dir / "summary_mapping_visibility_delta.csv").exists())
            self.assertTrue((report_dir / "summary_literal_controls.csv").exists())
            with (report_dir / "run_level.csv").open("r", encoding="utf-8", newline="") as f:
                first = next(csv.DictReader(f))
            self.assertIn("case_hash", first)
            self.assertIn("metaphor_mode", first)
            self.assertIn("vehicle_spec", first)
            self.assertIn("stance_pattern", first)
            self.assertIn("mapping_visibility", first)


if __name__ == "__main__":
    unittest.main()
