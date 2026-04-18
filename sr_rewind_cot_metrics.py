#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metrics helpers and CLI for SR Rewind-CoT result inspection."""

from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence


def summarize_bridge_for_summary(res: Dict[str, Any], key: str) -> Dict[str, Any]:
    bundle = res.get(key) or {}
    if not isinstance(bundle, dict):
        return {
            f"{key}_best_auc_match": None,
            f"{key}_balanced_auc_match": None,
            f"{key}_best_synergy": None,
            f"{key}_positive_synergy_fraction": None,
            f"{key}_best_loop_closure": None,
            f"{key}_best_core_score": None,
            f"{key}_minimal_core_budget": None,
            f"{key}_minimal_core_loop_closure": None,
            f"{key}_top_core_step": None,
            f"{key}_top_core_support": None,
        }
    best_core = bundle.get("best_core_candidate") or {}
    minimal_core = bundle.get("minimal_core_candidate") or {}
    prototypes = bundle.get("core_step_prototypes") or []
    top_proto = prototypes[0] if prototypes and isinstance(prototypes[0], dict) else {}
    return {
        f"{key}_best_auc_match": ((bundle.get("budget_best_curve") or {}).get("auc_match") if isinstance(bundle.get("budget_best_curve"), dict) else None),
        f"{key}_balanced_auc_match": ((bundle.get("budget_balanced_curve") or {}).get("auc_match") if isinstance(bundle.get("budget_balanced_curve"), dict) else None),
        f"{key}_best_synergy": ((bundle.get("best_synergy_cell") or {}).get("synergy_over_max") if isinstance(bundle.get("best_synergy_cell"), dict) else None),
        f"{key}_positive_synergy_fraction": bundle.get("positive_synergy_fraction"),
        f"{key}_best_loop_closure": (best_core.get("loop_closure_score") if isinstance(best_core, dict) else None),
        f"{key}_best_core_score": (best_core.get("core_score") if isinstance(best_core, dict) else None),
        f"{key}_minimal_core_budget": (minimal_core.get("budget") if isinstance(minimal_core, dict) else None),
        f"{key}_minimal_core_loop_closure": (minimal_core.get("loop_closure_score") if isinstance(minimal_core, dict) else None),
        f"{key}_top_core_step": (top_proto.get("representative_step") if isinstance(top_proto, dict) else None),
        f"{key}_top_core_support": (top_proto.get("weighted_support") if isinstance(top_proto, dict) else None),
    }


def summarize_trace_vs_rewind_for_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    comp = res.get("trace_vs_rewind") or {}
    if not isinstance(comp, dict):
        return {
            "trace_vs_rewind_similarity_mean": None,
            "trace_vs_rewind_similarity_concat": None,
            "trace_vs_rewind_preservation_rate": None,
            "trace_vs_rewind_exact_match_count": None,
            "trace_vs_rewind_changed_count": None,
            "trace_vs_rewind_missing_count": None,
            "trace_vs_rewind_extra_count": None,
        }
    return {
        "trace_vs_rewind_similarity_mean": comp.get("similarity_mean"),
        "trace_vs_rewind_similarity_concat": comp.get("similarity_concat"),
        "trace_vs_rewind_preservation_rate": comp.get("preservation_rate"),
        "trace_vs_rewind_exact_match_count": comp.get("exact_match_count"),
        "trace_vs_rewind_changed_count": comp.get("changed_count"),
        "trace_vs_rewind_missing_count": comp.get("missing_count"),
        "trace_vs_rewind_extra_count": comp.get("extra_count"),
    }


def summarize_rewind_core_for_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    core = res.get("rewind_core") or {}
    if not isinstance(core, dict):
        return {
            "rewind_fixed_point_depth": None,
            "rewind_answer_reproduction_rate": None,
            "rewind_fixed_point_recurrence_rate": None,
            "rewind_pre_fixed_novelty_mean": None,
            "rewind_core_strength": None,
        }
    return {
        "rewind_fixed_point_depth": core.get("fixed_point_depth"),
        "rewind_answer_reproduction_rate": core.get("answer_reproduction_rate"),
        "rewind_fixed_point_recurrence_rate": core.get("fixed_point_recurrence_rate"),
        "rewind_pre_fixed_novelty_mean": core.get("pre_fixed_novelty_mean"),
        "rewind_core_strength": core.get("core_strength"),
    }


def summarize_rewind_process_reward_for_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    rewind_trace = res.get("rewind_trace") or {}
    if not isinstance(rewind_trace, dict):
        return {
            "rewind_process_reward_mode": None,
            "rewind_step_candidate_count": None,
            "rewind_process_reward_mean": None,
            "rewind_base_process_reward_mean": None,
            "rewind_process_reward_score_gain_vs_base": None,
        }
    return {
        "rewind_process_reward_mode": rewind_trace.get("process_reward_mode"),
        "rewind_step_candidate_count": rewind_trace.get("process_reward_step_candidates"),
        "rewind_process_reward_mean": rewind_trace.get("process_reward_mean"),
        "rewind_base_process_reward_mean": rewind_trace.get("process_reward_base_mean"),
        "rewind_process_reward_score_gain_vs_base": rewind_trace.get("process_reward_score_gain_vs_base"),
    }


def summarize_rewind_axes_for_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    comp = res.get("rewind_axes") or {}
    if not isinstance(comp, dict):
        return {
            "rewind_axis_base_prm_similarity_mean": None,
            "rewind_axis_base_prm_preservation_rate": None,
        }
    base_prm = comp.get("rewind_base_prm") or {}
    return {
        "rewind_axis_base_prm_similarity_mean": (base_prm.get("similarity_mean") if isinstance(base_prm, dict) else None),
        "rewind_axis_base_prm_preservation_rate": (base_prm.get("preservation_rate") if isinstance(base_prm, dict) else None),
    }


def summarize_runtime_for_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    stage = res.get("stage_timings_s") or {}
    rewind_timings = res.get("rewind_timings_s") or {}
    rewind_workload = res.get("rewind_workload") or {}
    return {
        "time_trace_build_s": (stage.get("trace_build_s") if isinstance(stage, dict) else None),
        "time_forward_curve_s": (stage.get("forward_curve_s") if isinstance(stage, dict) else None),
        "time_rewind_total_s": (rewind_timings.get("rewind_total_s") if isinstance(rewind_timings, dict) else stage.get("rewind_total_s") if isinstance(stage, dict) else None),
        "time_rewind_trace_s": (rewind_timings.get("rewind_trace_s") if isinstance(rewind_timings, dict) else None),
        "time_rewind_curve_s": (rewind_timings.get("rewind_curve_s") if isinstance(rewind_timings, dict) else None),
        "time_oracle_tail_curve_s": (rewind_timings.get("oracle_tail_curve_s") if isinstance(rewind_timings, dict) else None),
        "time_bridge_total_s": (stage.get("bridge_total_s") if isinstance(stage, dict) else None),
        "time_permutation_tests_s": (stage.get("permutation_tests_s") if isinstance(stage, dict) else None),
        "time_total_s": (stage.get("total_s") if isinstance(stage, dict) else None),
        "rewind_trace_generation_calls": (rewind_workload.get("rewind_trace_generation_calls") if isinstance(rewind_workload, dict) else None),
        "rewind_trace_attempts": (rewind_workload.get("rewind_trace_attempts") if isinstance(rewind_workload, dict) else None),
        "rewind_tail_answer_calls": (rewind_workload.get("rewind_tail_answer_calls") if isinstance(rewind_workload, dict) else None),
        "oracle_tail_answer_calls": (rewind_workload.get("oracle_tail_answer_calls") if isinstance(rewind_workload, dict) else None),
        "rewind_total_generation_calls": (rewind_workload.get("rewind_total_generation_calls") if isinstance(rewind_workload, dict) else None),
    }


def summarize_mlx_reuse_for_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    runtime_metrics = res.get("runtime_metrics") or {}
    reuse = ((runtime_metrics.get("mlx_prompt_reuse") or {}) if isinstance(runtime_metrics, dict) else {})
    if not isinstance(reuse, dict) or not reuse:
        return {
            "mlx_reuse_batch_calls": None,
            "mlx_reuse_batch_samples": None,
            "mlx_reuse_cache_reuse_calls": None,
            "mlx_reuse_cache_reuse_samples": None,
            "mlx_reuse_prompt_tokens_total": None,
            "mlx_reuse_prefix_tokens_total": None,
            "mlx_reuse_suffix_tokens_total": None,
            "mlx_reuse_naive_prefill_tokens_est": None,
            "mlx_reuse_actual_prefill_tokens_est": None,
            "mlx_reuse_saved_prefill_tokens_est": None,
            "mlx_reuse_saved_prefill_ratio_est": None,
            "mlx_reuse_cache_prepare_s_total": None,
            "mlx_reuse_batch_total_s": None,
            "mlx_reuse_cache_deepcopy_calls": None,
            "mlx_reuse_cache_deepcopy_s_total": None,
            "mlx_reuse_output_chars_total": None,
            "mlx_reuse_output_tokens_est_total": None,
            "mlx_reuse_output_tokens_per_s_est": None,
            "mlx_reuse_rewind_answer_calls": None,
            "mlx_reuse_rewind_answer_samples": None,
            "mlx_reuse_rewind_answer_saved_prefill_tokens_est": None,
            "mlx_reuse_rewind_answer_output_tokens_est_total": None,
        }
    naive_prefill = reuse.get("naive_prefill_tokens_est")
    saved_prefill = reuse.get("saved_prefill_tokens_est")
    ratio = None
    try:
        if naive_prefill is not None and float(naive_prefill) > 0.0:
            ratio = float(saved_prefill or 0.0) / float(naive_prefill)
    except Exception:
        ratio = None
    rewind_answer = ((reuse.get("stages") or {}).get("rewind_answer") or {}) if isinstance(reuse.get("stages"), dict) else {}
    output_tokens_per_s_est = None
    try:
        if float(reuse.get("batch_total_s") or 0.0) > 0.0:
            output_tokens_per_s_est = float(reuse.get("output_tokens_est_total") or 0.0) / float(reuse.get("batch_total_s") or 1.0)
    except Exception:
        output_tokens_per_s_est = None
    return {
        "mlx_reuse_batch_calls": reuse.get("batch_calls_total"),
        "mlx_reuse_batch_samples": reuse.get("batch_samples_total"),
        "mlx_reuse_cache_reuse_calls": reuse.get("cache_reuse_calls"),
        "mlx_reuse_cache_reuse_samples": reuse.get("cache_reuse_samples"),
        "mlx_reuse_prompt_tokens_total": reuse.get("prompt_tokens_total"),
        "mlx_reuse_prefix_tokens_total": reuse.get("prefix_tokens_total"),
        "mlx_reuse_suffix_tokens_total": reuse.get("suffix_tokens_total"),
        "mlx_reuse_naive_prefill_tokens_est": naive_prefill,
        "mlx_reuse_actual_prefill_tokens_est": reuse.get("actual_prefill_tokens_est"),
        "mlx_reuse_saved_prefill_tokens_est": saved_prefill,
        "mlx_reuse_saved_prefill_ratio_est": ratio,
        "mlx_reuse_cache_prepare_s_total": reuse.get("cache_prepare_s_total"),
        "mlx_reuse_batch_total_s": reuse.get("batch_total_s"),
        "mlx_reuse_cache_deepcopy_calls": reuse.get("cache_deepcopy_calls"),
        "mlx_reuse_cache_deepcopy_s_total": reuse.get("cache_deepcopy_s_total"),
        "mlx_reuse_output_chars_total": reuse.get("output_chars_total"),
        "mlx_reuse_output_tokens_est_total": reuse.get("output_tokens_est_total"),
        "mlx_reuse_output_tokens_per_s_est": output_tokens_per_s_est,
        "mlx_reuse_rewind_answer_calls": (rewind_answer.get("batch_calls_total") if isinstance(rewind_answer, dict) else None),
        "mlx_reuse_rewind_answer_samples": (rewind_answer.get("batch_samples_total") if isinstance(rewind_answer, dict) else None),
        "mlx_reuse_rewind_answer_saved_prefill_tokens_est": (rewind_answer.get("saved_prefill_tokens_est") if isinstance(rewind_answer, dict) else None),
        "mlx_reuse_rewind_answer_output_tokens_est_total": (rewind_answer.get("output_tokens_est_total") if isinstance(rewind_answer, dict) else None),
    }


def summarize_process_reward_for_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    prm = res.get("trace_process_reward") or {}
    if not isinstance(prm, dict):
        return {
            "process_reward_mode": None,
            "trace_candidate_count": None,
            "trace_selected_candidate_index": None,
            "trace_base_score": None,
            "trace_selected_score": None,
            "trace_score_gain_vs_base": None,
        }
    return {
        "process_reward_mode": prm.get("mode"),
        "trace_candidate_count": prm.get("candidate_count_ok"),
        "trace_selected_candidate_index": prm.get("selected_candidate_index"),
        "trace_base_score": prm.get("base_score"),
        "trace_selected_score": prm.get("selected_score"),
        "trace_score_gain_vs_base": prm.get("score_gain_vs_base"),
    }


def summarize_trace_axes_for_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    comp = res.get("trace_axes") or {}
    if not isinstance(comp, dict):
        return {
            "trace_axis_base_prm_similarity_mean": None,
            "trace_axis_base_prm_preservation_rate": None,
            "trace_axis_base_rewind_similarity_mean": None,
            "trace_axis_base_rewind_preservation_rate": None,
            "trace_axis_prm_rewind_similarity_mean": None,
            "trace_axis_prm_rewind_preservation_rate": None,
        }
    base_prm = comp.get("base_prm") or {}
    base_rewind = comp.get("base_rewind") or {}
    prm_rewind = comp.get("prm_rewind") or {}
    return {
        "trace_axis_base_prm_similarity_mean": (base_prm.get("similarity_mean") if isinstance(base_prm, dict) else None),
        "trace_axis_base_prm_preservation_rate": (base_prm.get("preservation_rate") if isinstance(base_prm, dict) else None),
        "trace_axis_base_rewind_similarity_mean": (base_rewind.get("similarity_mean") if isinstance(base_rewind, dict) else None),
        "trace_axis_base_rewind_preservation_rate": (base_rewind.get("preservation_rate") if isinstance(base_rewind, dict) else None),
        "trace_axis_prm_rewind_similarity_mean": (prm_rewind.get("similarity_mean") if isinstance(prm_rewind, dict) else None),
        "trace_axis_prm_rewind_preservation_rate": (prm_rewind.get("preservation_rate") if isinstance(prm_rewind, dict) else None),
    }


def to_csv(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())

    def esc(v: Any) -> str:
        if v is None:
            return ""
        s = str(v)
        if any(ch in s for ch in [",", "\"", "\n", "\r"]):
            s = "\"" + s.replace("\"", "\"\"") + "\""
        return s

    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(esc(row.get(k)) for k in keys))
    return "\n".join(lines) + "\n"


DEFAULT_SHOW_FIELDS = [
    "question_id",
    "L",
    "time_total_s",
    "time_rewind_total_s",
    "time_rewind_curve_s",
    "time_oracle_tail_curve_s",
    "rewind_total_generation_calls",
    "mlx_reuse_saved_prefill_tokens_est",
    "mlx_reuse_saved_prefill_ratio_est",
    "mlx_reuse_cache_prepare_s_total",
    "mlx_reuse_cache_deepcopy_s_total",
    "mlx_reuse_output_tokens_est_total",
    "mlx_reuse_output_tokens_per_s_est",
    "mlx_reuse_batch_total_s",
]


def resolve_summary_path(path_or_dir: str) -> pathlib.Path:
    p = pathlib.Path(path_or_dir)
    if p.is_dir():
        return p / "summary.csv"
    return p


def read_summary_rows(path_or_dir: str) -> List[Dict[str, str]]:
    path = resolve_summary_path(path_or_dir)
    if not path.exists():
        raise FileNotFoundError(f"summary.csv not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric_aggregate(rows: Iterable[Dict[str, str]], fields: Sequence[str]) -> Dict[str, float]:
    agg: Dict[str, float] = defaultdict(float)
    for row in rows:
        for field in fields:
            try:
                agg[field] += float(row[field])
            except Exception:
                continue
    return dict(agg)


def format_rows(rows: List[Dict[str, str]], fields: Sequence[str]) -> str:
    selected = [{field: row.get(field) for field in fields} for row in rows]
    return to_csv(selected)


def cmd_show(args: argparse.Namespace) -> int:
    rows = read_summary_rows(args.run)
    fields = args.fields or DEFAULT_SHOW_FIELDS
    print(format_rows(rows, fields), end="")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    left_rows = read_summary_rows(args.left)
    right_rows = read_summary_rows(args.right)
    fields = args.fields or DEFAULT_SHOW_FIELDS
    left_by_qid = {row.get("question_id"): row for row in left_rows}
    right_by_qid = {row.get("question_id"): row for row in right_rows}
    merged: List[Dict[str, Any]] = []
    for qid in sorted(set(left_by_qid) | set(right_by_qid)):
        left = left_by_qid.get(qid, {})
        right = right_by_qid.get(qid, {})
        row: Dict[str, Any] = {"question_id": qid}
        for field in fields:
            if field == "question_id":
                continue
            row[f"{field}_left"] = left.get(field)
            row[f"{field}_right"] = right.get(field)
            try:
                row[f"{field}_delta"] = float(right.get(field) or 0.0) - float(left.get(field) or 0.0)
            except Exception:
                row[f"{field}_delta"] = None
        merged.append(row)
    print(to_csv(merged), end="")
    if args.aggregate:
        numeric_fields = [field for field in fields if field != "question_id"]
        left_agg = numeric_aggregate(left_rows, numeric_fields)
        right_agg = numeric_aggregate(right_rows, numeric_fields)
        agg_row: Dict[str, Any] = {"question_id": "ALL"}
        for field in numeric_fields:
            agg_row[f"{field}_left"] = left_agg.get(field)
            agg_row[f"{field}_right"] = right_agg.get(field)
            agg_row[f"{field}_delta"] = right_agg.get(field, 0.0) - left_agg.get(field, 0.0)
        print("\n# aggregate")
        print(to_csv([agg_row]), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sr_rewind_cot_metrics.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="Show selected summary.csv columns for one run.")
    p_show.add_argument("--run", required=True, help="Run directory or summary.csv path.")
    p_show.add_argument("--fields", nargs="*", default=None, help="Optional columns to print.")
    p_show.set_defaults(func=cmd_show)

    p_compare = sub.add_parser("compare", help="Compare two run summaries question-by-question.")
    p_compare.add_argument("--left", required=True, help="Left run directory or summary.csv path.")
    p_compare.add_argument("--right", required=True, help="Right run directory or summary.csv path.")
    p_compare.add_argument("--fields", nargs="*", default=None, help="Optional columns to compare.")
    p_compare.add_argument("--aggregate", action="store_true", help="Also print aggregate numeric deltas.")
    p_compare.set_defaults(func=cmd_compare)
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
