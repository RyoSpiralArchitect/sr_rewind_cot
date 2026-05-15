#!/usr/bin/env python3
"""Run fixed-trace prefix-interference matrices for sr_rewind_cot."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import sr_rewind_cot as sr


DEFAULT_SPEC = os.path.join(
    sr.SCRIPT_DIR,
    "sr_rewind_cot_assets",
    "trace_matrices",
    "syllogism_interference_matrix_v1.json",
)


def load_matrix_spec(path: str) -> Dict[str, Any]:
    spec = sr.read_json(path)
    if not isinstance(spec, dict):
        raise RuntimeError(f"Matrix spec must be a JSON object: {path}")
    cases = spec.get("cases")
    variants = spec.get("variants")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("Matrix spec requires a non-empty 'cases' list.")
    if not isinstance(variants, list) or not variants:
        raise RuntimeError("Matrix spec requires a non-empty 'variants' list.")
    if not str(spec.get("question_template", "")).strip():
        raise RuntimeError("Matrix spec requires 'question_template'.")
    for case in cases:
        if not isinstance(case, dict) or not str(case.get("id", "")).strip():
            raise RuntimeError("Every matrix case must be an object with an id.")
    for variant in variants:
        if not isinstance(variant, dict) or not str(variant.get("id", "")).strip():
            raise RuntimeError("Every matrix variant must be an object with an id.")
        if not isinstance(variant.get("steps"), list) or not variant.get("steps"):
            raise RuntimeError(f"Variant {variant.get('id')!r} requires non-empty steps.")
    return spec


def _format_template(template: str, values: Dict[str, Any]) -> str:
    try:
        return str(template).format(**values)
    except KeyError as exc:
        raise RuntimeError(f"Missing template key {exc} for values: {values}") from exc


def expand_matrix_items(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    question_template = str(spec.get("question_template", ""))
    for case in spec.get("cases", []) or []:
        case_id = str(case.get("id", "")).strip()
        question = _format_template(question_template, case)
        for variant in spec.get("variants", []) or []:
            variant_id = str(variant.get("id", "")).strip()
            steps = [_format_template(str(step), case) for step in (variant.get("steps") or [])]
            answer = _format_template(str(variant.get("answer", spec.get("expected_label", ""))), case)
            items.append({
                "case_id": case_id,
                "variant_id": variant_id,
                "question_id": f"{case_id}__{variant_id}",
                "question": question,
                "answer": answer,
                "steps": steps,
                "case": dict(case),
                "variant": dict(variant),
            })
    return items


def yes_no_label(answer: str) -> str:
    text = sr.normalize_answer(sr.semantic_text(answer))
    text = re.sub(r"^answer\s*[:=-]\s*", "", text, flags=re.IGNORECASE).strip()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return "other"


def _rate_row(
    *,
    matrix_id: str,
    temperature: float,
    case_id: str,
    variant_id: str,
    kind: str,
    k: int,
    answers: List[str],
) -> Dict[str, Any]:
    labels = [yes_no_label(answer) for answer in answers]
    counts = Counter(labels)
    top_answer = Counter(str(answer).strip() for answer in answers).most_common(1)[0][0] if answers else ""
    n = len(labels)
    return {
        "matrix_id": matrix_id,
        "temperature": temperature,
        "case_id": case_id,
        "variant_id": variant_id,
        "kind": kind,
        "k": k,
        "n": n,
        "yes": counts.get("yes", 0),
        "no": counts.get("no", 0),
        "other": counts.get("other", 0),
        "yes_rate": counts.get("yes", 0) / max(1, n),
        "no_rate": counts.get("no", 0) / max(1, n),
        "other_rate": counts.get("other", 0) / max(1, n),
        "top_answer": top_answer,
    }


def evaluate_item(
    backend: sr.Backend,
    item: Dict[str, Any],
    *,
    matrix_id: str,
    temperature: float,
    n: int,
    seed_base: int,
    prompt_version: str,
    prompt_family: str,
) -> Dict[str, Any]:
    steps = [str(step) for step in item.get("steps", [])]
    question = str(item.get("question", ""))
    rows: List[Dict[str, Any]] = []
    raw_by_k: List[Dict[str, Any]] = []

    baseline_seeds = [seed_base + 10_000 + i for i in range(n)]
    baseline_answers = sr.reanswer_many_from_prefix(
        backend,
        question,
        steps,
        len(steps),
        temperature=temperature,
        seeds=baseline_seeds,
        prompt_version=prompt_version,
        prompt_family=prompt_family,
    )
    rows.append(_rate_row(
        matrix_id=matrix_id,
        temperature=temperature,
        case_id=str(item["case_id"]),
        variant_id=str(item["variant_id"]),
        kind="baseline_full",
        k=len(steps),
        answers=baseline_answers,
    ))

    for k in range(len(steps) + 1):
        seeds = [seed_base + 200_000 + k * 1000 + i for i in range(n)]
        answers = sr.reanswer_many_from_prefix(
            backend,
            question,
            steps,
            k,
            temperature=temperature,
            seeds=seeds,
            prompt_version=prompt_version,
            prompt_family=prompt_family,
        )
        rows.append(_rate_row(
            matrix_id=matrix_id,
            temperature=temperature,
            case_id=str(item["case_id"]),
            variant_id=str(item["variant_id"]),
            kind="prefix",
            k=k,
            answers=answers,
        ))
        raw_by_k.append({"k": k, "answers": answers})

    return {
        "case_id": item["case_id"],
        "variant_id": item["variant_id"],
        "question_id": item["question_id"],
        "question": question,
        "trace": {
            "steps": steps,
            "answer": item.get("answer", ""),
        },
        "temperature": temperature,
        "n": n,
        "baseline_answers": baseline_answers,
        "raw_answers_by_k": raw_by_k,
        "rates": rows,
    }


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out_dir: str, spec: Dict[str, Any], results: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    rate_rows = [row for result in results for row in result.get("rates", [])]
    rate_fields = [
        "matrix_id",
        "temperature",
        "case_id",
        "variant_id",
        "kind",
        "k",
        "n",
        "yes",
        "no",
        "other",
        "yes_rate",
        "no_rate",
        "other_rate",
        "top_answer",
    ]
    write_csv(os.path.join(out_dir, "matrix_rates.csv"), rate_rows, rate_fields)

    summary_rows: List[Dict[str, Any]] = []
    for result in results:
        prefix_rows = [row for row in result["rates"] if row["kind"] == "prefix"]
        full_prefix = next((row for row in prefix_rows if int(row["k"]) == len(result["trace"]["steps"])), None)
        max_yes = max(prefix_rows, key=lambda row: float(row["yes_rate"])) if prefix_rows else None
        summary_rows.append({
            "matrix_id": spec.get("id", ""),
            "temperature": result["temperature"],
            "case_id": result["case_id"],
            "variant_id": result["variant_id"],
            "L": len(result["trace"]["steps"]),
            "n": result["n"],
            "baseline_yes_rate": result["rates"][0]["yes_rate"] if result.get("rates") else None,
            "baseline_no_rate": result["rates"][0]["no_rate"] if result.get("rates") else None,
            "full_prefix_yes_rate": (full_prefix or {}).get("yes_rate"),
            "full_prefix_no_rate": (full_prefix or {}).get("no_rate"),
            "max_prefix_yes_rate": (max_yes or {}).get("yes_rate"),
            "max_prefix_yes_k": (max_yes or {}).get("k"),
        })
    write_csv(
        os.path.join(out_dir, "matrix_summary.csv"),
        summary_rows,
        [
            "matrix_id",
            "temperature",
            "case_id",
            "variant_id",
            "L",
            "n",
            "baseline_yes_rate",
            "baseline_no_rate",
            "full_prefix_yes_rate",
            "full_prefix_no_rate",
            "max_prefix_yes_rate",
            "max_prefix_yes_k",
        ],
    )

    sr.write_json(os.path.join(out_dir, "matrix_results.json"), {
        "spec": spec,
        "args": vars(args),
        "results": results,
    })


def plot_heatmaps(out_dir: str, results: List[Dict[str, Any]]) -> None:
    if sr.plt is None:
        return
    plots_dir = sr.ensure_dir(os.path.join(out_dir, "plots"))
    temps = sorted({float(result["temperature"]) for result in results})
    for temp in temps:
        selected = [result for result in results if float(result["temperature"]) == temp]
        if not selected:
            continue
        max_k = max(len(result["trace"]["steps"]) for result in selected)
        row_labels = [f"{result['case_id']} / {result['variant_id']}" for result in selected]
        data: List[List[Optional[float]]] = []
        for result in selected:
            row = [None] * (max_k + 1)
            for rate in result["rates"]:
                if rate["kind"] != "prefix":
                    continue
                k = int(rate["k"])
                row[k] = float(rate["yes_rate"])
            data.append(row)

        arr = [[float("nan") if value is None else float(value) for value in row] for row in data]
        fig_w = max(7.0, 1.0 + 0.8 * (max_k + 1))
        fig_h = max(5.0, 0.38 * len(row_labels))
        fig, ax = sr.plt.subplots(figsize=(fig_w, fig_h))
        cmap = sr.plt.get_cmap("magma").copy()
        cmap.set_bad(color="#f1efe8")
        im = ax.imshow(arr, vmin=0.0, vmax=1.0, aspect="auto", cmap=cmap)
        ax.set_title(f"Yes-rate by prefix depth (temperature={temp:g})")
        ax.set_xlabel("Prefix depth k")
        ax.set_ylabel("Case / trace variant")
        ax.set_xticks(range(max_k + 1))
        ax.set_xticklabels([str(k) for k in range(max_k + 1)])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        for y, row in enumerate(data):
            for x, value in enumerate(row):
                if value is None:
                    continue
                color = "white" if value >= 0.55 else "black"
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
        fig.colorbar(im, ax=ax, label="P(yes)")
        fig.tight_layout()
        safe_temp = str(temp).replace(".", "p")
        fig.savefig(os.path.join(plots_dir, f"yes_rate_heatmap_t{safe_temp}.png"), dpi=180)
        sr.plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sr_rewind_cot_trace_matrix.py")
    p.add_argument("--spec", default=DEFAULT_SPEC)
    p.add_argument("--model", default="./model/llama-3.2-3b")
    p.add_argument("--name", default="llama-3.2-it")
    p.add_argument("--device", default="mps")
    p.add_argument("--hf-backend", default="mlx", choices=["auto", "torch", "mlx"])
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temps", default="0.0,0.4")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--prompt-version", default="v2")
    p.add_argument("--prompt-family", default="general_reasoning")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out-dir", default="")
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    spec = load_matrix_spec(args.spec)
    items = expand_matrix_items(spec)
    temps = [float(part.strip()) for part in str(args.temps).split(",") if part.strip()]
    prompt_version, prompt_family = sr.normalize_prompt_selector(args.prompt_version, args.prompt_family)
    out_dir = sr.ensure_dir(args.out_dir or os.path.join("results", f"run_{sr.now_ts()}_{spec.get('id', 'trace_matrix')}"))
    sr.write_json(os.path.join(out_dir, "run_meta.json"), {
        "created_at": datetime.now().isoformat(),
        "script": "sr_rewind_cot_trace_matrix.py",
        "spec_path": os.path.abspath(args.spec),
        "spec_id": spec.get("id"),
        "n_items": len(items),
        "temperatures": temps,
        "args": vars(args),
    })

    backend = sr.HFLocalBackend(
        name=args.name,
        model_path=args.model,
        tokenizer_path=None,
        device=args.device,
        hf_backend=args.hf_backend,
        torch_dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
    )

    results: List[Dict[str, Any]] = []
    for temp_index, temp in enumerate(temps):
        for item_index, item in enumerate(items):
            result = evaluate_item(
                backend,
                item,
                matrix_id=str(spec.get("id", "")),
                temperature=temp,
                n=max(1, int(args.n)),
                seed_base=int(args.seed) + temp_index * 1_000_000 + item_index * 10_000,
                prompt_version=prompt_version,
                prompt_family=prompt_family,
            )
            results.append(result)
            full = next(
                (
                    row for row in result["rates"]
                    if row["kind"] == "prefix" and int(row["k"]) == len(result["trace"]["steps"])
                ),
                None,
            )
            print(
                f"[matrix] temp={temp:g} {item['question_id']} "
                f"full_yes={float((full or {}).get('yes_rate', 0.0)):.2f}",
                flush=True,
            )

    write_outputs(out_dir, spec, results, args)
    if not args.no_plots:
        plot_heatmaps(out_dir, results)
    print(f"[matrix] saved: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
