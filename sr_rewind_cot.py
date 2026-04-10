#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SR Rewind-CoT Harness (HF + OpenAI-compatible HTTP).

What it does
- Builds a reasoning trace (masked/unmasked) as JSON: {"steps":[...], "answer":"..."}
- Re-answers from prefix-k steps to estimate stability curve:
    match_rate(k) = P(A(k) == A*)
    entropy_bits(k) over answer strings
- Adds research metrics:
    - baseline stability gating (min confidence / max entropy)
    - robust converge/collapse points with smoothing + min-run
    - divergence vs baseline distribution (JS, KL)
    - stabilization speed (max slope of smoothed match rate)
    - attractor strength (early convergence × plateau quality)
    - hysteresis (collapse_k - converge_k)

Supports
- OpenAI-compatible HTTP endpoints (vLLM, Ollama, LiteLLM proxy, patapi, OpenAI, etc.)
- Local Hugging Face transformers models (AutoModelForCausalLM + AutoTokenizer)
  with robust tokenizer discovery.
- External English prompt templates under ./sr_rewind_cot_assets/prompts/

Usage (quick HF):
  python3 sr_rewind_cot.py quick-hf \
    --model ./model/llama-3.2-3b --device mps --dtype float16 --name llama \
    --question "Find the general term of the sequence: 1, 4, 9, 16, ..." \
    --prompt-version v2 --prompt-family math_reasoning \
    --temp-reanswer 0.8 --n-per-k 7

Usage (run with config):
  python3 sr_rewind_cot.py run --config experiment.yaml

Config example is shown via:
  python3 sr_rewind_cot.py sample-config

Dependencies
- minimal: requests, pyyaml (optional), matplotlib (optional)
- HF: transformers, torch (accelerate/bitsandbytes optional)
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Protocol, Union
import json
import os
import sys
import time
import math
import random
import re
import hashlib
import pathlib
import tempfile
import shutil
import threading
from difflib import SequenceMatcher
from collections import Counter
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "sr_rewind_cot_assets", "prompts")
SAFE_PROMPT_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
PROMPT_VERSION_DEFAULT_FAMILY = {
    "v1": "default",
    "v2": "general_reasoning",
}
GENERIC_QUESTION_PATTERNS = [
    re.compile(r"^\s*solve(?:\s+(?:the|this))?\s+puzzle\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*solve(?:\s+(?:the|this))?\s+problem\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*answer(?:\s+(?:the|this))?\s+question\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*find\s+the\s+answer\.?\s*$", re.IGNORECASE),
]
TRACE_OUTPUT_FORMATS = {"auto", "json", "text"}
HF_BACKEND_CHOICES = {"auto", "torch", "mlx"}
TRACE_ANSWER_LINE_RE = re.compile(r"^(?:final answer|answer|conclusion)\s*[:=-]\s*(.+)$", re.IGNORECASE)
TRACE_STEP_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\(?\d+\)?[.)]|step\s*\d+\s*[:.)-]?|[A-Za-z][.)])\s*", re.IGNORECASE)
CONTENT_WORD_RE = re.compile(r"[A-Za-z0-9']+")
COMMON_STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "into", "their", "there",
    "then", "than", "they", "them", "have", "will", "would", "could", "should",
    "about", "which", "while", "where", "when", "what", "your", "you", "are",
    "was", "were", "been", "being", "because", "under", "over", "also", "does",
    "doesn't", "dont", "not", "only", "more", "most", "some", "such", "just",
    "into", "onto", "between", "within", "after", "before", "these", "those",
    "have", "has", "had", "its", "it's", "our", "out", "due", "lack", "proof",
}

# Optional deps
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import requests  # type: ignore
except Exception as e:
    print("ERROR: requests is required. Install with: pip install requests", file=sys.stderr)
    raise

# Plotting is optional (only needed for plot subcommand)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


# ----------------------------- utilities -----------------------------

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_text(path: str, s: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_probably_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def normalize_answer(a: str) -> str:
    a = (a or "").strip().lower()
    a = " ".join(a.split())
    while a and a[-1] in ".。!！?？,，:：;；":
        a = a[:-1].strip()
    return a

def looks_like_placeholder_question(question: str) -> bool:
    text = str(question or "").strip()
    if not text:
        return True
    if len(text) > 64:
        return False
    return any(p.fullmatch(text) for p in GENERIC_QUESTION_PATTERNS)

def classify_trace_generation_failure(question: str, raw: str) -> Optional[str]:
    raw_norm = " ".join(str(raw or "").strip().lower().split())
    question_text = str(question or "").strip()
    if looks_like_placeholder_question(question_text):
        return (
            "The question text looks like a placeholder rather than the full task.\n"
            f"- received question: {question_text!r}\n"
            "Pass the complete puzzle/problem statement in --question.\n"
            "Example: --question 'Three people A, B, and C make statements ... Who is lying?'"
        )
    missing_context_markers = [
        "don't see the puzzle",
        "do not see the puzzle",
        "please provide the puzzle",
        "please provide the problem",
        "please provide the question",
        "i don't see the puzzle",
        "i do not see the puzzle",
        "not enough information",
        "insufficient information",
        "underspecified",
        "need more information",
    ]
    if any(marker in raw_norm for marker in missing_context_markers):
        return (
            "The model is asking for missing task content, so the input question is probably too vague.\n"
            f"- received question: {question_text!r}\n"
            "Pass the full puzzle/problem text in --question instead of a short label."
        )
    return None

def _normalize_prompt_token(value: Optional[str], label: str) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    if not SAFE_PROMPT_TOKEN_RE.fullmatch(s):
        raise RuntimeError(f"Invalid {label}: {s!r}. Use only letters, numbers, '_' or '-'.")
    return s

def normalize_prompt_selector(prompt_version: Optional[str], prompt_family: Optional[str]) -> Tuple[str, str]:
    version = _normalize_prompt_token(prompt_version or "v1", "prompt_version") or "v1"
    default_family = PROMPT_VERSION_DEFAULT_FAMILY.get(version, "default")
    family = _normalize_prompt_token(prompt_family, "prompt_family") or default_family
    if version == "v1":
        family = default_family
    elif family == "default":
        family = default_family
    return version, family

def normalize_trace_output_format(value: Optional[str]) -> str:
    s = str(value or "auto").strip().lower()
    return s if s in TRACE_OUTPUT_FORMATS else "auto"

def resolve_trace_output_format(value: Optional[str], *, stream_output: bool = False) -> str:
    fmt = normalize_trace_output_format(value)
    if fmt == "auto":
        return "text" if stream_output else "json"
    return fmt

def normalize_hf_backend_choice(value: Optional[str]) -> str:
    s = str(value or "auto").strip().lower()
    return s if s in HF_BACKEND_CHOICES else "auto"

def list_prompt_families(prompt_version: Optional[str] = None) -> List[str]:
    version, default_family = normalize_prompt_selector(prompt_version or "v1", None)
    version_dir = pathlib.Path(PROMPTS_DIR) / version
    families = sorted(p.name for p in version_dir.iterdir() if p.is_dir()) if version_dir.is_dir() else []
    if version == "v1" and not families:
        return [default_family]
    return families or [default_family]

def resolve_prompt_template_path(
    name: str,
    *,
    prompt_version: str = "v1",
    prompt_family: Optional[str] = None,
) -> str:
    base_name = str(name or "").strip()
    if not base_name:
        raise RuntimeError("Prompt template name is required.")
    if "/" in base_name or "\\" in base_name:
        raise RuntimeError(f"Prompt template name must be a file name, got: {base_name!r}")
    stem, ext = os.path.splitext(base_name)
    file_stem = _normalize_prompt_token(stem, "prompt_template")
    filename = file_stem + (ext or ".txt")
    if filename.endswith("..txt"):
        filename = file_stem + ".txt"

    version, family = normalize_prompt_selector(prompt_version, prompt_family)
    default_family = PROMPT_VERSION_DEFAULT_FAMILY.get(version, "default")
    candidates: List[str] = []

    if version == "v1":
        candidates.append(os.path.join(PROMPTS_DIR, filename))
        candidates.append(os.path.join(PROMPTS_DIR, "v1", default_family, filename))
    else:
        candidates.append(os.path.join(PROMPTS_DIR, version, family, filename))
        if family != default_family:
            candidates.append(os.path.join(PROMPTS_DIR, version, default_family, filename))

    for path in candidates:
        if os.path.exists(path):
            return path

    families = ", ".join(list_prompt_families(version))
    tried = "\n".join(f"  - {p}" for p in candidates)
    raise RuntimeError(
        f"Prompt template not found for version={version!r}, family={family!r}, template={filename!r}.\n"
        f"Tried:\n{tried}\n"
        f"Available families for {version}: {families}"
    )

def load_prompt_family_guidance(
    *,
    prompt_version: str = "v1",
    prompt_family: Optional[str] = None,
) -> str:
    version, family = normalize_prompt_selector(prompt_version, prompt_family)
    guidance_path = os.path.join(PROMPTS_DIR, version, family, "_guidance.txt")
    if os.path.exists(guidance_path):
        return read_text(guidance_path).strip()
    if version == "v1":
        return ""
    default_family = PROMPT_VERSION_DEFAULT_FAMILY.get(version, "default")
    fallback_path = os.path.join(PROMPTS_DIR, version, default_family, "_guidance.txt")
    if os.path.exists(fallback_path):
        return read_text(fallback_path).strip()
    return ""

def load_prompt_template(
    name: str,
    *,
    prompt_version: str = "v1",
    prompt_family: Optional[str] = None,
) -> str:
    path = resolve_prompt_template_path(name, prompt_version=prompt_version, prompt_family=prompt_family)
    return read_text(path).strip()

def render_prompt_template(
    name: str,
    *,
    prompt_version: str = "v1",
    prompt_family: Optional[str] = None,
    **values: Any,
) -> str:
    text = load_prompt_template(name, prompt_version=prompt_version, prompt_family=prompt_family)
    if "FAMILY_GUIDANCE" not in values:
        values["FAMILY_GUIDANCE"] = load_prompt_family_guidance(
            prompt_version=prompt_version,
            prompt_family=prompt_family,
        )
    for key, value in values.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", text)))
    if unresolved:
        raise RuntimeError(f"Unresolved placeholders in prompt template '{name}': {unresolved}")
    return text.strip()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def moving_average(xs: List[float], window: int) -> List[float]:
    if window <= 1:
        return xs[:]
    w = min(window, len(xs))
    out = []
    acc = 0.0
    q: List[float] = []
    for x in xs:
        q.append(x)
        acc += x
        if len(q) > w:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out

def entropy_bits_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p + 1e-12, 2)
    return h

def dist_from_answers(ans: List[str]) -> Counter[str]:
    return Counter(normalize_answer(a) for a in ans)

def majority_vote(ans: List[str]) -> Tuple[str, float]:
    c = dist_from_answers(ans)
    if not c:
        return "", 0.0
    top, cnt = c.most_common(1)[0]
    return top, cnt / max(1, sum(c.values()))

def _to_prob(counts: Counter[str], support: List[str], alpha: float = 1e-6) -> List[float]:
    """Convert counts into probabilities over a fixed support with additive smoothing."""
    total = sum(counts.values())
    denom = total + alpha * len(support)
    return [ (counts.get(k, 0) + alpha) / denom for k in support ]

def kl_divergence(p: List[float], q: List[float]) -> float:
    """D_KL(p||q) in nats."""
    d = 0.0
    for pi, qi in zip(p, q):
        d += pi * math.log((pi + 1e-12) / (qi + 1e-12))
    return d

def js_divergence(p: List[float], q: List[float]) -> float:
    """Jensen-Shannon divergence in nats."""
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def find_converge_k(series: List[float], tau: float, min_run: int) -> Optional[int]:
    """First k where series stays >= tau for min_run consecutive points."""
    if not series:
        return None
    run = 0
    for k, v in enumerate(series):
        if v >= tau:
            run += 1
            if run >= min_run:
                return k - min_run + 1
        else:
            run = 0
    return None

def find_collapse_k(series: List[float], tau: float, min_run: int) -> Optional[int]:
    """
    Collapse point scanning from high k to low k:
    Find first run of length min_run where series < tau (when going downward).
    Return last stable k just above that run (k_run_start + min_run).
    """
    if not series:
        return None
    below = 0
    # k desc
    for k in range(len(series) - 1, -1, -1):
        if series[k] < tau:
            below += 1
            if below >= min_run:
                # run covers k..k+min_run-1, so last stable is k+min_run
                cand = k + min_run
                if cand <= len(series) - 1:
                    return cand
                # if run hits the top end (unlikely), no stable segment
                return None
        else:
            below = 0
    return None

def discrete_derivative(series: List[float]) -> List[float]:
    if len(series) < 2:
        return []
    return [series[i+1] - series[i] for i in range(len(series)-1)]

def safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

def safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extracts the first JSON object from a string.
    - Finds the first '{' and then tries incremental parsing by balancing braces.
    """
    if not text:
        return None
    # quick path: whole string JSON
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # incremental brace matching
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = t[start:i+1]
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue
    return None

def decode_jsonish_string(text: str) -> str:
    s = str(text or "")
    try:
        return json.loads(f'"{s}"')
    except Exception:
        s = s.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
        return s

def extract_jsonish_array_for_key(text: str, key: str) -> Optional[str]:
    raw = str(text or "")
    m = re.search(rf'"{re.escape(key)}"\s*:\s*\[', raw)
    if m is None:
        return None
    start = raw.find("[", m.start())
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    return None

def extract_trace_payload_fallback(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "")
    if not raw:
        return None

    step_matches: List[str] = []

    steps_array_chunk = extract_jsonish_array_for_key(raw, "steps")
    if steps_array_chunk is not None:
        try:
            parsed_steps = json.loads(steps_array_chunk)
            step_matches = normalize_trace_steps(parsed_steps)
        except Exception:
            step_matches = []

    if not step_matches:
        step_matches = [
            decode_jsonish_string(m.group(1))
            for m in re.finditer(r'"step"\s*:\s*"((?:\\.|[^"\\])*)"', raw, flags=re.DOTALL)
        ]
    if not step_matches:
        return None

    answer = ""
    for key in ("answer", "final_answer", "final", "output"):
        m = re.search(rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"', raw, flags=re.DOTALL)
        if m is not None:
            answer = decode_jsonish_string(m.group(1)).strip()
            if answer:
                break

    return {
        "steps": step_matches,
        "answer": answer,
    }

def strip_markdown_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

def clean_trace_line(line: str) -> str:
    s = str(line or "").strip()
    if not s:
        return ""
    s = TRACE_STEP_PREFIX_RE.sub("", s).strip()
    while s and s[0] in "\"'":
        s = s[1:].strip()
    while s and s[-1] in "\"'":
        s = s[:-1].strip()
    return s

def extract_trace_payload_from_plaintext(text: str) -> Optional[Dict[str, Any]]:
    raw = strip_markdown_code_fence(text)
    if not raw:
        return None

    lines: List[str] = []
    answer = ""
    for raw_line in raw.splitlines():
        s = raw_line.strip()
        if not s:
            continue
        if s in {"{", "}", "[", "]"}:
            continue
        if s.startswith('"steps"') or s.startswith('"answer"'):
            continue
        m = TRACE_ANSWER_LINE_RE.match(s)
        if m is not None:
            answer = clean_trace_line(m.group(1))
            continue
        cleaned = clean_trace_line(s)
        if not cleaned:
            continue
        if cleaned.lower() in {"trace:", "reasoning:", "steps:"}:
            continue
        lines.append(cleaned)

    if not lines:
        return None

    if not answer:
        if len(lines) >= 2:
            answer = lines[-1]
            lines = lines[:-1]
        else:
            answer = lines[0]

    steps = [line for line in lines if line and normalize_answer(line) != normalize_answer(answer)]
    if not steps and lines:
        steps = lines[:1]
    if not steps:
        return None

    return {
        "steps": steps,
        "answer": answer,
    }

def normalize_trace_step_records(raw_steps: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_steps, list):
        raise RuntimeError(f"Trace JSON 'steps' is not a list: {type(raw_steps)}")

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_steps, start=1):
        text = ""
        record: Dict[str, Any] = {}
        if isinstance(item, str):
            text = item
            record = {"id": idx, "type": "step"}
        elif isinstance(item, dict):
            record = dict(item)
            for key in ("step", "text", "content", "reasoning"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    text = value
                    break
            record["id"] = int(item.get("id", idx)) if str(item.get("id", idx)).isdigit() else idx
            record["type"] = str(item.get("type") or item.get("kind") or "step").strip() or "step"
        else:
            text = str(item)
            record = {"id": idx, "type": "step"}

        text = text.strip()
        if not text:
            continue
        record["text"] = text
        out.append(record)
    return out

def normalize_trace_steps(raw_steps: Any) -> List[str]:
    return [record["text"] for record in normalize_trace_step_records(raw_steps)]

def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))

def trace_to_payload(trace: "Trace", *, include_metadata: bool = False) -> Dict[str, Any]:
    payload = {
        "steps": list(trace.steps),
        "step_records": list(trace.step_records or []),
        "answer": trace.answer,
    }
    if include_metadata and trace.metadata:
        payload["metadata"] = trace.metadata
    return payload

def trace_from_payload(payload: Dict[str, Any]) -> "Trace":
    raw_steps = payload.get("step_records") if isinstance(payload.get("step_records"), list) else payload.get("steps", [])
    step_records = normalize_trace_step_records(raw_steps)
    return Trace(
        steps=[record["text"] for record in step_records],
        answer=str(payload.get("answer", "")),
        step_records=step_records,
        metadata=(dict(payload.get("metadata")) if isinstance(payload.get("metadata"), dict) else None),
    )

def extract_trace_answer(obj: Dict[str, Any]) -> str:
    for key in ("answer", "final_answer", "final", "output"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(obj.get("answer", "")).strip()

def mean_or_none(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None

def sequence_similarity(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(a=normalize_answer(a), b=normalize_answer(b)).ratio()

def content_words(text: str) -> set[str]:
    words = {
        token
        for token in CONTENT_WORD_RE.findall(normalize_answer(text))
        if len(token) >= 3 and token not in COMMON_STOPWORDS
    }
    return words

def content_jaccard_similarity(a: str, b: str) -> float:
    wa = content_words(a)
    wb = content_words(b)
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / max(1, union)

def step_semantic_similarity(a: str, b: str) -> float:
    return max(sequence_similarity(a, b), content_jaccard_similarity(a, b))

def first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        s = line.strip().lstrip("-*•").strip()
        if s:
            return s
    return (text or "").strip()

def infer_generation_stage(prompt: str) -> str:
    text = str(prompt or "")
    if "Create a reasoning trace" in text:
        return "trace"
    if "Recover EXACTLY ONE short reasoning step" in text:
        return "rewind_step"
    if "Reconstruct ONLY the missing middle reasoning" in text:
        return "bridge_middle"
    if "Infer a coherent missing middle segment" in text:
        return "bridge_answer"
    if "Partial rewind trace" in text:
        return "rewind_answer"
    if "Partial trace (first" in text and "give ONLY the final answer" in text:
        return "forward_reanswer"
    return "unknown"

class GenerationLogger:
    def __init__(
        self,
        backend_name: str,
        question_id: str,
        question: str,
        prompt_version: str,
        prompt_family: str,
    ) -> None:
        self.backend_name = backend_name
        self.question_id = question_id
        self.question = question
        self.prompt_version = prompt_version
        self.prompt_family = prompt_family
        self.rows: List[Dict[str, Any]] = []
        self._event_index = 0

    def log(
        self,
        *,
        stage: str,
        prompt: str,
        temperature: float,
        seed: Optional[int],
        output: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self._event_index += 1
        self.rows.append({
            "event_index": self._event_index,
            "backend": self.backend_name,
            "question_id": self.question_id,
            "question": self.question,
            "prompt_version": self.prompt_version,
            "prompt_family": self.prompt_family,
            "stage": stage,
            "temperature": float(temperature),
            "seed": seed,
            "prompt_chars": len(prompt or ""),
            "output_chars": len(output or ""),
            "status": "error" if error is not None else "ok",
            "error": error,
            "prompt": prompt,
            "output": output,
        })

    def write_jsonl(self, path: str) -> None:
        write_jsonl(path, self.rows)

class LoggedBackend:
    def __init__(self, inner: Backend, logger: GenerationLogger) -> None:
        self.inner = inner
        self.logger = logger
        self.name = getattr(inner, "name", "backend")

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.inner, attr)

    def generate(self, prompt: str, temperature: float, seed: Optional[int] = None) -> str:
        stage = infer_generation_stage(prompt)
        try:
            out = self.inner.generate(prompt, temperature=temperature, seed=seed)
        except Exception as e:
            self.logger.log(
                stage=stage,
                prompt=prompt,
                temperature=temperature,
                seed=seed,
                error=str(e),
            )
            raise
        self.logger.log(
            stage=stage,
            prompt=prompt,
            temperature=temperature,
            seed=seed,
            output=out,
        )
        return out

    def stream_generate(self, prompt: str, temperature: float, seed: Optional[int] = None) -> Iterator[str]:
        stage = infer_generation_stage(prompt)
        stream_fn = getattr(self.inner, "stream_generate", None)
        if not callable(stream_fn):
            out = self.generate(prompt, temperature=temperature, seed=seed)
            yield out
            return

        parts: List[str] = []
        try:
            for chunk in stream_fn(prompt, temperature=temperature, seed=seed):
                text = str(chunk or "")
                if not text:
                    continue
                parts.append(text)
                yield text
        except Exception as e:
            self.logger.log(
                stage=stage,
                prompt=prompt,
                temperature=temperature,
                seed=seed,
                output="".join(parts) if parts else None,
                error=str(e),
            )
            raise
        self.logger.log(
            stage=stage,
            prompt=prompt,
            temperature=temperature,
            seed=seed,
            output="".join(parts),
        )

def emit_streamed_generation(
    backend: Backend,
    prompt: str,
    temperature: float,
    seed: Optional[int] = None,
    *,
    enable_stream: bool = False,
    label: Optional[str] = None,
) -> str:
    if enable_stream and label:
        print(f"\n=== {label} ===", flush=True)

    if enable_stream:
        stream_fn = getattr(backend, "stream_generate", None)
        if callable(stream_fn):
            parts: List[str] = []
            for chunk in stream_fn(prompt, temperature=temperature, seed=seed):
                text = str(chunk or "")
                if not text:
                    continue
                parts.append(text)
                print(text, end="", flush=True)
            print("", flush=True)
            return "".join(parts)

    out = backend.generate(prompt, temperature=temperature, seed=seed)
    if enable_stream:
        print(out, flush=True)
    return out

@contextlib.contextmanager
def temporary_backend_generation_overrides(backend: Backend, **overrides: Any) -> Iterator[None]:
    target = getattr(backend, "inner", backend)
    restore: Dict[str, Any] = {}
    applied: List[str] = []
    for key, value in overrides.items():
        if value is None or not hasattr(target, key):
            continue
        restore[key] = getattr(target, key)
        setattr(target, key, value)
        applied.append(key)
    try:
        yield
    finally:
        for key in applied:
            setattr(target, key, restore[key])

def parse_rewind_step_output(raw: str) -> Dict[str, Any]:
    obj = extract_first_json_object(raw)
    if obj is not None:
        step = str(obj.get("step") or "").strip()
        novelty = str(obj.get("novelty") or "").strip()
        why_earlier = str(obj.get("why_earlier") or "").strip()
        return {
            "step": step,
            "novelty": novelty,
            "why_earlier": why_earlier,
            "raw_obj": obj,
        }
    return {
        "step": first_nonempty_line(raw),
        "novelty": "",
        "why_earlier": "",
        "raw_obj": None,
    }

def rewind_step_novelty_info(
    step: str,
    previous_steps: List[str],
    max_similarity_threshold: float,
) -> Dict[str, Any]:
    if not previous_steps:
        return {
            "max_similarity_to_previous": 0.0,
            "max_char_similarity_to_previous": 0.0,
            "max_jaccard_similarity_to_previous": 0.0,
            "closest_previous_step": None,
            "novelty": 1.0,
            "is_fixed_point_candidate": False,
        }

    best_step = None
    best_sem = -1.0
    best_char = 0.0
    best_jaccard = 0.0
    for prev in previous_steps:
        char_sim = sequence_similarity(step, prev)
        jaccard_sim = content_jaccard_similarity(step, prev)
        sem = max(char_sim, jaccard_sim)
        if sem > best_sem:
            best_sem = sem
            best_char = char_sim
            best_jaccard = jaccard_sim
            best_step = prev
    novelty = 1.0 - max(0.0, best_sem)
    return {
        "max_similarity_to_previous": max(0.0, best_sem),
        "max_char_similarity_to_previous": best_char,
        "max_jaccard_similarity_to_previous": best_jaccard,
        "closest_previous_step": best_step,
        "novelty": novelty,
        "is_fixed_point_candidate": best_sem >= max_similarity_threshold,
    }

def summarize_rewind_novelty(rows: List[Dict[str, Any]], *, threshold: float) -> Dict[str, Any]:
    if not rows:
        return {
            "novelty_curve": [],
            "fixed_point_depth": None,
            "fixed_point_recurrence_rate": 0.0,
            "pre_fixed_novelty_mean": None,
            "novelty_mean": None,
        }
    fixed_point_depth = None
    for row in rows:
        if row.get("is_fixed_point_candidate"):
            fixed_point_depth = int(row.get("depth", 0)) or None
            break
    novelty_values = [float(row.get("novelty", 0.0) or 0.0) for row in rows]
    pre_fixed_rows = rows if fixed_point_depth is None else [row for row in rows if int(row.get("depth", 0) or 0) < fixed_point_depth]
    tail_rows = [] if fixed_point_depth is None else [row for row in rows if int(row.get("depth", 0) or 0) >= fixed_point_depth]
    return {
        "novelty_curve": rows,
        "fixed_point_depth": fixed_point_depth,
        "fixed_point_recurrence_rate": (
            sum(1 for row in tail_rows if row.get("is_fixed_point_candidate")) / max(1, len(tail_rows))
            if tail_rows else 0.0
        ),
        "pre_fixed_novelty_mean": mean_or_none([float(row.get("novelty", 0.0) or 0.0) for row in pre_fixed_rows]),
        "novelty_mean": mean_or_none(novelty_values),
        "novelty_threshold": threshold,
    }

def compute_rewind_core_metrics(
    rewind_trace: Dict[str, Any],
    rewind_curve: Dict[str, Any],
) -> Dict[str, Any]:
    fixed_point_depth = rewind_trace.get("fixed_point_depth")
    recurrence = float(rewind_trace.get("fixed_point_recurrence_rate", 0.0) or 0.0)
    non_redundancy = float(rewind_trace.get("pre_fixed_novelty_mean", 0.0) or 0.0)
    loop_closure = float(
        rewind_trace.get("loop_closure_tail_mean")
        if rewind_trace.get("loop_closure_tail_mean") is not None
        else (rewind_trace.get("loop_closure_mean") or 0.0)
    )
    curve_rows = (rewind_curve or {}).get("curve") or []
    start_idx = int(fixed_point_depth or 1)
    tail_answers = [normalize_answer(str(row.get("top_answer", ""))) for row in curve_rows[start_idx:] if str(row.get("top_answer", "")).strip()]
    answer_reproduction = 0.0
    if tail_answers:
        _, count = Counter(tail_answers).most_common(1)[0]
        answer_reproduction = count / max(1, len(tail_answers))
    core_strength = loop_closure * answer_reproduction * recurrence * non_redundancy
    return {
        "fixed_point_depth": fixed_point_depth,
        "answer_reproduction_rate": answer_reproduction,
        "fixed_point_recurrence_rate": recurrence,
        "pre_fixed_novelty_mean": non_redundancy,
        "core_strength": core_strength,
    }

def summarize_rewind_alignment(
    original_latest_to_earlier: List[str],
    recovered_latest_to_earlier: List[str],
) -> Dict[str, Any]:
    alignment: List[Dict[str, Any]] = []
    sims: List[float] = []
    for j, (orig, rec) in enumerate(zip(original_latest_to_earlier, recovered_latest_to_earlier)):
        sim = step_semantic_similarity(orig, rec)
        sims.append(sim)
        alignment.append({
            "index_from_end": j + 1,
            "index_from_start": len(original_latest_to_earlier) - j,
            "original_step": orig,
            "recovered_step": rec,
            "similarity": sim,
        })

    tail_n = max(1, int(math.ceil(len(sims) * 0.25))) if sims else 0
    return {
        "alignment": alignment,
        "loop_closure_mean": mean_or_none(sims),
        "loop_closure_tail_mean": mean_or_none(sims[:tail_n]) if sims else None,
    }

def summarize_rewind_sequence_metrics(
    question: str,
    answer: str,
    steps_latest_to_earlier: List[str],
    novelty_threshold: float,
) -> Dict[str, Any]:
    prior: List[str] = []
    recovered_records: List[Dict[str, Any]] = []
    novelty_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    for depth, raw_step in enumerate(steps_latest_to_earlier, start=1):
        step = str(raw_step).strip()
        novelty_info = rewind_step_novelty_info(step, prior, novelty_threshold)
        reward = score_rewind_candidate_lite(question, answer, step, prior)
        recovered_records.append({
            "depth": depth,
            "step": step,
            "attempts_used": 1,
            "selected_attempt": 1,
            "selected_candidate_index": 0,
            "novelty_annotation": "",
            "why_earlier": "",
            "process_reward_score": reward.get("score"),
            "process_reward_atomicity": reward.get("atomicity"),
            "process_reward_relevance": reward.get("relevance"),
        })
        novelty_rows.append({
            "depth": depth,
            "step": step,
            "max_similarity_to_previous": float(novelty_info.get("max_similarity_to_previous", 0.0) or 0.0),
            "max_char_similarity_to_previous": float(novelty_info.get("max_char_similarity_to_previous", 0.0) or 0.0),
            "max_jaccard_similarity_to_previous": float(novelty_info.get("max_jaccard_similarity_to_previous", 0.0) or 0.0),
            "closest_previous_step": novelty_info.get("closest_previous_step"),
            "novelty": float(novelty_info.get("novelty", 0.0) or 0.0),
            "is_fixed_point_candidate": bool(novelty_info.get("is_fixed_point_candidate", False)),
            "selected_attempt": 1,
        })
        score_rows.append({
            "depth": depth,
            "step": step,
            "score": float(reward.get("score", 0.0) or 0.0),
            "atomicity": float(reward.get("atomicity", 0.0) or 0.0),
            "relevance": float(reward.get("relevance", 0.0) or 0.0),
            "answer_leak_penalty": float(reward.get("answer_leak_penalty", 0.0) or 0.0),
            "summary_penalty": float(reward.get("summary_penalty", 0.0) or 0.0),
            "novelty": float(reward.get("novelty", 0.0) or 0.0),
        })
        prior.append(step)
    return {
        "step_records": list(reversed(recovered_records)),
        "novelty_curve": novelty_rows,
        "process_reward_step_scores": score_rows,
        "process_reward_mean": mean_or_none([float(row.get("score", 0.0) or 0.0) for row in score_rows]),
        **summarize_rewind_novelty(novelty_rows, threshold=novelty_threshold),
    }

def trace_step_relevance(question: str, text: str) -> float:
    return clamp01(max(
        content_jaccard_similarity(question, text),
        0.5 * sequence_similarity(question, text),
    ))

def trace_step_atomicity_score(text: str) -> float:
    lower = f" {normalize_answer(text)} "
    words = CONTENT_WORD_RE.findall(lower)
    score = 1.0
    if len(words) < 4:
        score -= 0.15
    if len(words) > 28:
        score -= min(0.45, 0.02 * float(len(words) - 28))
    clause_markers = sum(lower.count(tok) for tok in [
        " and ", " but ", " because ", " therefore ", " however ", " while ",
        " which ", " that ", " then ", " so ",
    ])
    punctuation_markers = text.count(";") + text.count(":") + max(0, text.count(",") - 1)
    score -= min(0.5, 0.12 * max(0, clause_markers - 1) + 0.05 * punctuation_markers)
    return clamp01(score)

def trace_step_summary_penalty(text: str, step_type: str, step_index: int, total_steps: int) -> float:
    lower = normalize_answer(text)
    penalty = 0.0
    if any(marker in lower for marker in [
        "in summary", "overall", "we can conclude", "this means",
        "the answer", "therefore", "thus", "so the difference",
    ]):
        penalty += 0.35
    if step_index < total_steps - 1 and step_type in {"summary", "comparison", "conclusion"}:
        penalty += 0.25
    return clamp01(penalty)

def trace_step_answer_leak_penalty(text: str, answer: str) -> float:
    answer_norm = normalize_answer(answer)
    if not answer_norm:
        return 0.0
    step_norm = normalize_answer(text)
    penalty = clamp01((step_semantic_similarity(text, answer) - 0.62) / 0.30)
    if answer_norm and answer_norm in step_norm:
        penalty = max(penalty, 0.95)
    return penalty

def score_trace_step_record(
    question: str,
    answer: str,
    record: Dict[str, Any],
    previous_steps: List[str],
    *,
    step_index: int,
    total_steps: int,
) -> Dict[str, Any]:
    text = str(record.get("text") or "").strip()
    step_type = str(record.get("type") or "step").strip() or "step"
    novelty_info = rewind_step_novelty_info(text, previous_steps, 0.82)
    atomicity = trace_step_atomicity_score(text)
    relevance = trace_step_relevance(question, text)
    answer_leak_penalty = trace_step_answer_leak_penalty(text, answer)
    if step_index == total_steps - 1:
        answer_leak_penalty *= 0.35
    summary_penalty = trace_step_summary_penalty(text, step_type, step_index, total_steps)
    type_bonus = 0.0
    if step_type in {"premise", "facet", "assumption", "constraint", "observation", "deduction", "prerequisite"}:
        type_bonus += 0.05
    if step_type in {"summary", "conclusion"} and step_index < total_steps - 1:
        type_bonus -= 0.05
    score = clamp01(
        0.42 * atomicity
        + 0.28 * float(novelty_info.get("novelty", 0.0) or 0.0)
        + 0.18 * relevance
        + type_bonus
        - 0.20 * answer_leak_penalty
        - 0.18 * summary_penalty
    )
    return {
        "id": record.get("id"),
        "type": step_type,
        "text": text,
        "atomicity": atomicity,
        "relevance": relevance,
        "novelty": float(novelty_info.get("novelty", 0.0) or 0.0),
        "max_similarity_to_previous": float(novelty_info.get("max_similarity_to_previous", 0.0) or 0.0),
        "answer_leak_penalty": answer_leak_penalty,
        "summary_penalty": summary_penalty,
        "score": score,
    }

def score_trace_candidate_lite(question: str, trace: "Trace") -> Dict[str, Any]:
    records = normalize_trace_step_records(trace.step_records if trace.step_records is not None else trace.steps)
    previous_steps: List[str] = []
    step_scores: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        row = score_trace_step_record(
            question,
            trace.answer,
            record,
            previous_steps,
            step_index=idx,
            total_steps=len(records),
        )
        step_scores.append(row)
        previous_steps.append(row["text"])

    mean_atomicity = mean_or_none([float(row["atomicity"]) for row in step_scores]) or 0.0
    mean_relevance = mean_or_none([float(row["relevance"]) for row in step_scores]) or 0.0
    mean_novelty = mean_or_none([float(row["novelty"]) for row in step_scores]) or 0.0
    mean_answer_leak = mean_or_none([float(row["answer_leak_penalty"]) for row in step_scores]) or 0.0
    mean_summary_penalty = mean_or_none([float(row["summary_penalty"]) for row in step_scores]) or 0.0
    mean_step_score = mean_or_none([float(row["score"]) for row in step_scores]) or 0.0
    step_count_score = clamp01((len(step_scores) - 1) / 4.0)
    if len(step_scores) > 8:
        step_count_score = clamp01(step_count_score - 0.03 * float(len(step_scores) - 8))
    answer_present = 1.0 if normalize_answer(trace.answer) else 0.0
    answer_relevance = trace_step_relevance(question, trace.answer) if trace.answer else 0.0
    redundancy_penalty = mean_or_none([
        float(row.get("max_similarity_to_previous", 0.0) or 0.0)
        for row in step_scores[1:]
    ]) or 0.0
    overall_score = clamp01(
        0.30 * mean_atomicity
        + 0.22 * mean_novelty
        + 0.18 * mean_relevance
        + 0.15 * step_count_score
        + 0.05 * answer_present
        + 0.05 * answer_relevance
        - 0.20 * mean_answer_leak
        - 0.15 * mean_summary_penalty
        - 0.10 * redundancy_penalty
    )
    return {
        "mode": "lite",
        "overall_score": overall_score,
        "mean_step_score": mean_step_score,
        "mean_atomicity": mean_atomicity,
        "mean_relevance": mean_relevance,
        "mean_novelty": mean_novelty,
        "mean_answer_leak_penalty": mean_answer_leak,
        "mean_summary_penalty": mean_summary_penalty,
        "redundancy_penalty": redundancy_penalty,
        "step_count_score": step_count_score,
        "answer_present_score": answer_present,
        "answer_relevance": answer_relevance,
        "step_scores": step_scores,
    }

def score_rewind_candidate_lite(
    question: str,
    answer: str,
    step: str,
    previous_steps: List[str],
    *,
    step_type: str = "rewind_step",
    novelty_annotation: str = "",
    why_earlier: str = "",
) -> Dict[str, Any]:
    novelty_info = rewind_step_novelty_info(step, previous_steps, 0.82)
    atomicity = trace_step_atomicity_score(step)
    relevance = trace_step_relevance(question, step)
    answer_leak_penalty = trace_step_answer_leak_penalty(step, answer)
    summary_penalty = trace_step_summary_penalty(step, step_type, 0, 2)
    explanation_bonus = 0.05 if str(why_earlier or "").strip() else 0.0
    novelty_bonus = 0.03 if str(novelty_annotation or "").strip() else 0.0
    score = clamp01(
        0.36 * float(novelty_info.get("novelty", 0.0) or 0.0)
        + 0.26 * atomicity
        + 0.18 * relevance
        + explanation_bonus
        + novelty_bonus
        - 0.18 * answer_leak_penalty
        - 0.14 * summary_penalty
    )
    return {
        **novelty_info,
        "atomicity": atomicity,
        "relevance": relevance,
        "answer_leak_penalty": answer_leak_penalty,
        "summary_penalty": summary_penalty,
        "explanation_bonus": explanation_bonus,
        "novelty_annotation_bonus": novelty_bonus,
        "score": score,
    }

def compare_step_sequences(
    left_steps: List[str],
    right_steps: List[str],
    *,
    left_label: str,
    right_label: str,
) -> Dict[str, Any]:
    left = [str(s).strip() for s in left_steps if str(s).strip()]
    right = [str(s).strip() for s in right_steps if str(s).strip()]
    if not left and not right:
        return {
            "left_label": left_label,
            "right_label": right_label,
            "left_len": 0,
            "right_len": 0,
            "similarity_mean": None,
            "similarity_concat": None,
            "preservation_rate": None,
            "exact_match_count": 0,
            "paraphrase_count": 0,
            "changed_count": 0,
            "missing_count": 0,
            "extra_count": 0,
            "pairwise_alignment": [],
        }

    pairwise: List[Dict[str, Any]] = []
    sims_both: List[float] = []
    counts = Counter()
    n = max(len(left), len(right))
    for i in range(n):
        left_step = left[i] if i < len(left) else ""
        right_step = right[i] if i < len(right) else ""
        if left_step and right_step:
            sim = step_semantic_similarity(left_step, right_step)
            sims_both.append(sim)
            if normalize_answer(left_step) == normalize_answer(right_step):
                status = "exact"
            elif sim >= 0.75:
                status = "paraphrase"
            else:
                status = "changed"
        elif left_step:
            sim = 0.0
            status = f"missing_in_{right_label}"
        elif right_step:
            sim = 0.0
            status = f"extra_in_{right_label}"
        else:
            sim = 0.0
            status = "empty"
        counts[status] += 1
        pairwise.append({
            "index": i,
            "left_step": left_step,
            "right_step": right_step,
            f"{left_label}_step": left_step,
            f"{right_label}_step": right_step,
            "similarity": sim,
            "status": status,
        })

    preservation_rate = (counts["exact"] + counts["paraphrase"]) / max(1, len(left))
    return {
        "left_label": left_label,
        "right_label": right_label,
        "left_len": len(left),
        "right_len": len(right),
        "similarity_mean": mean_or_none(sims_both),
        "similarity_concat": sequence_similarity("\n".join(left), "\n".join(right)),
        "preservation_rate": preservation_rate,
        "exact_match_count": counts["exact"],
        "paraphrase_count": counts["paraphrase"],
        "changed_count": counts["changed"],
        "missing_count": counts[f"missing_in_{right_label}"],
        "extra_count": counts[f"extra_in_{right_label}"],
        "pairwise_alignment": pairwise,
    }

def compare_forward_and_rewind_steps(original_steps: List[str], rewind_trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(rewind_trace, dict):
        return None
    recovered_forward = rewind_trace.get("forward_order") or list(reversed(rewind_trace.get("latest_to_earlier") or []))
    recovered_forward = [str(s).strip() for s in recovered_forward if str(s).strip()]
    original = [str(s).strip() for s in original_steps if str(s).strip()]
    if not recovered_forward:
        return None
    if not original and not recovered_forward:
        return None
    comp = compare_step_sequences(original, recovered_forward, left_label="original", right_label="rewind")
    pairwise = []
    for row in comp.get("pairwise_alignment", []) or []:
        item = dict(row)
        item["original_step"] = item.get("left_step", "")
        item["rewind_step"] = item.get("right_step", "")
        pairwise.append(item)
    return {
        "original_len": comp.get("left_len"),
        "rewind_len": comp.get("right_len"),
        "similarity_mean": comp.get("similarity_mean"),
        "similarity_concat": comp.get("similarity_concat"),
        "preservation_rate": comp.get("preservation_rate"),
        "exact_match_count": comp.get("exact_match_count"),
        "paraphrase_count": comp.get("paraphrase_count"),
        "changed_count": comp.get("changed_count"),
        "missing_count": comp.get("missing_count"),
        "extra_count": comp.get("extra_count"),
        "pairwise_alignment": pairwise,
    }

def build_three_axis_trace_comparison(
    base_trace: "Trace",
    prm_trace: "Trace",
    rewind_trace: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(rewind_trace, dict):
        return None
    rewind_forward = rewind_trace.get("forward_order") or list(reversed(rewind_trace.get("latest_to_earlier") or []))
    rewind_forward = [str(s).strip() for s in rewind_forward if str(s).strip()]
    if not rewind_forward:
        return None

    comp_base_prm = compare_step_sequences(base_trace.steps, prm_trace.steps, left_label="base", right_label="prm")
    comp_base_rewind = compare_step_sequences(base_trace.steps, rewind_forward, left_label="base", right_label="rewind")
    comp_prm_rewind = compare_step_sequences(prm_trace.steps, rewind_forward, left_label="prm", right_label="rewind")

    bp_rows = {int(row.get("index", -1)): row for row in comp_base_prm.get("pairwise_alignment", []) or []}
    br_rows = {int(row.get("index", -1)): row for row in comp_base_rewind.get("pairwise_alignment", []) or []}
    pr_rows = {int(row.get("index", -1)): row for row in comp_prm_rewind.get("pairwise_alignment", []) or []}

    n = max(len(base_trace.steps), len(prm_trace.steps), len(rewind_forward))
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        bp = bp_rows.get(i, {})
        br = br_rows.get(i, {})
        pr = pr_rows.get(i, {})
        rows.append({
            "index": i,
            "base_step": base_trace.steps[i] if i < len(base_trace.steps) else "",
            "prm_step": prm_trace.steps[i] if i < len(prm_trace.steps) else "",
            "rewind_step": rewind_forward[i] if i < len(rewind_forward) else "",
            "base_prm_similarity": bp.get("similarity"),
            "base_prm_status": bp.get("status"),
            "base_rewind_similarity": br.get("similarity"),
            "base_rewind_status": br.get("status"),
            "prm_rewind_similarity": pr.get("similarity"),
            "prm_rewind_status": pr.get("status"),
        })

    return {
        "base_len": len(base_trace.steps),
        "prm_len": len(prm_trace.steps),
        "rewind_len": len(rewind_forward),
        "base_prm": comp_base_prm,
        "base_rewind": comp_base_rewind,
        "prm_rewind": comp_prm_rewind,
        "rows": rows,
    }

def build_rewind_axis_comparison(rewind_trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(rewind_trace, dict):
        return None
    base_rewind_obj = rewind_trace.get("base_rewind") or {}
    if not isinstance(base_rewind_obj, dict):
        return None
    base_forward = base_rewind_obj.get("forward_order") or list(reversed(base_rewind_obj.get("latest_to_earlier") or []))
    prm_forward = rewind_trace.get("forward_order") or list(reversed(rewind_trace.get("latest_to_earlier") or []))
    base_forward = [str(s).strip() for s in base_forward if str(s).strip()]
    prm_forward = [str(s).strip() for s in prm_forward if str(s).strip()]
    if not base_forward or not prm_forward:
        return None
    comp = compare_step_sequences(base_forward, prm_forward, left_label="rewind_base", right_label="rewind_prm")
    rows = []
    for row in comp.get("pairwise_alignment", []) or []:
        rows.append({
            "index": row.get("index"),
            "rewind_base_step": row.get("left_step"),
            "rewind_prm_step": row.get("right_step"),
            "rewind_base_prm_similarity": row.get("similarity"),
            "rewind_base_prm_status": row.get("status"),
        })
    return {
        "rewind_base_len": len(base_forward),
        "rewind_prm_len": len(prm_forward),
        "rewind_base_prm": comp,
        "rows": rows,
    }

def trace_vs_rewind_jsonl_rows(comp: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [{
        "record_type": "summary",
        "original_len": comp.get("original_len"),
        "rewind_len": comp.get("rewind_len"),
        "similarity_mean": comp.get("similarity_mean"),
        "similarity_concat": comp.get("similarity_concat"),
        "preservation_rate": comp.get("preservation_rate"),
        "exact_match_count": comp.get("exact_match_count"),
        "paraphrase_count": comp.get("paraphrase_count"),
        "changed_count": comp.get("changed_count"),
        "missing_count": comp.get("missing_count"),
        "extra_count": comp.get("extra_count"),
    }]
    for row in comp.get("pairwise_alignment", []) or []:
        item = dict(row)
        item["record_type"] = "step"
        rows.append(item)
    return rows

def trace_axes_jsonl_rows(comp: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [{
        "record_type": "summary",
        "base_len": comp.get("base_len"),
        "prm_len": comp.get("prm_len"),
        "rewind_len": comp.get("rewind_len"),
        "base_prm_similarity_mean": ((comp.get("base_prm") or {}).get("similarity_mean") if isinstance(comp.get("base_prm"), dict) else None),
        "base_rewind_similarity_mean": ((comp.get("base_rewind") or {}).get("similarity_mean") if isinstance(comp.get("base_rewind"), dict) else None),
        "prm_rewind_similarity_mean": ((comp.get("prm_rewind") or {}).get("similarity_mean") if isinstance(comp.get("prm_rewind"), dict) else None),
        "base_prm_preservation_rate": ((comp.get("base_prm") or {}).get("preservation_rate") if isinstance(comp.get("base_prm"), dict) else None),
        "base_rewind_preservation_rate": ((comp.get("base_rewind") or {}).get("preservation_rate") if isinstance(comp.get("base_rewind"), dict) else None),
        "prm_rewind_preservation_rate": ((comp.get("prm_rewind") or {}).get("preservation_rate") if isinstance(comp.get("prm_rewind"), dict) else None),
    }]
    for row in comp.get("rows", []) or []:
        item = dict(row)
        item["record_type"] = "step"
        rows.append(item)
    return rows

def rewind_axes_jsonl_rows(comp: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [{
        "record_type": "summary",
        "rewind_base_len": comp.get("rewind_base_len"),
        "rewind_prm_len": comp.get("rewind_prm_len"),
        "rewind_base_prm_similarity_mean": ((comp.get("rewind_base_prm") or {}).get("similarity_mean") if isinstance(comp.get("rewind_base_prm"), dict) else None),
        "rewind_base_prm_preservation_rate": ((comp.get("rewind_base_prm") or {}).get("preservation_rate") if isinstance(comp.get("rewind_base_prm"), dict) else None),
    }]
    for row in comp.get("rows", []) or []:
        item = dict(row)
        item["record_type"] = "step"
        rows.append(item)
    return rows




# ----------------------------- backends -----------------------------

class Backend(Protocol):
    name: str
    def generate(self, prompt: str, temperature: float, seed: Optional[int] = None) -> str: ...


@dataclass
class HTTPBackend:
    name: str
    base_url: str
    api_key: str
    model: str
    timeout_s: int = 120
    extra_headers: Optional[Dict[str, str]] = None
    # If True, will include "seed" param; if endpoint rejects it, we auto-disable.
    try_seed: bool = True

    _seed_supported: Optional[bool] = None

    def _normalize_base(self) -> str:
        b = self.base_url.rstrip("/")
        if b.endswith("/v1"):
            return b
        # allow passing ".../v1" or "..."
        if b.endswith("/v1/"):
            return b[:-1]
        # if it's likely already has v1? keep simple
        return b + "/v1"

    def generate(self, prompt: str, temperature: float, seed: Optional[int] = None) -> str:
        url = self._normalize_base() + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.extra_headers:
            headers.update(self.extra_headers)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
        }

        # best-effort seed
        if seed is not None:
            if self._seed_supported is False:
                pass
            else:
                if self.try_seed:
                    payload["seed"] = int(seed)

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        if resp.status_code >= 400:
            # if seed rejected, retry once without seed
            if seed is not None and ("seed" in payload) and self.try_seed:
                try:
                    j = resp.json()
                    msg = json.dumps(j, ensure_ascii=False)
                except Exception:
                    msg = resp.text
                if ("seed" in msg.lower()) or ("unrecognized" in msg.lower()) or ("unknown" in msg.lower()):
                    self._seed_supported = False
                    payload.pop("seed", None)
                    resp2 = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                    resp2.raise_for_status()
                    out = resp2.json()
                    return out["choices"][0]["message"]["content"]
            resp.raise_for_status()

        out = resp.json()
        return out["choices"][0]["message"]["content"]


# ----------------------------- HF local backend -----------------------------

@dataclass
class HFLocalBackend:
    name: str
    model_path: str
    tokenizer_path: Optional[str] = None
    device: str = "cpu"          # "cpu", "cuda", "mps"
    hf_backend: str = "auto"     # auto | torch | mlx
    torch_dtype: str = "auto"    # "auto", "float16", "bfloat16", "float32"
    device_map: Optional[str] = None  # e.g. "auto"
    trust_remote_code: bool = False
    use_fast_tokenizer: Optional[bool] = None  # None -> transformers default
    max_new_tokens: int = 256
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0

    _loaded: bool = False
    _tokenizer: Any = None
    _model: Any = None
    _runtime_backend: str = "unloaded"

    def _normalize_model_hint(self, s: Optional[str]) -> str:
        return "".join(ch.lower() for ch in str(s or "") if ch.isalnum())

    def _read_model_dir_info(self, path: str) -> Optional[Dict[str, Any]]:
        cfg_path = os.path.join(path, "config.json")
        if not os.path.exists(cfg_path):
            return None
        try:
            cfg = read_json(cfg_path)
        except Exception:
            cfg = {}

        archs_raw = cfg.get("architectures") or []
        if isinstance(archs_raw, str):
            archs = [archs_raw]
        elif isinstance(archs_raw, list):
            archs = [str(x) for x in archs_raw]
        else:
            archs = []

        model_type = str(cfg.get("model_type") or "").strip()
        basename = os.path.basename(os.path.abspath(path))
        has_weights = any([
            os.path.exists(os.path.join(path, "model.safetensors")),
            os.path.exists(os.path.join(path, "model.safetensors.index.json")),
            os.path.exists(os.path.join(path, "pytorch_model.bin")),
            os.path.exists(os.path.join(path, "pytorch_model.bin.index.json")),
            any(pathlib.Path(path).glob("model-*.safetensors")),
            any(pathlib.Path(path).glob("pytorch_model-*.bin")),
        ])
        has_tokenizer_assets = any(
            os.path.exists(os.path.join(path, fn))
            for fn in [
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "spiece.model",
            ]
        )
        is_generation_model = any(
            token in arch.lower()
            for arch in archs
            for token in ["causallm", "conditionalgeneration", "lmheadmodel"]
        )
        return {
            "path": path,
            "basename": basename,
            "config": cfg,
            "architectures": archs,
            "model_type": model_type,
            "has_model_type": bool(model_type),
            "has_weights": has_weights,
            "has_tokenizer_assets": has_tokenizer_assets,
            "is_generation_model": is_generation_model,
        }

    def _find_model_dir_candidates(self, root: str, maxdepth: int = 3) -> List[Dict[str, Any]]:
        rootp = pathlib.Path(root)
        candidates: List[Dict[str, Any]] = []
        if not rootp.exists():
            return candidates

        q: List[Tuple[pathlib.Path, int]] = [(rootp, 0)]
        seen = set()
        while q:
            p, d = q.pop(0)
            p_str = str(p)
            if p_str in seen:
                continue
            seen.add(p_str)

            if p.is_dir():
                info = self._read_model_dir_info(p_str)
                if info and info["has_weights"]:
                    candidates.append(info)

            if d >= maxdepth:
                continue
            try:
                for child in p.iterdir():
                    if child.is_dir():
                        q.append((child, d + 1))
            except Exception:
                continue

        generation_candidates = [c for c in candidates if c["is_generation_model"]]
        if generation_candidates:
            candidates = generation_candidates
        candidates.sort(key=lambda c: (len(c["path"]), c["basename"]))
        return candidates

    def _score_model_dir_candidate(self, info: Dict[str, Any], hints: List[str]) -> float:
        score = 0.0
        if info.get("is_generation_model"):
            score += 0.35
        if info.get("has_tokenizer_assets"):
            score += 0.10

        targets = [
            info.get("basename"),
            info.get("model_type"),
            info.get("config", {}).get("_name_or_path"),
        ]
        target_norms = [self._normalize_model_hint(x) for x in targets if self._normalize_model_hint(x)]

        name_score = 0.0
        for hint in hints:
            hint_norm = self._normalize_model_hint(hint)
            if not hint_norm:
                continue
            for target_norm in target_norms:
                ratio = SequenceMatcher(None, hint_norm, target_norm).ratio()
                if hint_norm == target_norm:
                    ratio += 0.35
                elif hint_norm in target_norm or target_norm in hint_norm:
                    ratio += 0.15
                name_score = max(name_score, ratio)
        score += min(name_score, 1.0)
        return score

    def _candidate_desc(self, info: Dict[str, Any]) -> str:
        archs = info.get("architectures") or []
        arch = archs[0] if archs else (info.get("model_type") or "unknown")
        return f"{info['path']} ({arch})"

    def _resolve_model_path(self) -> str:
        raw_path = self.model_path
        if not raw_path or is_probably_url(raw_path) or not os.path.exists(raw_path):
            return raw_path
        if not os.path.isdir(raw_path):
            return raw_path

        direct_info = self._read_model_dir_info(raw_path)
        if direct_info and direct_info["has_model_type"] and direct_info["has_weights"]:
            return raw_path

        candidates = [c for c in self._find_model_dir_candidates(raw_path) if os.path.abspath(c["path"]) != os.path.abspath(raw_path)]
        if not candidates:
            return raw_path
        if len(candidates) == 1:
            resolved = candidates[0]["path"]
            print(f"[hf:{self.name}] resolved model directory '{raw_path}' -> '{resolved}'")
            return resolved

        hint_values: List[str] = [self.name]
        tok_base = os.path.basename(str(self.tokenizer_path or "").rstrip("/"))
        if tok_base and tok_base.lower() not in {"original", "tokenizer", "model"}:
            hint_values.append(tok_base)

        ranked = sorted(
            ((self._score_model_dir_candidate(info, hint_values), info) for info in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        if ranked:
            best_score, best = ranked[0]
            next_score = ranked[1][0] if len(ranked) > 1 else -1.0
            if best_score >= 0.80 and (len(ranked) == 1 or best_score - next_score >= 0.15):
                resolved = best["path"]
                print(
                    f"[hf:{self.name}] resolved model directory '{raw_path}' -> '{resolved}' "
                    f"(score={best_score:.2f})"
                )
                return resolved

        candidate_lines = "\n".join(f"  - {self._candidate_desc(info)}" for info in candidates[:12])
        raise RuntimeError(
            "Model path points to a container directory, but the target model is ambiguous.\n"
            f"- requested model path: {raw_path}\n"
            f"- backend name hint: {self.name}\n"
            "Candidates:\n"
            f"{candidate_lines}\n"
            "Pass --model with the exact model subdirectory."
        )

    def _resolve_tokenizer_path(self) -> Optional[str]:
        raw_path = self.tokenizer_path
        if not raw_path:
            return None
        if is_probably_url(raw_path):
            return raw_path
        if os.path.exists(raw_path):
            return raw_path
        print(f"[hf:{self.name}] tokenizer path '{raw_path}' not found; falling back to model tokenizer assets")
        return None

    def _dtype_obj(self) -> Any:
        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError("torch is required for hf_local backend. Install with: pip install torch") from e
        if self.torch_dtype in ("auto", "", None):
            return "auto"
        m = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return m.get(self.torch_dtype.lower(), "auto")

    def _maybe_fix_mps_dtype(self) -> None:
        # MPS can be picky; prefer float16 if "auto" would pick bf16 unexpectedly.
        if self.device == "mps":
            if self.torch_dtype in ("bfloat16", "bf16"):
                # convert to float16 silently (better than hard-fail)
                self.torch_dtype = "float16"

    @property
    def runtime_backend(self) -> str:
        return self._runtime_backend if self._runtime_backend != "unloaded" else normalize_hf_backend_choice(self.hf_backend)

    def _should_try_mlx(self) -> bool:
        choice = normalize_hf_backend_choice(self.hf_backend)
        if choice == "mlx":
            return True
        if choice == "auto" and self.device == "mps":
            return True
        return False

    def _find_tokenizer_candidates(self, root: str, maxdepth: int = 3) -> List[str]:
        rootp = pathlib.Path(root)
        candidates: List[str] = []
        if not rootp.exists():
            return candidates

        # score directories by presence of tokenizer assets
        def score_dir(p: pathlib.Path) -> int:
            score = 0
            for fn in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                       "vocab.json", "merges.txt", "spiece.model"]:
                if (p / fn).exists():
                    score += 1
            return score

        # BFS up to maxdepth
        q: List[Tuple[pathlib.Path, int]] = [(rootp, 0)]
        seen = set()
        while q:
            p, d = q.pop(0)
            if str(p) in seen:
                continue
            seen.add(str(p))
            s = score_dir(p)
            if s >= 2:
                candidates.append(str(p))
            if d >= maxdepth:
                continue
            try:
                for child in p.iterdir():
                    if child.is_dir():
                        q.append((child, d + 1))
            except Exception:
                continue
        # sort by score desc then shorter path
        candidates.sort(key=lambda x: (-score_dir(pathlib.Path(x)), len(x)))
        return candidates

    def _overlay_for_tokenizer(self, tokenizer_dir: str, model_dir: str) -> str:
        """
        Create a temp directory that contains:
        - config.json (and similar) from model_dir
        - tokenizer files from tokenizer_dir
        Using symlinks when possible.
        """
        tmp = tempfile.mkdtemp(prefix="hf_tok_overlay_")
        def link_or_copy(src: str, dst: str) -> None:
            try:
                os.symlink(src, dst)
            except Exception:
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass

        # bring config.json if exists
        for fn in ["config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json"]:
            src = os.path.join(model_dir, fn)
            if os.path.exists(src):
                link_or_copy(src, os.path.join(tmp, fn))

        # bring tokenizer assets from tokenizer_dir
        for fn in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt", "spiece.model"]:
            src = os.path.join(tokenizer_dir, fn)
            if os.path.exists(src):
                link_or_copy(src, os.path.join(tmp, fn))

        return tmp

    def _load_tokenizer_torch(self) -> Any:
        from transformers import AutoTokenizer  # type: ignore

        tok_kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.use_fast_tokenizer is not None:
            tok_kwargs["use_fast"] = bool(self.use_fast_tokenizer)

        def try_load_tokenizer(path: str, use_fast_override: Optional[bool] = None) -> Any:
            kw = dict(tok_kwargs)
            if use_fast_override is not None:
                kw["use_fast"] = use_fast_override
            return AutoTokenizer.from_pretrained(path, **kw)

        tok_path = self.tokenizer_path or self.model_path
        tokenizer = None
        last_err: Optional[Exception] = None

        try:
            tokenizer = try_load_tokenizer(tok_path, None)
        except Exception as e1:
            last_err = e1
            try:
                tokenizer = try_load_tokenizer(tok_path, False)
                last_err = None
            except Exception as e2:
                last_err = e2

        if tokenizer is None:
            candidates = self._find_tokenizer_candidates(self.model_path)
            candidates = sorted(candidates, key=lambda p: (0 if p.endswith(os.sep + "original") or p.endswith("/original") else 1, len(p)))
            for cand in candidates[:5]:
                try:
                    overlay = self._overlay_for_tokenizer(cand, self.model_path)
                    try:
                        tokenizer = try_load_tokenizer(overlay, None)
                    except Exception:
                        tokenizer = try_load_tokenizer(overlay, False)
                    if tokenizer is not None:
                        break
                except Exception as e3:
                    last_err = e3
                    continue

        if tokenizer is None:
            raise RuntimeError(
                "Failed to load tokenizer.\n"
                f"- model_path: {self.model_path}\n"
                f"- tokenizer_path: {self.tokenizer_path}\n"
                "Common fixes:\n"
                "  - Ensure the folder contains tokenizer files (tokenizer.json or tokenizer.model / spiece.model).\n"
                "  - If tokenizer files live in a subfolder (e.g., ./model/original), pass --tokenizer that folder.\n"
                "  - Try forcing slow tokenizer: --use-fast-tokenizer false\n"
                f"Original error: {last_err}"
            )

        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            try:
                tokenizer.pad_token = tokenizer.eos_token  # type: ignore
            except Exception:
                pass
        return tokenizer

    def _resolve_mlx_load_path(self) -> str:
        tok_path = self.tokenizer_path or self.model_path
        if tok_path and os.path.abspath(tok_path) != os.path.abspath(self.model_path):
            return self._overlay_for_tokenizer(tok_path, self.model_path)
        return self.model_path

    def _load_torch(self) -> None:
        try:
            from transformers import AutoModelForCausalLM  # type: ignore
        except Exception as e:
            raise RuntimeError("HF backend requires: pip install transformers torch") from e

        dtype_obj = self._dtype_obj()
        model_kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        if dtype_obj != "auto":
            model_kwargs["torch_dtype"] = dtype_obj

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [hf:{self.name}] loading torch model from '{self.model_path}'"
            f" (device={self.device}, device_map={self.device_map}, dtype={self.torch_dtype})"
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        model.eval()

        if not self.device_map:
            try:
                model.to(self.device)
            except Exception:
                pass

        self._tokenizer = self._load_tokenizer_torch()
        self._model = model
        self._runtime_backend = "torch"
        self._loaded = True

    def _load_mlx(self) -> None:
        try:
            import mlx_lm  # type: ignore
        except Exception as e:
            raise RuntimeError("MLX backend requires mlx_lm. Install with: pip install mlx-lm") from e

        load_path = self._resolve_mlx_load_path()
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [hf:{self.name}] loading mlx model from '{load_path}'"
            f" (requested_device={self.device}, dtype={self.torch_dtype})"
        )
        model, tokenizer = mlx_lm.load(load_path)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            try:
                tokenizer.pad_token = tokenizer.eos_token  # type: ignore
            except Exception:
                pass
        self._model = model
        self._tokenizer = tokenizer
        self._runtime_backend = "mlx"
        self._loaded = True

    def _load(self) -> None:
        self._maybe_fix_mps_dtype()
        self.hf_backend = normalize_hf_backend_choice(self.hf_backend)
        self.model_path = self._resolve_model_path()
        self.tokenizer_path = self._resolve_tokenizer_path()

        if self._should_try_mlx():
            try:
                self._load_mlx()
                return
            except Exception as e:
                if self.hf_backend == "mlx":
                    raise
                print(f"[hf:{self.name}] mlx load failed, falling back to torch: {e}")

        self._load_torch()

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    def _format_prompt(self, user_prompt: str) -> Tuple[str, bool]:
        """
        Returns (text, used_chat_template)
        """
        self._ensure_loaded()
        tok = self._tokenizer
        # If chat template exists, use it
        if hasattr(tok, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": user_prompt}]
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                return text, True
            except Exception:
                pass
        # fallback
        text = f"User: {user_prompt}\nAssistant:"
        return text, False

    def _seed_torch(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError("torch is required for hf_local backend") from e

        try:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def _generate_torch(self, prompt: str, temperature: float, seed: Optional[int] = None) -> str:
        self._seed_torch(seed)
        text, _ = self._format_prompt(prompt)
        tok = self._tokenizer
        model = self._model

        inputs = tok(text, return_tensors="pt")
        # move to device if possible and not device_map
        if not self.device_map:
            try:
                for k in inputs:
                    inputs[k] = inputs[k].to(self.device)
            except Exception:
                pass

        do_sample = float(temperature) > 1e-8
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=int(self.max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            top_p=float(self.top_p),
            top_k=int(self.top_k),
            repetition_penalty=float(self.repetition_penalty),
            pad_token_id=getattr(tok, "pad_token_id", None),
            eos_token_id=getattr(tok, "eos_token_id", None),
        )
        # remove None to avoid warnings
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        # decode only generated part if possible
        try:
            gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
            return tok.decode(gen_tokens, skip_special_tokens=True).strip()
        except Exception:
            return tok.decode(out[0], skip_special_tokens=True).strip()

    def _stream_generate_torch(self, prompt: str, temperature: float, seed: Optional[int] = None) -> Iterator[str]:
        self._seed_torch(seed)
        try:
            import torch  # type: ignore
            from transformers import TextIteratorStreamer  # type: ignore
        except Exception as e:
            raise RuntimeError("torch streaming requires transformers + torch") from e

        text, _ = self._format_prompt(prompt)
        tok = self._tokenizer
        model = self._model
        inputs = tok(text, return_tensors="pt")
        if not self.device_map:
            try:
                for k in inputs:
                    inputs[k] = inputs[k].to(self.device)
            except Exception:
                pass

        do_sample = float(temperature) > 1e-8
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs: Dict[str, Any] = dict(
            **inputs,
            max_new_tokens=int(self.max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            top_p=float(self.top_p),
            top_k=int(self.top_k),
            repetition_penalty=float(self.repetition_penalty),
            pad_token_id=getattr(tok, "pad_token_id", None),
            eos_token_id=getattr(tok, "eos_token_id", None),
            streamer=streamer,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        error_holder: Dict[str, BaseException] = {}

        def _run_generate() -> None:
            try:
                model.generate(**gen_kwargs)
            except BaseException as e:
                error_holder["error"] = e
                try:
                    streamer.on_finalized_text("", stream_end=True)
                except Exception:
                    pass

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield str(chunk)
        finally:
            thread.join()
        if "error" in error_holder:
            raise error_holder["error"]

    def _extract_mlx_chunk_text(self, chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        for key in ("text", "token_text", "content", "last_segment"):
            value = getattr(chunk, key, None)
            if isinstance(value, str):
                return value
        return ""

    def _generate_mlx(self, prompt: str, temperature: float, seed: Optional[int] = None) -> str:
        try:
            import mlx_lm  # type: ignore
        except Exception as e:
            raise RuntimeError("MLX backend requires mlx_lm") from e

        text, _ = self._format_prompt(prompt)
        temp = float(temperature)
        try:
            return str(
                mlx_lm.generate(
                    self._model,
                    self._tokenizer,
                    prompt=text,
                    max_tokens=int(self.max_new_tokens),
                    temp=temp,
                    top_p=float(self.top_p),
                    repetition_penalty=float(self.repetition_penalty),
                    verbose=False,
                    seed=seed,
                )
            ).strip()
        except TypeError:
            return str(
                mlx_lm.generate(
                    self._model,
                    self._tokenizer,
                    prompt=text,
                    max_tokens=int(self.max_new_tokens),
                    temp=temp,
                    top_p=float(self.top_p),
                    repetition_penalty=float(self.repetition_penalty),
                    verbose=False,
                )
            ).strip()

    def _stream_generate_mlx(self, prompt: str, temperature: float, seed: Optional[int] = None) -> Iterator[str]:
        try:
            import mlx_lm  # type: ignore
            import mlx.core as mx  # type: ignore
            from mlx_lm.sample_utils import make_logits_processors, make_sampler  # type: ignore
        except Exception as e:
            raise RuntimeError("MLX backend requires mlx_lm") from e

        text, _ = self._format_prompt(prompt)
        if seed is not None:
            try:
                mx.random.seed(int(seed))
            except Exception:
                pass
        kwargs: Dict[str, Any] = {
            "prompt": text,
            "max_tokens": int(self.max_new_tokens),
            "sampler": make_sampler(
                temp=float(temperature),
                top_p=float(self.top_p),
                top_k=int(self.top_k),
            ),
            "logits_processors": make_logits_processors(
                repetition_penalty=(float(self.repetition_penalty) if float(self.repetition_penalty) != 1.0 else None),
            ),
        }
        iterator = mlx_lm.stream_generate(self._model, self._tokenizer, **kwargs)
        for chunk in iterator:
            text_chunk = self._extract_mlx_chunk_text(chunk)
            if text_chunk:
                yield text_chunk

    def generate(self, prompt: str, temperature: float, seed: Optional[int] = None) -> str:
        self._ensure_loaded()
        if self._runtime_backend == "mlx":
            return self._generate_mlx(prompt, temperature=temperature, seed=seed)
        return self._generate_torch(prompt, temperature=temperature, seed=seed)

    def stream_generate(self, prompt: str, temperature: float, seed: Optional[int] = None) -> Iterator[str]:
        self._ensure_loaded()
        if self._runtime_backend == "mlx":
            yield from self._stream_generate_mlx(prompt, temperature=temperature, seed=seed)
            return
        yield from self._stream_generate_torch(prompt, temperature=temperature, seed=seed)


# ----------------------------- experiment core -----------------------------

@dataclass
class Trace:
    steps: List[str]
    answer: str
    step_records: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CurvePoint:
    k: int
    match_rate: float
    entropy_bits: float
    top_answer: str
    js_div_full: Optional[float] = None
    kl_div_full: Optional[float] = None

@dataclass
class ExperimentSettings:
    trace_mode: str = "masked"              # masked|unmasked
    trace_source: str = "per_backend"       # per_backend or a backend name to share trace
    trace_format: str = "auto"              # auto|json|text
    stream_output: bool = False
    temperature_trace: float = 0.0
    process_reward_mode: str = "off"        # off|lite
    trace_candidates: int = 1
    trace_candidate_temperature: float = 0.4
    temperature_reanswer: float = 0.8
    n_samples_baseline: int = 9
    n_samples_per_k: int = 9
    tau: float = 0.8
    max_steps: int = 40
    prompt_version: str = "v1"
    prompt_family: str = "default"

    # research extras
    baseline_min_conf: Optional[float] = None     # e.g. 0.8
    baseline_max_entropy: Optional[float] = None  # e.g. 1.0 bits
    skip_unstable_baseline: bool = False

    smooth_window: int = 1
    converge_min_run: int = 1
    collapse_min_run: int = 1

    compute_divergence: bool = True
    divergence_alpha: float = 1e-6

    permutation_tests: int = 0  # 0 disables
    permutation_seed: int = 1234

    # rewind-CoT extras
    enable_rewind: bool = True
    rewind_step_temperature: float = 0.4
    rewind_samples_per_depth: int = 0      # 0 -> reuse n_samples_per_k
    rewind_order: str = "reverse"          # reverse|forward
    compute_oracle_tail: bool = True
    rewind_process_reward_mode: str = "off"  # off|lite
    rewind_step_candidates: int = 1
    rewind_novelty_max_similarity: float = 0.82
    rewind_novelty_retries: int = 3
    rewind_escape_temperature: float = 0.6
    rewind_step_top_p: float = 0.9
    rewind_step_repetition_penalty: float = 1.05

    # bridge-CoT extras
    enable_bridge: bool = True
    bridge_tail_source: str = "recovered"  # recovered|oracle|both
    bridge_samples_per_cell: int = 0       # 0 -> reuse n_samples_per_k
    bridge_num_prefix_points: int = 9      # 0 -> full 0..L
    bridge_num_tail_points: int = 9        # 0 -> full 0..D
    bridge_allow_overlap: bool = False
    bridge_middle_temperature: float = 0.0
    bridge_reconstruct_middle: bool = True
    bridge_middle_cases: int = 5
    bridge_loop_closure_samples: int = 3
    bridge_core_cluster_similarity: float = 0.72
    bridge_core_min_loop_closure: float = 0.75
    bridge_core_min_match: float = 0.75
    bridge_core_min_synergy: float = 0.0
    bridge_core_top_prototypes: int = 6
    save_raw_generations: bool = False
    save_generation_log_jsonl: bool = False

def build_trace_prompt(
    question: str,
    mode: str,
    prompt_version: str = "v1",
    prompt_family: Optional[str] = None,
    output_format: str = "json",
) -> str:
    if mode not in ("masked", "unmasked"):
        mode = "masked"
    resolved_format = resolve_trace_output_format(output_format, stream_output=False)
    if resolved_format == "text":
        template_name = "trace_masked_text.txt" if mode == "masked" else "trace_unmasked_text.txt"
    else:
        template_name = "trace_masked.txt" if mode == "masked" else "trace_unmasked.txt"
    return render_prompt_template(
        template_name,
        prompt_version=prompt_version,
        prompt_family=prompt_family,
        QUESTION=question,
    )

def parse_trace_output(question: str, raw: str) -> Trace:
    obj = extract_first_json_object(raw)
    if obj is None:
        obj = extract_trace_payload_fallback(raw)
    if obj is None:
        obj = extract_trace_payload_from_plaintext(raw)
    if obj is None:
        failure_hint = classify_trace_generation_failure(question, raw)
        hint_block = f"{failure_hint}\n" if failure_hint else ""
        raise RuntimeError(
            "Failed to extract a usable trace from model output.\n"
            f"{hint_block}"
            "Tip: try --trace-format text, use an instruct/chat model, or provide --trace-file with a prebuilt trace.\n"
            f"Raw output (first 400 chars): {raw[:400]!r}"
        )
    step_records = normalize_trace_step_records(obj.get("steps", []))
    steps = [record["text"] for record in step_records]
    answer = extract_trace_answer(obj)
    if not steps:
        raise RuntimeError("Trace output had no usable steps.")
    return Trace(steps=steps, answer=answer, step_records=step_records)

def print_trace_preview(trace: Trace, *, label: str, score: Optional[float] = None) -> None:
    title = f"\n=== {label} ==="
    if score is not None:
        title += f" [score={score:.3f}]"
    print(title, flush=True)
    for step in trace.steps:
        print(f"- {step}", flush=True)
    if trace.answer:
        print(f"Answer: {trace.answer}", flush=True)

def build_trace_once(
    backend: Backend,
    question: str,
    settings: ExperimentSettings,
    seed: int,
    *,
    temperature: Optional[float] = None,
    enable_stream: Optional[bool] = None,
    label: Optional[str] = None,
) -> Trace:
    early_hint = classify_trace_generation_failure(question, "")
    if early_hint and looks_like_placeholder_question(question):
        raise RuntimeError(
            "Trace generation skipped because the question is underspecified.\n"
            f"{early_hint}"
        )
    prompt = build_trace_prompt(
        question,
        settings.trace_mode,
        prompt_version=settings.prompt_version,
        prompt_family=settings.prompt_family,
        output_format=resolve_trace_output_format(
            settings.trace_format,
            stream_output=settings.stream_output if enable_stream is None else bool(enable_stream),
        ),
    )
    raw = emit_streamed_generation(
        backend,
        prompt,
        temperature=(settings.temperature_trace if temperature is None else float(temperature)),
        seed=seed,
        enable_stream=settings.stream_output if enable_stream is None else bool(enable_stream),
        label=label or f"{getattr(backend, 'name', 'backend')} trace (CoT)",
    )
    return parse_trace_output(question, raw)

def build_trace(backend: Backend, question: str, settings: ExperimentSettings, seed: int) -> Trace:
    return build_trace_once(backend, question, settings, seed)

def build_trace_bundle(
    backend: Backend,
    question: str,
    settings: ExperimentSettings,
    seed: int,
) -> Dict[str, Any]:
    base_trace = build_trace_once(
        backend,
        question,
        settings,
        seed,
        enable_stream=settings.stream_output,
        label=f"{getattr(backend, 'name', 'backend')} trace (CoT)",
    )
    mode = str(settings.process_reward_mode or "off").strip().lower()
    if mode != "lite" or settings.trace_candidates <= 1:
        return {
            "selected_trace": base_trace,
            "base_trace": base_trace,
            "process_reward": None,
        }

    candidate_rows: List[Dict[str, Any]] = []
    candidate_traces: Dict[int, Trace] = {}
    base_reward = score_trace_candidate_lite(question, base_trace)
    candidate_rows.append({
        "candidate_index": 0,
        "status": "ok",
        "seed": seed,
        "temperature": float(settings.temperature_trace),
        "trace": trace_to_payload(base_trace),
        "process_reward": base_reward,
    })
    candidate_traces[0] = base_trace

    candidate_temp = max(float(settings.temperature_trace), float(settings.trace_candidate_temperature))
    for candidate_index in range(1, max(1, int(settings.trace_candidates))):
        cand_seed = seed + 10_000 * candidate_index
        try:
            cand_trace = build_trace_once(
                backend,
                question,
                settings,
                cand_seed,
                temperature=candidate_temp,
                enable_stream=False,
                label=None,
            )
            cand_reward = score_trace_candidate_lite(question, cand_trace)
            candidate_rows.append({
                "candidate_index": candidate_index,
                "status": "ok",
                "seed": cand_seed,
                "temperature": candidate_temp,
                "trace": trace_to_payload(cand_trace),
                "process_reward": cand_reward,
            })
            candidate_traces[candidate_index] = cand_trace
        except Exception as e:
            candidate_rows.append({
                "candidate_index": candidate_index,
                "status": "error",
                "seed": cand_seed,
                "temperature": candidate_temp,
                "error": str(e),
            })

    successful = [row for row in candidate_rows if row.get("status") == "ok"]
    if not successful:
        return {
            "selected_trace": base_trace,
            "base_trace": base_trace,
            "process_reward": None,
        }

    def _candidate_rank_key(row: Dict[str, Any]) -> Tuple[float, float, int]:
        reward = row.get("process_reward") or {}
        return (
            float(reward.get("overall_score", 0.0) or 0.0),
            float(reward.get("mean_step_score", 0.0) or 0.0),
            len(((row.get("trace") or {}).get("steps") or [])),
        )

    selected_row = max(successful, key=_candidate_rank_key)
    selected_index = int(selected_row.get("candidate_index", 0) or 0)
    selected_trace = candidate_traces.get(selected_index, base_trace)
    process_reward = {
        "mode": "lite",
        "candidate_count_requested": int(settings.trace_candidates),
        "candidate_count_ok": len(successful),
        "selected_candidate_index": selected_index,
        "base_candidate_index": 0,
        "base_score": ((candidate_rows[0].get("process_reward") or {}).get("overall_score") if candidate_rows else None),
        "selected_score": ((selected_row.get("process_reward") or {}).get("overall_score") if selected_row else None),
        "score_gain_vs_base": (
            float((selected_row.get("process_reward") or {}).get("overall_score", 0.0) or 0.0)
            - float((candidate_rows[0].get("process_reward") or {}).get("overall_score", 0.0) or 0.0)
        ),
        "candidates": candidate_rows,
    }
    selected_trace.metadata = {"process_reward": process_reward}
    if settings.stream_output:
        print(
            f"[trace-prm] selected candidate {selected_index + 1}/{len(successful)} "
            f"score={float(process_reward.get('selected_score', 0.0) or 0.0):.3f} "
            f"(base={float(process_reward.get('base_score', 0.0) or 0.0):.3f})",
            flush=True,
        )
        if selected_index != 0:
            print_trace_preview(
                selected_trace,
                label=f"{getattr(backend, 'name', 'backend')} trace (PRM-selected)",
                score=float(process_reward.get("selected_score", 0.0) or 0.0),
            )
    return {
        "selected_trace": selected_trace,
        "base_trace": base_trace,
        "process_reward": process_reward,
    }

def reanswer_from_prefix(
    backend: Backend,
    question: str,
    steps: List[str],
    k: int,
    temperature: float,
    seed: int,
    prompt_version: str = "v1",
    prompt_family: Optional[str] = None,
) -> str:
    prefix = steps[:k]
    steps_block = "\n".join(f"- {s}" for s in prefix) if prefix else "(no steps)"
    prompt = render_prompt_template(
        "reanswer_from_prefix.txt",
        prompt_version=prompt_version,
        prompt_family=prompt_family,
        QUESTION=question,
        K=k,
        STEPS_BLOCK=steps_block,
    )
    return backend.generate(prompt, temperature=temperature, seed=seed).strip()


def build_rewind_step_prompt(
    question: str,
    answer: str,
    recovered_latest_to_earlier: List[str],
    prompt_version: str = "v1",
    prompt_family: Optional[str] = None,
    output_format: str = "json",
    novelty_block: str = "",
) -> str:
    later_block = "\n".join(f"- {s}" for s in recovered_latest_to_earlier) if recovered_latest_to_earlier else "(none yet)"
    return render_prompt_template(
        "rewind_step_text.txt" if resolve_trace_output_format(output_format, stream_output=False) == "text" else "rewind_step.txt",
        prompt_version=prompt_version,
        prompt_family=prompt_family,
        QUESTION=question,
        ANSWER=answer,
        LATER_BLOCK=later_block,
        NOVELTY_BLOCK=novelty_block.strip(),
    )

def build_rewind_answer_prompt(
    question: str,
    latest_to_earlier_steps: List[str],
    depth: int,
    order: str,
    prompt_version: str = "v1",
    prompt_family: Optional[str] = None,
) -> str:
    tail = latest_to_earlier_steps[:depth]
    if order == "forward":
        shown = list(reversed(tail))
        order_desc = "earlier to later"
        template_name = "rewind_answer_forward.txt"
    else:
        shown = tail
        order_desc = "latest to earlier"
        template_name = "rewind_answer_reverse.txt"
    block = "\n".join(f"- {s}" for s in shown) if shown else "(no rewind steps)"
    return render_prompt_template(
        template_name,
        prompt_version=prompt_version,
        prompt_family=prompt_family,
        QUESTION=question,
        ORDER_DESC=order_desc,
        BLOCK=block,
    )

def build_rewind_novelty_block(
    recovered_latest_to_earlier: List[str],
    *,
    attempt: int,
    max_items: int = 4,
) -> str:
    if not recovered_latest_to_earlier:
        return ""
    recent = recovered_latest_to_earlier[-max_items:]
    lines = [
        "Novelty constraints:",
        "- The new step must add at least one concept not already present in the later recovered steps.",
        "- It must not be semantically equivalent to any recovered step.",
        "- Prefer a local prerequisite, not a compressed summary.",
        "Avoid paraphrasing these recovered steps:",
    ]
    lines.extend(f"- {step}" for step in recent)
    if attempt > 0:
        lines.append(f"- This is retry {attempt}; be more local and more distinct than before.")
    return "\n".join(lines)

def recover_rewind_trace(
    backend: Backend,
    question: str,
    trace: Trace,
    settings: ExperimentSettings,
    run_seed: int
) -> Dict[str, Any]:
    steps = trace.steps[: settings.max_steps]
    original_latest_to_earlier = list(reversed(steps))
    recovered_latest_to_earlier: List[str] = []
    base_recovered_latest_to_earlier: List[str] = []
    raw_recoveries: List[Dict[str, Any]] = []
    recovered_records: List[Dict[str, Any]] = []
    novelty_rows: List[Dict[str, Any]] = []
    rewind_score_rows: List[Dict[str, Any]] = []
    rewind_prm_mode = str(settings.rewind_process_reward_mode or "off").strip().lower()
    generation_calls = 0
    generation_attempts = 0

    for depth in range(1, len(steps) + 1):
        best_candidate: Optional[Dict[str, Any]] = None
        attempt_rows: List[Dict[str, Any]] = []
        base_candidate_for_depth: Optional[Dict[str, Any]] = None
        for attempt in range(0, settings.rewind_novelty_retries + 1):
            generation_attempts += 1
            prompt = build_rewind_step_prompt(
                question,
                trace.answer,
                recovered_latest_to_earlier,
                prompt_version=settings.prompt_version,
                prompt_family=settings.prompt_family,
                output_format=resolve_trace_output_format(settings.trace_format, stream_output=settings.stream_output),
                novelty_block=build_rewind_novelty_block(recovered_latest_to_earlier, attempt=attempt),
            )
            step_temp = (
                settings.rewind_step_temperature
                if attempt == 0
                else max(settings.rewind_escape_temperature, settings.rewind_step_temperature + 0.1 * attempt)
            )
            candidate_variants: List[Dict[str, Any]] = []
            n_candidates = int(settings.rewind_step_candidates if rewind_prm_mode == "lite" else 1)
            for candidate_index in range(max(1, n_candidates)):
                with temporary_backend_generation_overrides(
                    backend,
                    top_p=settings.rewind_step_top_p,
                    repetition_penalty=settings.rewind_step_repetition_penalty,
                ):
                    raw = emit_streamed_generation(
                        backend,
                        prompt,
                        temperature=step_temp,
                        seed=run_seed + 700_000 + depth * 1000 + attempt * 100 + candidate_index,
                        enable_stream=(settings.stream_output and n_candidates <= 1),
                        label=f"{getattr(backend, 'name', 'backend')} rewind step {depth}/{len(steps)} attempt {attempt + 1}",
                    )
                parsed = parse_rewind_step_output(raw)
                step = parsed["step"] or f"(rewind step {depth} unavailable)"
                novelty_info = rewind_step_novelty_info(
                    step,
                    recovered_latest_to_earlier,
                    settings.rewind_novelty_max_similarity,
                )
                rewind_reward = score_rewind_candidate_lite(
                    question,
                    trace.answer,
                    step,
                    recovered_latest_to_earlier,
                    novelty_annotation=parsed.get("novelty") or "",
                    why_earlier=parsed.get("why_earlier") or "",
                )
                candidate_variants.append({
                    "depth": depth,
                    "attempt": attempt + 1,
                    "candidate_index": candidate_index,
                    "temperature": step_temp,
                    "step": step,
                    "novelty_annotation": parsed.get("novelty") or "",
                    "why_earlier": parsed.get("why_earlier") or "",
                    "raw_output": raw,
                    **novelty_info,
                    "process_reward": rewind_reward,
                })
            generation_calls += len(candidate_variants)
            if attempt == 0 and candidate_variants:
                base_candidate_for_depth = candidate_variants[0]

            def _rewind_rank_key(row: Dict[str, Any]) -> Tuple[int, float, float, float]:
                reward = row.get("process_reward") or {}
                return (
                    1 if not row.get("is_fixed_point_candidate") else 0,
                    float(reward.get("score", 0.0) or 0.0),
                    float(row.get("novelty", 0.0) or 0.0),
                    -float(row.get("max_similarity_to_previous", 0.0) or 0.0),
                )

            candidate = max(candidate_variants, key=_rewind_rank_key)
            if settings.stream_output and n_candidates > 1:
                reward = candidate.get("process_reward") or {}
                print(
                    f"[rewind-prm] depth={depth} attempt={attempt + 1} "
                    f"selected candidate {int(candidate.get('candidate_index', 0)) + 1}/{n_candidates} "
                    f"score={float(reward.get('score', 0.0) or 0.0):.3f} "
                    f"novelty={float(candidate.get('novelty', 0.0) or 0.0):.3f}",
                    flush=True,
                )
                print(f"- {str(candidate.get('step') or '').strip()}", flush=True)
            attempt_rows.append(candidate)
            if best_candidate is None or _rewind_rank_key(candidate) > _rewind_rank_key(best_candidate):
                best_candidate = candidate
            if not candidate["is_fixed_point_candidate"]:
                break

        chosen = best_candidate or {
            "depth": depth,
            "attempt": 1,
            "temperature": settings.rewind_step_temperature,
            "step": f"(rewind step {depth} unavailable)",
            "novelty_annotation": "",
            "why_earlier": "",
            "raw_output": "",
            "max_similarity_to_previous": 0.0,
            "max_char_similarity_to_previous": 0.0,
            "max_jaccard_similarity_to_previous": 0.0,
            "closest_previous_step": None,
            "novelty": 1.0,
            "is_fixed_point_candidate": False,
            "process_reward": {"score": 0.0},
        }
        base_candidate = base_candidate_for_depth or chosen
        step = str(chosen["step"]).strip() or f"(rewind step {depth} unavailable)"
        chosen_reward = dict(chosen.get("process_reward") or {})
        recovered_latest_to_earlier.append(step)
        base_recovered_latest_to_earlier.append(str(base_candidate.get("step") or "").strip() or f"(rewind step {depth} unavailable)")
        recovered_records.append({
            "depth": depth,
            "step": step,
            "attempts_used": len(attempt_rows),
            "selected_attempt": chosen.get("attempt"),
            "selected_candidate_index": chosen.get("candidate_index"),
            "novelty_annotation": chosen.get("novelty_annotation") or "",
            "why_earlier": chosen.get("why_earlier") or "",
            "process_reward_score": chosen_reward.get("score"),
            "process_reward_atomicity": chosen_reward.get("atomicity"),
            "process_reward_relevance": chosen_reward.get("relevance"),
        })
        novelty_rows.append({
            "depth": depth,
            "step": step,
            "max_similarity_to_previous": float(chosen.get("max_similarity_to_previous", 0.0) or 0.0),
            "max_char_similarity_to_previous": float(chosen.get("max_char_similarity_to_previous", 0.0) or 0.0),
            "max_jaccard_similarity_to_previous": float(chosen.get("max_jaccard_similarity_to_previous", 0.0) or 0.0),
            "closest_previous_step": chosen.get("closest_previous_step"),
            "novelty": float(chosen.get("novelty", 0.0) or 0.0),
            "is_fixed_point_candidate": bool(chosen.get("is_fixed_point_candidate", False)),
            "selected_attempt": int(chosen.get("attempt", 1) or 1),
        })
        rewind_score_rows.append({
            "depth": depth,
            "step": step,
            "score": float(chosen_reward.get("score", 0.0) or 0.0),
            "atomicity": float(chosen_reward.get("atomicity", 0.0) or 0.0),
            "relevance": float(chosen_reward.get("relevance", 0.0) or 0.0),
            "answer_leak_penalty": float(chosen_reward.get("answer_leak_penalty", 0.0) or 0.0),
            "summary_penalty": float(chosen_reward.get("summary_penalty", 0.0) or 0.0),
            "novelty": float(chosen_reward.get("novelty", 0.0) or 0.0),
        })
        if settings.save_raw_generations:
            raw_recoveries.append({
                "depth": depth,
                "raw_output": chosen.get("raw_output", ""),
                "parsed_step": step,
                "attempts": attempt_rows,
            })

    alignment_summary = summarize_rewind_alignment(original_latest_to_earlier, recovered_latest_to_earlier)
    novelty_summary = summarize_rewind_novelty(
        novelty_rows,
        threshold=settings.rewind_novelty_max_similarity,
    )
    base_rewind_summary = summarize_rewind_sequence_metrics(
        question,
        trace.answer,
        base_recovered_latest_to_earlier,
        settings.rewind_novelty_max_similarity,
    )
    base_rewind_alignment = summarize_rewind_alignment(original_latest_to_earlier, base_recovered_latest_to_earlier)
    base_rewind = {
        "latest_to_earlier": base_recovered_latest_to_earlier,
        "forward_order": list(reversed(base_recovered_latest_to_earlier)),
        **base_rewind_summary,
        **base_rewind_alignment,
    }

    out = {
        "latest_to_earlier": recovered_latest_to_earlier,
        "forward_order": list(reversed(recovered_latest_to_earlier)),
        "step_records": list(reversed(recovered_records)),
        **alignment_summary,
        "process_reward_mode": rewind_prm_mode,
        "process_reward_step_candidates": int(settings.rewind_step_candidates if rewind_prm_mode == "lite" else 1),
        "process_reward_step_scores": rewind_score_rows,
        "process_reward_mean": mean_or_none([float(row.get("score", 0.0) or 0.0) for row in rewind_score_rows]),
        "process_reward_base_mean": base_rewind.get("process_reward_mean"),
        "process_reward_score_gain_vs_base": (
            float(mean_or_none([float(row.get("score", 0.0) or 0.0) for row in rewind_score_rows]) or 0.0)
            - float(base_rewind.get("process_reward_mean", 0.0) or 0.0)
        ),
        "generation_calls": generation_calls,
        "generation_attempts": generation_attempts,
        "base_rewind": base_rewind,
        **novelty_summary,
    }
    if settings.save_raw_generations:
        out["raw_recoveries"] = raw_recoveries
    return out

def reanswer_from_rewind_tail(
    backend: Backend,
    question: str,
    latest_to_earlier_steps: List[str],
    depth: int,
    temperature: float,
    seed: int,
    order: str = "reverse",
    prompt_version: str = "v1",
    prompt_family: Optional[str] = None,
) -> str:
    prompt = build_rewind_answer_prompt(
        question,
        latest_to_earlier_steps,
        depth,
        order=order,
        prompt_version=prompt_version,
        prompt_family=prompt_family,
    )
    return backend.generate(prompt, temperature=temperature, seed=seed).strip()

def compute_tail_curve(
    backend: Backend,
    question: str,
    latest_to_earlier_steps: List[str],
    settings: ExperimentSettings,
    run_seed: int,
    baseline_A_star: str,
    baseline_distribution: Dict[str, int],
    label: str,
) -> Dict[str, Any]:
    D = len(latest_to_earlier_steps)
    n_samples = settings.rewind_samples_per_depth if settings.rewind_samples_per_depth > 0 else settings.n_samples_per_k
    base_counter = Counter({str(k): int(v) for k, v in baseline_distribution.items()})
    save_raw = bool(settings.save_raw_generations)

    curve: List[CurvePoint] = []
    raw_answers_by_depth: List[Dict[str, Any]] = []
    base_support = list(base_counter.keys())

    for depth in range(0, D + 1):
        answers_depth: List[str] = []
        for j in range(n_samples):
            answers_depth.append(
                reanswer_from_rewind_tail(
                    backend,
                    question,
                    latest_to_earlier_steps,
                    depth,
                    temperature=settings.temperature_reanswer,
                    seed=run_seed + 900_000 + depth * 1000 + j,
                    order=settings.rewind_order,
                    prompt_version=settings.prompt_version,
                    prompt_family=settings.prompt_family,
                )
            )
        if save_raw:
            raw_answers_by_depth.append({"depth": depth, "answers": answers_depth[:]})
        dist_d = dist_from_answers(answers_depth)
        match = 0.0
        if baseline_A_star:
            match = dist_d.get(baseline_A_star, 0) / max(1, sum(dist_d.values()))
        top = dist_d.most_common(1)[0][0] if dist_d else ""
        ent = entropy_bits_from_counts(dist_d)

        jsd = None
        kld = None
        if settings.compute_divergence:
            support = sorted(set(base_support) | set(dist_d.keys()))
            p = _to_prob(dist_d, support, alpha=settings.divergence_alpha)
            q = _to_prob(base_counter, support, alpha=settings.divergence_alpha)
            jsd = js_divergence(p, q)
            kld = kl_divergence(p, q)

        curve.append(CurvePoint(k=depth, match_rate=match, entropy_bits=ent, top_answer=top, js_div_full=jsd, kl_div_full=kld))

    match_series = [pt.match_rate for pt in curve]
    smooth_match = moving_average(match_series, settings.smooth_window)
    converge_depth = find_converge_k(smooth_match, settings.tau, settings.converge_min_run)
    collapse_depth = find_collapse_k(smooth_match, settings.tau, settings.collapse_min_run)
    d = discrete_derivative(smooth_match)
    max_slope = max(d) if d else None
    max_slope_depth = (d.index(max_slope) if (d and max_slope is not None) else None)

    plateau_mean = None
    if converge_depth is not None:
        tail = smooth_match[converge_depth:]
        plateau_mean = sum(tail) / max(1, len(tail))
    else:
        start = int(max(0, math.floor(0.8 * D)))
        tail = smooth_match[start:] if smooth_match else []
        plateau_mean = sum(tail) / max(1, len(tail)) if tail else None

    conv_score = 0.0
    if converge_depth is not None and D > 0:
        conv_score = 1.0 - min(converge_depth, D) / float(D)
        conv_score = max(0.0, min(1.0, conv_score))
    plateau = plateau_mean if plateau_mean is not None else 0.0
    attractor_strength = conv_score * plateau
    auc_match = sum(smooth_match) / max(1, len(smooth_match)) if smooth_match else 0.0

    out = {
        "label": label,
        "depth_max": D,
        "order": settings.rewind_order,
        "answer_calls": (D + 1) * n_samples,
        "converge_depth": converge_depth,
        "collapse_depth": collapse_depth,
        "stabilization_max_slope": max_slope,
        "stabilization_max_slope_depth": max_slope_depth,
        "plateau_mean": plateau_mean,
        "attractor_strength": attractor_strength,
        "auc_match": auc_match,
        "curve": [dataclasses.asdict(pt) for pt in curve],
        "smooth_match_rate": smooth_match,
    }
    if save_raw:
        out["raw_answers_by_depth"] = raw_answers_by_depth
    return out

def compute_rewind_bundle(
    backend: Backend,
    question: str,
    trace: Trace,
    settings: ExperimentSettings,
    forward_result: Dict[str, Any],
    run_seed: int,
) -> Dict[str, Any]:
    t_bundle = time.perf_counter()
    baseline_A_star = str(forward_result.get("A_star", ""))
    baseline_distribution = dict(forward_result.get("baseline_distribution", {}) or {})
    original_latest_to_earlier = list(reversed(trace.steps[: settings.max_steps]))

    t0 = time.perf_counter()
    rewind_trace = recover_rewind_trace(backend, question, trace, settings, run_seed=run_seed + 50_000)
    rewind_trace_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    rewind_curve = compute_tail_curve(
        backend, question, rewind_trace["latest_to_earlier"], settings,
        run_seed=run_seed + 60_000,
        baseline_A_star=baseline_A_star,
        baseline_distribution=baseline_distribution,
        label="recovered_rewind_tail",
    )
    rewind_curve_s = time.perf_counter() - t0

    oracle_tail_curve = None
    oracle_tail_curve_s = 0.0
    if settings.compute_oracle_tail:
        t0 = time.perf_counter()
        oracle_tail_curve = compute_tail_curve(
            backend, question, original_latest_to_earlier, settings,
            run_seed=run_seed + 70_000,
            baseline_A_star=baseline_A_star,
            baseline_distribution=baseline_distribution,
            label="oracle_original_tail",
        )
        oracle_tail_curve_s = time.perf_counter() - t0

    forward_curve = forward_result.get("curve", []) or []
    rewind_curve_points = rewind_curve.get("curve", []) or []
    directional_gap_mean = None
    directional_gap_auc = None
    if forward_curve and rewind_curve_points and len(forward_curve) == len(rewind_curve_points):
        gaps = [float(f.get("match_rate", 0.0)) - float(r.get("match_rate", 0.0)) for f, r in zip(forward_curve, rewind_curve_points)]
        directional_gap_mean = mean_or_none(gaps)
        directional_gap_auc = float(forward_result.get("auc_match", 0.0)) - float(rewind_curve.get("auc_match", 0.0))

    oracle_recovery_gap_auc = None
    oracle_recovery_gap_mean = None
    if oracle_tail_curve is not None:
        oracle_recovery_gap_auc = float(oracle_tail_curve.get("auc_match", 0.0)) - float(rewind_curve.get("auc_match", 0.0))
        oc = oracle_tail_curve.get("curve", []) or []
        rc = rewind_curve.get("curve", []) or []
        if oc and rc and len(oc) == len(rc):
            gaps = [float(o.get("match_rate", 0.0)) - float(r.get("match_rate", 0.0)) for o, r in zip(oc, rc)]
            oracle_recovery_gap_mean = mean_or_none(gaps)
    rewind_core = compute_rewind_core_metrics(rewind_trace, rewind_curve)
    rewind_workload = {
        "rewind_trace_generation_calls": rewind_trace.get("generation_calls"),
        "rewind_trace_attempts": rewind_trace.get("generation_attempts"),
        "rewind_tail_answer_calls": rewind_curve.get("answer_calls"),
        "oracle_tail_answer_calls": (oracle_tail_curve.get("answer_calls") if isinstance(oracle_tail_curve, dict) else 0),
    }
    rewind_workload["rewind_total_generation_calls"] = (
        int(rewind_workload.get("rewind_trace_generation_calls") or 0)
        + int(rewind_workload.get("rewind_tail_answer_calls") or 0)
        + int(rewind_workload.get("oracle_tail_answer_calls") or 0)
    )
    rewind_timings = {
        "rewind_trace_s": rewind_trace_s,
        "rewind_curve_s": rewind_curve_s,
        "oracle_tail_curve_s": oracle_tail_curve_s,
        "rewind_total_s": time.perf_counter() - t_bundle,
    }

    return {
        "rewind_enabled": True,
        "rewind_trace": rewind_trace,
        "rewind_curve": rewind_curve,
        "oracle_tail_curve": oracle_tail_curve,
        "directional_gap_mean": directional_gap_mean,
        "directional_gap_auc": directional_gap_auc,
        "oracle_recovery_gap_auc": oracle_recovery_gap_auc,
        "oracle_recovery_gap_mean": oracle_recovery_gap_mean,
        "rewind_core": rewind_core,
        "rewind_workload": rewind_workload,
        "rewind_timings_s": rewind_timings,
    }


def evenly_spaced_ints(max_value: int, n_points: int) -> List[int]:
    if max_value <= 0:
        return [0]
    if n_points <= 0 or n_points >= (max_value + 1):
        return list(range(0, max_value + 1))
    if n_points == 1:
        return [0]
    pts = {0, max_value}
    for i in range(1, max(1, n_points - 1)):
        val = int(round(i * max_value / float(max(1, n_points - 1))))
        pts.add(max(0, min(max_value, val)))
    return sorted(pts)

def build_bridge_answer_prompt(
    question: str,
    prefix_steps: List[str],
    k: int,
    latest_to_earlier_steps: List[str],
    depth: int,
    order: str,
    prompt_version: str = "v1",
    prompt_family: Optional[str] = None,
) -> str:
    prefix_block = "\n".join(f"- {s}" for s in prefix_steps[:k]) if k > 0 else "(no prefix steps)"
    tail = latest_to_earlier_steps[:depth]
    if order == "forward":
        shown_tail = list(reversed(tail))
        order_desc = "earlier to later"
        tail_hint = "This late-side trace has already been unrolled into the forward direction."
    else:
        shown_tail = tail
        order_desc = "latest to earlier"
        tail_hint = "Mentally unroll the late-side trace into the forward direction before bridging."
    tail_block = "\n".join(f"- {s}" for s in shown_tail) if shown_tail else "(no tail steps)"
    return render_prompt_template(
        "bridge_answer.txt",
        prompt_version=prompt_version,
        prompt_family=prompt_family,
        QUESTION=question,
        PREFIX_BLOCK=prefix_block,
        ORDER_DESC=order_desc,
        TAIL_BLOCK=tail_block,
        TAIL_HINT=tail_hint,
    )

def reanswer_from_bridge(
    backend: Backend,
    question: str,
    prefix_steps: List[str],
    k: int,
    latest_to_earlier_steps: List[str],
    depth: int,
    temperature: float,
    seed: int,
    order: str = "reverse",
    prompt_version: str = "v1",
    prompt_family: Optional[str] = None,
) -> str:
    prompt = build_bridge_answer_prompt(
        question,
        prefix_steps,
        k,
        latest_to_earlier_steps,
        depth,
        order=order,
        prompt_version=prompt_version,
        prompt_family=prompt_family,
    )
    return backend.generate(prompt, temperature=temperature, seed=seed).strip()

def build_bridge_middle_prompt(
    question: str,
    prefix_steps: List[str],
    k: int,
    latest_to_earlier_steps: List[str],
    depth: int,
    order: str,
    approx_missing_steps: Optional[int] = None,
    prompt_version: str = "v1",
    prompt_family: Optional[str] = None,
) -> str:
    prefix_block = "\n".join(f"- {s}" for s in prefix_steps[:k]) if k > 0 else "(no prefix steps)"
    tail = latest_to_earlier_steps[:depth]
    if order == "forward":
        shown_tail = list(reversed(tail))
        order_desc = "earlier to later"
        tail_hint = "Treat the late-side trace as already unrolled into the forward direction."
    else:
        shown_tail = tail
        order_desc = "latest to earlier"
        tail_hint = "Mentally unroll the late-side trace into the forward direction before filling the middle."
    tail_block = "\n".join(f"- {s}" for s in shown_tail) if shown_tail else "(no tail steps)"
    approx_rule = ""
    if approx_missing_steps is not None:
        approx_rule = f"- Aim for about {approx_missing_steps} missing middle step(s) if that seems coherent."
    return render_prompt_template(
        "bridge_middle.txt",
        prompt_version=prompt_version,
        prompt_family=prompt_family,
        QUESTION=question,
        PREFIX_BLOCK=prefix_block,
        ORDER_DESC=order_desc,
        TAIL_BLOCK=tail_block,
        TAIL_HINT=tail_hint,
        APPROX_RULE=approx_rule,
    )



def tail_forward_from_latest(latest_to_earlier_steps: List[str], depth: int) -> List[str]:
    return list(reversed(latest_to_earlier_steps[: max(0, int(depth))]))

def compose_bridge_forward_trace(
    forward_steps: List[str],
    k: int,
    latest_to_earlier_steps: List[str],
    depth: int,
    middle_steps: List[str],
) -> Tuple[List[str], List[str], int]:
    """
    Build a forward-direction trace from:
    - known prefix forward_steps[:k]
    - reconstructed middle_steps
    - tail-side steps latest_to_earlier_steps[:depth] (stored latest->earlier, so reverse to forward)
    If prefix and tail overlap in the original timeline, trim the earliest part of the tail to avoid duplication.
    Returns: (full_trace_steps, trimmed_tail_forward_steps, overlap_steps_trimmed)
    """
    L = len(forward_steps)
    prefix = list(forward_steps[: max(0, int(k))])
    tail_forward = tail_forward_from_latest(latest_to_earlier_steps, depth)
    overlap_steps = max(0, (int(k) + int(depth)) - int(L))
    if overlap_steps > 0:
        tail_forward = tail_forward[overlap_steps:]
    return prefix + list(middle_steps) + tail_forward, tail_forward, overlap_steps

def alignment_stats(seq_a: List[str], seq_b: List[str], left_name: str = "original", right_name: str = "reconstructed") -> Dict[str, Any]:
    pairwise: List[Dict[str, Any]] = []
    sims: List[float] = []
    n = max(len(seq_a), len(seq_b))
    for i in range(n):
        a = seq_a[i] if i < len(seq_a) else ""
        b = seq_b[i] if i < len(seq_b) else ""
        sim = sequence_similarity(a, b)
        sims.append(sim)
        pairwise.append({
            "index": i,
            f"{left_name}_step": a,
            f"{right_name}_step": b,
            "similarity": sim,
        })
    return {
        "pairwise_alignment": pairwise,
        "similarity_mean": mean_or_none(sims),
        "similarity_concat": sequence_similarity("\n".join(seq_a), "\n".join(seq_b)),
        "len_left": len(seq_a),
        "len_right": len(seq_b),
    }

def closure_answer_stats(
    backend: Backend,
    question: str,
    composed_steps: List[str],
    baseline_A_star: str,
    temperature: float,
    seed_base: int,
    n_samples: int,
    prompt_version: str = "v1",
    prompt_family: Optional[str] = None,
) -> Dict[str, Any]:
    n = max(1, int(n_samples))
    answers: List[str] = []
    for j in range(n):
        answers.append(
            reanswer_from_prefix(
                backend,
                question,
                composed_steps,
                len(composed_steps),
                temperature=temperature,
                seed=seed_base + j,
                prompt_version=prompt_version,
                prompt_family=prompt_family,
            )
        )
    dist = dist_from_answers(answers)
    match = 0.0
    if baseline_A_star:
        match = dist.get(baseline_A_star, 0) / max(1, sum(dist.values()))
    return {
        "n_samples": n,
        "distribution": dict(dist),
        "match_rate": match,
        "entropy_bits": entropy_bits_from_counts(dist),
        "top_answer": dist.most_common(1)[0][0] if dist else "",
    }

def choose_bridge_reconstruction_cells(
    valid_cells: List[Dict[str, Any]],
    best_cell: Optional[Dict[str, Any]],
    best_synergy_cell: Optional[Dict[str, Any]],
    settings: ExperimentSettings,
) -> List[Tuple[str, Dict[str, Any]]]:
    if not valid_cells or settings.bridge_middle_cases <= 0:
        return []

    max_budget = max(int(c["budget"]) for c in valid_cells) if valid_cells else 0
    incomplete = [c for c in valid_cells if 0 < int(c["budget"]) < max_budget]

    selected: List[Tuple[str, Dict[str, Any]]] = []
    seen = set()

    def add(tag: str, cell: Optional[Dict[str, Any]]) -> None:
        if cell is None:
            return
        key = (int(cell["k"]), int(cell["depth"]))
        if key in seen:
            return
        seen.add(key)
        selected.append((tag, cell))

    add("best_match", best_cell)
    add("best_synergy", best_synergy_cell)

    if incomplete:
        add(
            "best_incomplete",
            max(
                incomplete,
                key=lambda c: (
                    float(c.get("match_rate", 0.0)),
                    float(c.get("synergy_over_max", 0.0)),
                    -int(c.get("budget", 0)),
                ),
            ),
        )
        mid_target = max_budget / 2.0
        add(
            "mid_budget_balanced",
            min(
                incomplete,
                key=lambda c: (
                    abs(float(c["budget"]) - mid_target),
                    abs(int(c["k"]) - int(c["depth"])),
                    -float(c.get("match_rate", 0.0)),
                ),
            ),
        )
        low_budget_positive = [c for c in incomplete if float(c.get("synergy_over_max", 0.0)) > 0.0]
        if low_budget_positive:
            add(
                "low_budget_positive_synergy",
                sorted(
                    low_budget_positive,
                    key=lambda c: (
                        int(c["budget"]),
                        -float(c.get("match_rate", 0.0)),
                        -float(c.get("synergy_over_max", 0.0)),
                    ),
                )[0],
            )

    ranked = sorted(
        valid_cells,
        key=lambda c: (
            -(0.70 * float(c.get("match_rate", 0.0)) + 0.30 * max(0.0, float(c.get("synergy_over_max", 0.0)))),
            int(c.get("budget", 0)),
            abs(int(c["k"]) - int(c["depth"])),
        ),
    )
    for idx, cell in enumerate(ranked):
        add(f"ranked_{idx+1}", cell)
        if len(selected) >= max(0, settings.bridge_middle_cases):
            break

    return selected[: max(0, settings.bridge_middle_cases)]

def cluster_bridge_steps(step_items: List[Dict[str, Any]], sim_threshold: float, top_n: int) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for item in step_items:
        step = str(item.get("step", "")).strip()
        if not step:
            continue
        best_idx = None
        best_sim = -1.0
        for i, cl in enumerate(clusters):
            sim = sequence_similarity(step, str(cl["representative_step"]))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx is not None and best_sim >= sim_threshold:
            cl = clusters[best_idx]
            cl["members"].append(item)
            if len(step) > len(str(cl["representative_step"])):
                cl["representative_step"] = step
        else:
            clusters.append({
                "representative_step": step,
                "members": [item],
            })

    out: List[Dict[str, Any]] = []
    for cl in clusters:
        members = cl["members"]
        out.append({
            "representative_step": cl["representative_step"],
            "support_count": len(members),
            "weighted_support": sum(float(m.get("weight", 1.0)) for m in members),
            "mean_weight": mean_or_none([float(m.get("weight", 1.0)) for m in members]),
            "example_tags": sorted(set(str(m.get("example_tag", "")) for m in members if m.get("example_tag"))),
            "mean_budget": mean_or_none([float(m.get("budget", 0.0)) for m in members]),
            "mean_loop_closure_score": mean_or_none([float(m.get("loop_closure_score", 0.0)) for m in members]),
            "member_steps": [str(m.get("step", "")) for m in members],
        })
    out = sorted(
        out,
        key=lambda c: (
            -float(c.get("weighted_support", 0.0)),
            -int(c.get("support_count", 0)),
            len(str(c.get("representative_step", ""))),
        ),
    )
    return out[: max(0, int(top_n))]

def summarize_bridge_core(
    examples: List[Dict[str, Any]],
    settings: ExperimentSettings,
    label: str,
    max_budget: int,
) -> Dict[str, Any]:
    if not examples:
        return {
            "best_core_candidate": None,
            "minimal_core_candidate": None,
            "core_examples_ranked": [],
            "core_step_prototypes": [],
        }

    scored_examples: List[Dict[str, Any]] = []
    step_items: List[Dict[str, Any]] = []

    denom_budget = max(1, int(max_budget))
    for ex in examples:
        budget = int(ex.get("budget", 0))
        match = float(ex.get("cell_match_rate", 0.0) or 0.0)
        synergy = float(ex.get("cell_synergy_over_max", 0.0) or 0.0)
        loop_score = float(ex.get("loop_closure_score", 0.0) or 0.0)
        compactness = 1.0 - min(budget, denom_budget) / float(denom_budget)
        synergy_pos = min(1.0, max(0.0, synergy))
        core_support = 0.5 * match + 0.5 * synergy_pos
        core_score = loop_score * core_support * (0.5 + 0.5 * compactness)

        ex["core_compactness"] = compactness
        ex["core_support"] = core_support
        ex["core_score"] = core_score
        scored_examples.append(ex)

        for step in ex.get("reconstructed_middle_steps", []) or []:
            step_items.append({
                "step": step,
                "weight": core_score if core_score > 0 else loop_score,
                "loop_closure_score": loop_score,
                "budget": budget,
                "example_tag": ex.get("tag"),
            })

    ranked = sorted(
        scored_examples,
        key=lambda ex: (
            -float(ex.get("core_score", 0.0)),
            int(ex.get("budget", 0)),
            -float(ex.get("loop_closure_score", 0.0)),
            -float(ex.get("cell_match_rate", 0.0)),
        ),
    )
    viable = [
        ex for ex in ranked
        if float(ex.get("loop_closure_score", 0.0)) >= float(settings.bridge_core_min_loop_closure)
        and float(ex.get("cell_match_rate", 0.0)) >= float(settings.bridge_core_min_match)
        and float(ex.get("cell_synergy_over_max", 0.0)) >= float(settings.bridge_core_min_synergy)
    ]
    minimal = None
    if viable:
        minimal = sorted(
            viable,
            key=lambda ex: (
                int(ex.get("budget", 0)),
                -float(ex.get("loop_closure_score", 0.0)),
                -float(ex.get("cell_match_rate", 0.0)),
                -float(ex.get("cell_synergy_over_max", 0.0)),
            ),
        )[0]

    prototypes = cluster_bridge_steps(
        step_items,
        sim_threshold=float(settings.bridge_core_cluster_similarity),
        top_n=int(settings.bridge_core_top_prototypes),
    )

    return {
        "best_core_candidate": ranked[0] if ranked else None,
        "minimal_core_candidate": minimal,
        "core_examples_ranked": ranked[: max(0, settings.bridge_middle_cases)],
        "core_step_prototypes": prototypes,
    }

def reconstruct_bridge_middle(
    backend: Backend,
    question: str,
    trace: Trace,
    latest_to_earlier_steps: List[str],
    settings: ExperimentSettings,
    k: int,
    depth: int,
    tag: str,
    run_seed: int,
    baseline_A_star: str,
) -> Dict[str, Any]:
    steps = trace.steps[: settings.max_steps]
    L = len(steps)
    approx_missing = max(0, L - k - depth)
    prompt = build_bridge_middle_prompt(
        question,
        steps,
        k,
        latest_to_earlier_steps,
        depth,
        order=settings.rewind_order,
        approx_missing_steps=approx_missing,
        prompt_version=settings.prompt_version,
        prompt_family=settings.prompt_family,
    )
    raw = backend.generate(prompt, temperature=settings.bridge_middle_temperature, seed=run_seed + 1_700_000 + k * 1000 + depth)
    obj = extract_first_json_object(raw)
    middle_steps: List[str] = []
    if obj is not None and isinstance(obj.get("middle_steps"), list):
        middle_steps = [str(s).strip() for s in obj.get("middle_steps", []) if str(s).strip()]
    else:
        lines = []
        for line in (raw or "").splitlines():
            s = line.strip().lstrip("-*•").strip()
            if s:
                lines.append(s)
        middle_steps = lines

    orig_middle_end = max(k, min(L, L - depth))
    original_middle = steps[k:orig_middle_end]
    pairwise: List[Dict[str, Any]] = []
    sims: List[float] = []
    n = max(len(original_middle), len(middle_steps))
    for i in range(n):
        o = original_middle[i] if i < len(original_middle) else ""
        r = middle_steps[i] if i < len(middle_steps) else ""
        sim = sequence_similarity(o, r)
        sims.append(sim)
        pairwise.append({
            "index": i,
            "original_step": o,
            "reconstructed_step": r,
            "similarity": sim,
        })

    concat_sim = sequence_similarity("\n".join(original_middle), "\n".join(middle_steps))

    reconstructed_full_trace, trimmed_tail_forward, overlap_steps = compose_bridge_forward_trace(
        steps,
        k,
        latest_to_earlier_steps,
        depth,
        middle_steps,
    )
    original_tail_forward = steps[orig_middle_end:]
    tail_align = alignment_stats(original_tail_forward, trimmed_tail_forward, left_name="original", right_name="supplied_tail")
    full_align = alignment_stats(steps, reconstructed_full_trace, left_name="original", right_name="reconstructed")

    closure_stats = closure_answer_stats(
        backend,
        question,
        reconstructed_full_trace,
        baseline_A_star=baseline_A_star,
        temperature=settings.temperature_reanswer,
        seed_base=run_seed + 1_750_000 + k * 1000 + depth * 10,
        n_samples=max(1, int(settings.bridge_loop_closure_samples)),
        prompt_version=settings.prompt_version,
        prompt_family=settings.prompt_family,
    )

    loop_components = {
        "middle_similarity_concat": concat_sim,
        "full_trace_similarity_concat": full_align.get("similarity_concat"),
        "answer_match_rate": closure_stats.get("match_rate"),
    }
    vals = [float(v) for v in loop_components.values() if v is not None]
    loop_closure_score = mean_or_none(vals)

    return {
        "tag": tag,
        "k": k,
        "depth": depth,
        "budget": k + depth,
        "approx_missing_steps": approx_missing,
        "original_middle_steps": original_middle,
        "reconstructed_middle_steps": middle_steps,
        "pairwise_alignment": pairwise,
        "middle_similarity_mean": mean_or_none(sims),
        "middle_similarity_concat": concat_sim,
        "supplied_tail_forward_steps": trimmed_tail_forward,
        "tail_overlap_trimmed_steps": overlap_steps,
        "tail_alignment": tail_align,
        "reconstructed_full_trace_steps": reconstructed_full_trace,
        "full_trace_alignment": full_align,
        "loop_closure_answer": closure_stats,
        "loop_closure_components": loop_components,
        "loop_closure_score": loop_closure_score,
    }

def summarize_irregular_curve(

    points: List[Dict[str, Any]],
    settings: ExperimentSettings,
    x_key: str,
    x_max: Optional[float] = None,
    label: str = "",
) -> Dict[str, Any]:
    xs = [pt[x_key] for pt in points]
    match_series = [float(pt.get("match_rate", 0.0)) for pt in points]
    smooth_match = moving_average(match_series, settings.smooth_window)

    conv_idx = find_converge_k(smooth_match, settings.tau, settings.converge_min_run)
    col_idx = find_collapse_k(smooth_match, settings.tau, settings.collapse_min_run)
    converge_x = xs[conv_idx] if conv_idx is not None and conv_idx < len(xs) else None
    collapse_x = xs[col_idx] if col_idx is not None and col_idx < len(xs) else None

    d = discrete_derivative(smooth_match)
    max_slope = max(d) if d else None
    max_slope_x = (xs[d.index(max_slope)] if d and max_slope is not None and d.index(max_slope) < len(xs) else None)

    plateau_mean = None
    if conv_idx is not None:
        tail = smooth_match[conv_idx:]
        plateau_mean = sum(tail) / max(1, len(tail))
    else:
        start = int(max(0, math.floor(0.8 * len(xs))))
        tail = smooth_match[start:] if smooth_match else []
        plateau_mean = sum(tail) / max(1, len(tail)) if tail else None

    denom = x_max if x_max is not None and x_max not in (0, None) else (max(xs) if xs else 0)
    conv_score = 0.0
    if converge_x is not None and denom:
        conv_score = 1.0 - min(float(converge_x), float(denom)) / float(denom)
        conv_score = max(0.0, min(1.0, conv_score))
    attractor_strength = conv_score * (plateau_mean if plateau_mean is not None else 0.0)
    auc_match = sum(smooth_match) / max(1, len(smooth_match)) if smooth_match else 0.0

    hysteresis = None
    if converge_x is not None and collapse_x is not None:
        try:
            hysteresis = float(collapse_x) - float(converge_x)
        except Exception:
            hysteresis = None

    return {
        "label": label,
        "smooth_match_rate": smooth_match,
        "converge_index": conv_idx,
        "collapse_index": col_idx,
        "converge_value": converge_x,
        "collapse_value": collapse_x,
        "stabilization_max_slope": max_slope,
        "stabilization_max_slope_value": max_slope_x,
        "plateau_mean": plateau_mean,
        "attractor_strength": attractor_strength,
        "auc_match": auc_match,
        "hysteresis": hysteresis,
    }

def compute_bridge_grid(
    backend: Backend,
    question: str,
    trace: Trace,
    latest_to_earlier_steps: List[str],
    settings: ExperimentSettings,
    run_seed: int,
    baseline_A_star: str,
    baseline_distribution: Dict[str, int],
    forward_curve: List[Dict[str, Any]],
    tail_curve_obj: Dict[str, Any],
    label: str,
) -> Dict[str, Any]:
    steps = trace.steps[: settings.max_steps]
    L = len(steps)
    D = len(latest_to_earlier_steps)
    ks = evenly_spaced_ints(L, settings.bridge_num_prefix_points)
    ds = evenly_spaced_ints(D, settings.bridge_num_tail_points)
    n_samples = settings.bridge_samples_per_cell if settings.bridge_samples_per_cell > 0 else settings.n_samples_per_k

    base_counter = Counter({str(k): int(v) for k, v in baseline_distribution.items()})
    base_support = list(base_counter.keys())
    prefix_match_by_k = {int(pt["k"]): float(pt.get("match_rate", 0.0)) for pt in (forward_curve or []) if "k" in pt}
    tail_curve_points = tail_curve_obj.get("curve", []) or []
    tail_match_by_d = {int(pt["k"]): float(pt.get("match_rate", 0.0)) for pt in tail_curve_points if "k" in pt}

    grid: List[Dict[str, Any]] = []
    valid_cells: List[Dict[str, Any]] = []

    for k in ks:
        for depth in ds:
            overlap_steps = max(0, (k + depth) - L)
            valid = True
            if (not settings.bridge_allow_overlap) and overlap_steps > 0:
                valid = False

            cell: Dict[str, Any] = {
                "k": int(k),
                "depth": int(depth),
                "budget": int(k + depth),
                "overlap_steps": int(overlap_steps),
                "valid": bool(valid),
            }

            prefix_only_match = float(prefix_match_by_k.get(int(k), 0.0))
            tail_only_match = float(tail_match_by_d.get(int(depth), 0.0))
            cell["prefix_only_match"] = prefix_only_match
            cell["tail_only_match"] = tail_only_match

            if not valid:
                cell["match_rate"] = None
                cell["entropy_bits"] = None
                cell["top_answer"] = ""
                cell["js_div_full"] = None
                cell["kl_div_full"] = None
                cell["synergy_over_max"] = None
                cell["synergy_over_mean"] = None
                grid.append(cell)
                continue

            answers_bd: List[str] = []
            for j in range(n_samples):
                answers_bd.append(
                    reanswer_from_bridge(
                        backend,
                        question,
                        steps,
                        int(k),
                        latest_to_earlier_steps,
                        int(depth),
                        temperature=settings.temperature_reanswer,
                        seed=run_seed + 1_200_000 + int(k) * 10_000 + int(depth) * 100 + j,
                        order=settings.rewind_order,
                        prompt_version=settings.prompt_version,
                        prompt_family=settings.prompt_family,
                    )
                )

            dist_bd = dist_from_answers(answers_bd)
            match = 0.0
            if baseline_A_star:
                match = dist_bd.get(baseline_A_star, 0) / max(1, sum(dist_bd.values()))
            ent = entropy_bits_from_counts(dist_bd)
            top = dist_bd.most_common(1)[0][0] if dist_bd else ""

            jsd = None
            kld = None
            if settings.compute_divergence:
                support = sorted(set(base_support) | set(dist_bd.keys()))
                p = _to_prob(dist_bd, support, alpha=settings.divergence_alpha)
                q = _to_prob(base_counter, support, alpha=settings.divergence_alpha)
                jsd = js_divergence(p, q)
                kld = kl_divergence(p, q)

            cell.update({
                "match_rate": match,
                "entropy_bits": ent,
                "top_answer": top,
                "js_div_full": jsd,
                "kl_div_full": kld,
                "synergy_over_max": match - max(prefix_only_match, tail_only_match),
                "synergy_over_mean": match - 0.5 * (prefix_only_match + tail_only_match),
            })
            grid.append(cell)
            valid_cells.append(cell)

    def choose_budget_rep(cells_for_budget: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
        if mode == "balanced":
            return sorted(
                cells_for_budget,
                key=lambda c: (
                    abs(int(c["k"]) - int(c["depth"])),
                    -float(c.get("match_rate", 0.0)),
                    -float(c.get("synergy_over_max", 0.0)),
                ),
            )[0]
        return sorted(
            cells_for_budget,
            key=lambda c: (
                -float(c.get("match_rate", 0.0)),
                -float(c.get("synergy_over_max", 0.0)),
                abs(int(c["k"]) - int(c["depth"])),
            ),
        )[0]

    def build_budget_curve(mode: str) -> Dict[str, Any]:
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for cell in valid_cells:
            groups.setdefault(int(cell["budget"]), []).append(cell)
        budgets = sorted(groups.keys())
        curve: List[Dict[str, Any]] = []
        for b in budgets:
            sel = dict(choose_budget_rep(groups[b], mode))
            curve.append({
                "budget": int(b),
                "k": int(sel["k"]),
                "depth": int(sel["depth"]),
                "match_rate": float(sel.get("match_rate", 0.0)),
                "entropy_bits": sel.get("entropy_bits"),
                "top_answer": sel.get("top_answer"),
                "js_div_full": sel.get("js_div_full"),
                "kl_div_full": sel.get("kl_div_full"),
                "synergy_over_max": sel.get("synergy_over_max"),
                "synergy_over_mean": sel.get("synergy_over_mean"),
            })
        stats = summarize_irregular_curve(curve, settings, x_key="budget", x_max=max(budgets) if budgets else None, label=f"{label}_{mode}_budget")
        stats.update({
            "mode": mode,
            "curve": curve,
            "budget_max": max(budgets) if budgets else 0,
            "converge_budget": stats.pop("converge_value"),
            "collapse_budget": stats.pop("collapse_value"),
            "stabilization_max_slope_budget": stats.pop("stabilization_max_slope_value"),
        })
        return stats

    budget_best_curve = build_budget_curve("best")
    budget_balanced_curve = build_budget_curve("balanced")

    best_cell = max(valid_cells, key=lambda c: float(c.get("match_rate", 0.0))) if valid_cells else None
    best_synergy_cell = max(valid_cells, key=lambda c: float(c.get("synergy_over_max", float("-inf")))) if valid_cells else None
    positive_synergy_fraction = None
    mean_synergy_over_max = None
    mean_synergy_over_mean = None
    if valid_cells:
        syn_max = [float(c.get("synergy_over_max", 0.0)) for c in valid_cells]
        syn_mean = [float(c.get("synergy_over_mean", 0.0)) for c in valid_cells]
        positive_synergy_fraction = sum(1 for x in syn_max if x > 0.0) / float(len(syn_max))
        mean_synergy_over_max = mean_or_none(syn_max)
        mean_synergy_over_mean = mean_or_none(syn_mean)

    examples: List[Dict[str, Any]] = []
    core_summary: Dict[str, Any] = {
        "best_core_candidate": None,
        "minimal_core_candidate": None,
        "core_examples_ranked": [],
        "core_step_prototypes": [],
    }
    if settings.bridge_reconstruct_middle and valid_cells:
        max_budget = max(int(c["budget"]) for c in valid_cells) if valid_cells else 0
        cell_lookup = {(int(c["k"]), int(c["depth"])): c for c in valid_cells}
        selected = choose_bridge_reconstruction_cells(valid_cells, best_cell, best_synergy_cell, settings)
        for tag, cell in selected:
            ex = reconstruct_bridge_middle(
                backend,
                question,
                trace,
                latest_to_earlier_steps,
                settings,
                int(cell["k"]),
                int(cell["depth"]),
                tag=tag,
                run_seed=run_seed,
                baseline_A_star=baseline_A_star,
            )
            cell_ref = cell_lookup.get((int(cell["k"]), int(cell["depth"])), cell)
            ex.update({
                "cell_match_rate": cell_ref.get("match_rate"),
                "cell_entropy_bits": cell_ref.get("entropy_bits"),
                "cell_synergy_over_max": cell_ref.get("synergy_over_max"),
                "cell_synergy_over_mean": cell_ref.get("synergy_over_mean"),
                "cell_prefix_only_match": cell_ref.get("prefix_only_match"),
                "cell_tail_only_match": cell_ref.get("tail_only_match"),
            })
            examples.append(ex)
            if len(examples) >= max(0, settings.bridge_middle_cases):
                break
        core_summary = summarize_bridge_core(examples, settings, label=label, max_budget=max_budget)

    return {
        "label": label,
        "tail_source": label,
        "tail_order": settings.rewind_order,
        "allow_overlap": settings.bridge_allow_overlap,
        "prefix_points": ks,
        "tail_points": ds,
        "cell_count": len(grid),
        "valid_cell_count": len(valid_cells),
        "grid": grid,
        "best_cell": best_cell,
        "best_synergy_cell": best_synergy_cell,
        "positive_synergy_fraction": positive_synergy_fraction,
        "mean_synergy_over_max": mean_synergy_over_max,
        "mean_synergy_over_mean": mean_synergy_over_mean,
        "budget_best_curve": budget_best_curve,
        "budget_balanced_curve": budget_balanced_curve,
        "middle_reconstruction_examples": examples,
        "best_core_candidate": core_summary.get("best_core_candidate"),
        "minimal_core_candidate": core_summary.get("minimal_core_candidate"),
        "core_examples_ranked": core_summary.get("core_examples_ranked"),
        "core_step_prototypes": core_summary.get("core_step_prototypes"),
    }

def compute_bridge_bundles(
    backend: Backend,
    question: str,
    trace: Trace,
    settings: ExperimentSettings,
    forward_result: Dict[str, Any],
    run_seed: int,
) -> Dict[str, Any]:
    baseline_A_star = str(forward_result.get("A_star", ""))
    baseline_distribution = dict(forward_result.get("baseline_distribution", {}) or {})
    forward_curve = forward_result.get("curve", []) or []

    bundles: Dict[str, Any] = {
        "bridge_enabled": True,
        "bridge_tail_source": settings.bridge_tail_source,
    }

    wanted = settings.bridge_tail_source
    want_recovered = wanted in ("recovered", "both")
    want_oracle = wanted in ("oracle", "both")

    recovered_bundle = None
    oracle_bundle = None

    if want_recovered:
        rewind_trace = forward_result.get("rewind_trace") or {}
        rewind_curve = forward_result.get("rewind_curve") or {}
        latest_to_earlier = rewind_trace.get("latest_to_earlier") or []
        if latest_to_earlier and isinstance(latest_to_earlier, list):
            recovered_bundle = compute_bridge_grid(
                backend,
                question,
                trace,
                [str(s) for s in latest_to_earlier],
                settings,
                run_seed=run_seed + 80_000,
                baseline_A_star=baseline_A_star,
                baseline_distribution=baseline_distribution,
                forward_curve=forward_curve,
                tail_curve_obj=rewind_curve if isinstance(rewind_curve, dict) else {},
                label="recovered",
            )
        else:
            bundles["bridge_recovered_error"] = "Recovered rewind trace not available."

    if want_oracle:
        oracle_curve = forward_result.get("oracle_tail_curve") or {}
        original_latest_to_earlier = list(reversed(trace.steps[: settings.max_steps]))
        oracle_bundle = compute_bridge_grid(
            backend,
            question,
            trace,
            original_latest_to_earlier,
            settings,
            run_seed=run_seed + 90_000,
            baseline_A_star=baseline_A_star,
            baseline_distribution=baseline_distribution,
            forward_curve=forward_curve,
            tail_curve_obj=oracle_curve if isinstance(oracle_curve, dict) else {},
            label="oracle",
        )

    bundles["bridge_recovered"] = recovered_bundle
    bundles["bridge_oracle"] = oracle_bundle

    if recovered_bundle is not None and oracle_bundle is not None:
        try:
            bundles["bridge_oracle_minus_recovered_best_auc"] = float((oracle_bundle.get("budget_best_curve") or {}).get("auc_match", 0.0)) - float((recovered_bundle.get("budget_best_curve") or {}).get("auc_match", 0.0))
        except Exception:
            bundles["bridge_oracle_minus_recovered_best_auc"] = None
        try:
            bundles["bridge_oracle_minus_recovered_best_synergy"] = float(((oracle_bundle.get("best_synergy_cell") or {}).get("synergy_over_max", 0.0) or 0.0)) - float(((recovered_bundle.get("best_synergy_cell") or {}).get("synergy_over_max", 0.0) or 0.0))
        except Exception:
            bundles["bridge_oracle_minus_recovered_best_synergy"] = None

    return bundles



def compute_curve(
    backend: Backend,
    question: str,
    trace: Trace,
    settings: ExperimentSettings,
    run_seed: int
) -> Dict[str, Any]:
    steps = trace.steps[: settings.max_steps]
    L = len(steps)
    save_raw = bool(settings.save_raw_generations)

    # Baseline: full prefix answers distribution
    baseline_answers: List[str] = []
    for i in range(settings.n_samples_baseline):
        baseline_answers.append(
            reanswer_from_prefix(
                backend, question, steps, L,
                temperature=settings.temperature_reanswer,
                seed=run_seed + 10_000 + i,
                prompt_version=settings.prompt_version,
                prompt_family=settings.prompt_family,
            )
        )
    A_star, baseline_conf = majority_vote(baseline_answers)
    baseline_dist = dist_from_answers(baseline_answers)
    baseline_entropy = entropy_bits_from_counts(baseline_dist)

    # baseline gating
    unstable_reasons: List[str] = []
    if settings.baseline_min_conf is not None and baseline_conf < settings.baseline_min_conf:
        unstable_reasons.append(f"baseline_conf({baseline_conf:.3f}) < min_conf({settings.baseline_min_conf})")
    if settings.baseline_max_entropy is not None and baseline_entropy > settings.baseline_max_entropy:
        unstable_reasons.append(f"baseline_entropy({baseline_entropy:.3f}) > max_entropy({settings.baseline_max_entropy})")

    is_baseline_stable = (len(unstable_reasons) == 0)

    # Curve points
    curve: List[CurvePoint] = []
    curve_raw_answers: List[Dict[str, Any]] = []
    # Support for divergence: union of baseline + current
    base_support = list(baseline_dist.keys())

    for k in range(0, L + 1):
        answers_k: List[str] = []
        for j in range(settings.n_samples_per_k):
            answers_k.append(
                reanswer_from_prefix(
                    backend, question, steps, k,
                    temperature=settings.temperature_reanswer,
                    seed=run_seed + 200_000 + k * 1000 + j,
                    prompt_version=settings.prompt_version,
                    prompt_family=settings.prompt_family,
                )
            )
        dist_k = dist_from_answers(answers_k)
        if save_raw:
            curve_raw_answers.append({"k": k, "answers": answers_k[:]})
        match = 0.0
        if A_star:
            match = sum(dist_k.get(A_star, 0) for _ in [0])  # use count directly
            match = dist_k.get(A_star, 0) / max(1, sum(dist_k.values()))

        top = dist_k.most_common(1)[0][0] if dist_k else ""
        ent = entropy_bits_from_counts(dist_k)

        jsd = None
        kld = None
        if settings.compute_divergence:
            support = sorted(set(base_support) | set(dist_k.keys()))
            p = _to_prob(dist_k, support, alpha=settings.divergence_alpha)
            q = _to_prob(baseline_dist, support, alpha=settings.divergence_alpha)
            jsd = js_divergence(p, q)
            kld = kl_divergence(p, q)
        curve.append(CurvePoint(k=k, match_rate=match, entropy_bits=ent, top_answer=top, js_div_full=jsd, kl_div_full=kld))

    # Smoothing
    match_series = [pt.match_rate for pt in curve]
    smooth_match = moving_average(match_series, settings.smooth_window)

    # Converge / collapse with min-run on smoothed series
    converge_k = find_converge_k(smooth_match, settings.tau, settings.converge_min_run)
    collapse_k = find_collapse_k(smooth_match, settings.tau, settings.collapse_min_run)

    # Derivatives (stabilization speed)
    d = discrete_derivative(smooth_match)
    max_slope = max(d) if d else None
    max_slope_k = (d.index(max_slope) if (d and max_slope is not None) else None)

    # Plateau quality after convergence
    plateau_mean = None
    if converge_k is not None:
        tail = smooth_match[converge_k:]
        plateau_mean = sum(tail) / max(1, len(tail))
    else:
        # use last 20% as a proxy
        start = int(max(0, math.floor(0.8 * L)))
        tail = smooth_match[start:] if smooth_match else []
        plateau_mean = sum(tail) / max(1, len(tail)) if tail else None

    # Attractor strength: early convergence × plateau
    conv_score = 0.0
    if converge_k is not None and L > 0:
        conv_score = 1.0 - min(converge_k, L) / float(L)
        conv_score = max(0.0, min(1.0, conv_score))
    plateau = plateau_mean if plateau_mean is not None else 0.0
    attractor_strength = conv_score * plateau

    # AUC of match_rate (smoothed) as an overall stability mass
    auc_match = sum(smooth_match) / max(1, len(smooth_match)) if smooth_match else 0.0

    # Hysteresis: collapse - converge (None-safe)
    hysteresis = None
    if converge_k is not None and collapse_k is not None:
        hysteresis = collapse_k - converge_k

    out = {
        "L": L,
        "A_star": A_star,
        "baseline_confidence": baseline_conf,
        "baseline_entropy_bits": baseline_entropy,
        "baseline_distribution": dict(baseline_dist),
        "is_baseline_stable": is_baseline_stable,
        "baseline_unstable_reasons": unstable_reasons,
        "tau": settings.tau,
        "smooth_window": settings.smooth_window,
        "converge_min_run": settings.converge_min_run,
        "collapse_min_run": settings.collapse_min_run,
        "converge_k": converge_k,
        "collapse_k": collapse_k,
        "hysteresis": hysteresis,
        "stabilization_max_slope": max_slope,
        "stabilization_max_slope_k": max_slope_k,
        "plateau_mean": plateau_mean,
        "attractor_strength": attractor_strength,
        "auc_match": auc_match,
        "curve": [dataclasses.asdict(pt) for pt in curve],
        "smooth_match_rate": smooth_match,
    }
    if save_raw:
        out["baseline_raw_answers"] = baseline_answers[:]
        out["forward_raw_answers_by_k"] = curve_raw_answers
    return out


def permute_steps(steps: List[str], seed: int) -> List[str]:
    rng = random.Random(seed)
    idx = list(range(len(steps)))
    rng.shuffle(idx)
    return [steps[i] for i in idx]

def run_permutation_tests(
    backend: Backend,
    question: str,
    trace: Trace,
    settings: ExperimentSettings,
    run_seed: int
) -> List[Dict[str, Any]]:
    """
    Optional: shuffle steps and recompute a reduced set of metrics.
    """
    if settings.permutation_tests <= 0:
        return []
    results: List[Dict[str, Any]] = []
    for t in range(settings.permutation_tests):
        perm_steps = permute_steps(trace.steps, seed=settings.permutation_seed + t)
        perm_trace = Trace(steps=perm_steps, answer=trace.answer)
        res = compute_curve(backend, question, perm_trace, settings, run_seed=run_seed + 9_000_000 + t * 10_000)
        results.append({
            "perm_id": t,
            "perm_seed": settings.permutation_seed + t,
            "auc_match": res.get("auc_match"),
            "converge_k": res.get("converge_k"),
            "collapse_k": res.get("collapse_k"),
            "attractor_strength": res.get("attractor_strength"),
            "baseline_confidence": res.get("baseline_confidence"),
            "baseline_entropy_bits": res.get("baseline_entropy_bits"),
        })
    return results




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

# ----------------------------- config & IO -----------------------------

def load_config(path: str) -> Dict[str, Any]:
    text = read_text(path)
    if path.endswith(".json"):
        return json.loads(text)
    if path.endswith(".yaml") or path.endswith(".yml"):
        if yaml is None:
            raise RuntimeError("pyyaml is required to load YAML config. Install with: pip install pyyaml")
        return yaml.safe_load(text)
    # auto
    try:
        return json.loads(text)
    except Exception:
        if yaml is None:
            raise RuntimeError("Config is not valid JSON and pyyaml is not installed.")
        return yaml.safe_load(text)

def parse_backends(cfg: Dict[str, Any]) -> List[Backend]:
    backends_cfg = cfg.get("backends", [])
    out: List[Backend] = []
    for b in backends_cfg:
        btype = b.get("type")
        name = b.get("name") or b.get("alias") or "backend"
        if btype == "hf_local":
            out.append(HFLocalBackend(
                name=name,
                model_path=b["model"],
                tokenizer_path=b.get("tokenizer"),
                device=b.get("device", "cpu"),
                hf_backend=b.get("hf_backend", "auto"),
                torch_dtype=b.get("torch_dtype", b.get("dtype", "auto")),
                device_map=b.get("device_map"),
                trust_remote_code=bool(b.get("trust_remote_code", False)),
                use_fast_tokenizer=b.get("use_fast_tokenizer"),
                max_new_tokens=int(b.get("max_new_tokens", 256)),
                top_p=float(b.get("top_p", 1.0)),
                top_k=int(b.get("top_k", 50)),
                repetition_penalty=float(b.get("repetition_penalty", 1.0)),
            ))
        else:
            # HTTP assumed
            api_key = b.get("api_key", "")
            api_key_env = b.get("api_key_env")
            if api_key_env:
                api_key = os.environ.get(api_key_env, api_key)
            out.append(HTTPBackend(
                name=name,
                base_url=b["base_url"],
                api_key=api_key,
                model=b["model"],
                timeout_s=int(cfg.get("api", {}).get("timeout_s", b.get("timeout_s", 120))),
            ))
    return out

def parse_questions(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    qs: List[Dict[str, str]] = []
    if "questions" in cfg and isinstance(cfg["questions"], list):
        for q in cfg["questions"]:
            qs.append({"id": str(q.get("id", f"q{len(qs)+1}")), "question": str(q.get("question", ""))})
        return qs
    # allow questions.jsonl path
    qpath = cfg.get("questions_path")
    if qpath:
        rows = []
        with open(qpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        for r in rows:
            qs.append({"id": str(r.get("id", f"q{len(qs)+1}")), "question": str(r.get("question", ""))})
        return qs
    raise RuntimeError("No questions found. Provide 'questions' list or 'questions_path'.")

def parse_experiment_settings(cfg: Dict[str, Any]) -> ExperimentSettings:
    e = cfg.get("experiment", {}) or {}
    s = ExperimentSettings(
        trace_mode=str(e.get("trace_mode", "masked")),
        trace_source=str(e.get("trace_source", "per_backend")),
        trace_format=str(e.get("trace_format", "auto")),
        stream_output=bool(e.get("stream_output", False)),
        temperature_trace=float(e.get("temperature_trace", 0.0)),
        process_reward_mode=str(e.get("process_reward_mode", "off")),
        trace_candidates=int(e.get("trace_candidates", 1)),
        trace_candidate_temperature=float(e.get("trace_candidate_temperature", 0.4)),
        temperature_reanswer=float(e.get("temperature_reanswer", 0.8)),
        n_samples_baseline=int(e.get("n_samples_baseline", 9)),
        n_samples_per_k=int(e.get("n_samples_per_k", 9)),
        tau=float(e.get("tau", 0.8)),
        max_steps=int(e.get("max_steps", 40)),
        prompt_version=str(e.get("prompt_version", "v1")),
        prompt_family=str(e.get("prompt_family", "")),

        baseline_min_conf=(float(e["baseline_min_conf"]) if "baseline_min_conf" in e and e["baseline_min_conf"] is not None else None),
        baseline_max_entropy=(float(e["baseline_max_entropy"]) if "baseline_max_entropy" in e and e["baseline_max_entropy"] is not None else None),
        skip_unstable_baseline=bool(e.get("skip_unstable_baseline", False)),

        smooth_window=int(e.get("smooth_window", 1)),
        converge_min_run=int(e.get("converge_min_run", 1)),
        collapse_min_run=int(e.get("collapse_min_run", 1)),

        compute_divergence=bool(e.get("compute_divergence", True)),
        divergence_alpha=float(e.get("divergence_alpha", 1e-6)),

        permutation_tests=int(e.get("permutation_tests", 0)),
        permutation_seed=int(e.get("permutation_seed", 1234)),

        enable_rewind=bool(e.get("enable_rewind", True)),
        rewind_step_temperature=float(e.get("rewind_step_temperature", 0.4)),
        rewind_samples_per_depth=int(e.get("rewind_samples_per_depth", 0)),
        rewind_order=str(e.get("rewind_order", "reverse")),
        compute_oracle_tail=bool(e.get("compute_oracle_tail", True)),
        rewind_process_reward_mode=str(e.get("rewind_process_reward_mode", "off")),
        rewind_step_candidates=int(e.get("rewind_step_candidates", 1)),
        rewind_novelty_max_similarity=float(e.get("rewind_novelty_max_similarity", 0.82)),
        rewind_novelty_retries=int(e.get("rewind_novelty_retries", 3)),
        rewind_escape_temperature=float(e.get("rewind_escape_temperature", 0.6)),
        rewind_step_top_p=float(e.get("rewind_step_top_p", 0.9)),
        rewind_step_repetition_penalty=float(e.get("rewind_step_repetition_penalty", 1.05)),

        enable_bridge=bool(e.get("enable_bridge", True)),
        bridge_tail_source=str(e.get("bridge_tail_source", "recovered")),
        bridge_samples_per_cell=int(e.get("bridge_samples_per_cell", 0)),
        bridge_num_prefix_points=int(e.get("bridge_num_prefix_points", 9)),
        bridge_num_tail_points=int(e.get("bridge_num_tail_points", 9)),
        bridge_allow_overlap=bool(e.get("bridge_allow_overlap", False)),
        bridge_middle_temperature=float(e.get("bridge_middle_temperature", 0.0)),
        bridge_reconstruct_middle=bool(e.get("bridge_reconstruct_middle", True)),
        bridge_middle_cases=int(e.get("bridge_middle_cases", 5)),
        bridge_loop_closure_samples=int(e.get("bridge_loop_closure_samples", 3)),
        bridge_core_cluster_similarity=float(e.get("bridge_core_cluster_similarity", 0.72)),
        bridge_core_min_loop_closure=float(e.get("bridge_core_min_loop_closure", 0.75)),
        bridge_core_min_match=float(e.get("bridge_core_min_match", 0.75)),
        bridge_core_min_synergy=float(e.get("bridge_core_min_synergy", 0.0)),
        bridge_core_top_prototypes=int(e.get("bridge_core_top_prototypes", 6)),
        save_raw_generations=bool(e.get("save_raw_generations", False)),
        save_generation_log_jsonl=bool(e.get("save_generation_log_jsonl", False)),
    )
    # clamp
    s.smooth_window = max(1, s.smooth_window)
    s.converge_min_run = max(1, s.converge_min_run)
    s.collapse_min_run = max(1, s.collapse_min_run)
    s.prompt_version, s.prompt_family = normalize_prompt_selector(s.prompt_version, s.prompt_family)
    s.trace_format = normalize_trace_output_format(s.trace_format)
    s.process_reward_mode = s.process_reward_mode if s.process_reward_mode in ("off", "lite") else "off"
    s.trace_candidates = max(1, s.trace_candidates)
    s.trace_candidate_temperature = max(0.0, s.trace_candidate_temperature)
    s.rewind_order = s.rewind_order if s.rewind_order in ("reverse", "forward") else "reverse"
    s.rewind_process_reward_mode = s.rewind_process_reward_mode if s.rewind_process_reward_mode in ("off", "lite") else "off"
    s.rewind_step_candidates = max(1, s.rewind_step_candidates)
    s.rewind_samples_per_depth = max(0, s.rewind_samples_per_depth)
    s.rewind_novelty_max_similarity = max(0.0, min(1.0, s.rewind_novelty_max_similarity))
    s.rewind_novelty_retries = max(0, s.rewind_novelty_retries)
    s.rewind_escape_temperature = max(0.0, s.rewind_escape_temperature)
    s.rewind_step_top_p = max(0.0, min(1.0, s.rewind_step_top_p))
    s.rewind_step_repetition_penalty = max(1.0, s.rewind_step_repetition_penalty)
    s.bridge_tail_source = s.bridge_tail_source if s.bridge_tail_source in ("recovered", "oracle", "both") else "recovered"
    s.bridge_samples_per_cell = max(0, s.bridge_samples_per_cell)
    s.bridge_num_prefix_points = max(0, s.bridge_num_prefix_points)
    s.bridge_num_tail_points = max(0, s.bridge_num_tail_points)
    s.bridge_middle_cases = max(0, s.bridge_middle_cases)
    s.bridge_loop_closure_samples = max(1, s.bridge_loop_closure_samples)
    s.bridge_core_cluster_similarity = max(0.0, min(1.0, s.bridge_core_cluster_similarity))
    s.bridge_core_min_loop_closure = max(0.0, min(1.0, s.bridge_core_min_loop_closure))
    s.bridge_core_min_match = max(0.0, min(1.0, s.bridge_core_min_match))
    s.bridge_core_top_prototypes = max(0, s.bridge_core_top_prototypes)
    return s


# ----------------------------- plotting -----------------------------

def plot_run(run_dir: str) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib")
    run_dir = os.path.abspath(run_dir)
    plots_dir = ensure_dir(os.path.join(run_dir, "plots"))

    # Find result json files
    files = [f for f in os.listdir(run_dir) if f.endswith(".json") and "__" in f]
    for fn in files:
        path = os.path.join(run_dir, fn)
        obj = read_json(path)
        curve = obj.get("curve", [])
        smooth = obj.get("smooth_match_rate", None)
        if not curve:
            continue
        ks = [pt["k"] for pt in curve]
        match = [pt["match_rate"] for pt in curve]
        ent = [pt["entropy_bits"] for pt in curve]
        jsd = [pt.get("js_div_full") for pt in curve]
        kld = [pt.get("kl_div_full") for pt in curve]

        base = fn.replace(".json", "")
        # match
        plt.figure()
        plt.plot(ks, match, marker="o")
        if smooth is not None:
            plt.plot(ks, smooth, marker="x")
        plt.xlabel("k (prefix length)")
        plt.ylabel("match_rate")
        plt.title(f"{base} match_rate")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{base}__match.png"), dpi=160)
        plt.close()

        # entropy
        plt.figure()
        plt.plot(ks, ent, marker="o")
        plt.xlabel("k (prefix length)")
        plt.ylabel("entropy_bits")
        plt.title(f"{base} entropy_bits")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{base}__entropy.png"), dpi=160)
        plt.close()

        # divergence
        if any(v is not None for v in jsd):
            plt.figure()
            plt.plot(ks, [v if v is not None else float("nan") for v in jsd], marker="o")
            plt.xlabel("k")
            plt.ylabel("JS divergence (nats) vs full")
            plt.title(f"{base} JS divergence")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{base}__jsd.png"), dpi=160)
            plt.close()
        if any(v is not None for v in kld):
            plt.figure()
            plt.plot(ks, [v if v is not None else float("nan") for v in kld], marker="o")
            plt.xlabel("k")
            plt.ylabel("KL divergence (nats) vs full")
            plt.title(f"{base} KL divergence")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{base}__kld.png"), dpi=160)
            plt.close()

        # rewind/oracle compare
        rewind_obj = obj.get("rewind_curve") or {}
        oracle_obj = obj.get("oracle_tail_curve") or {}
        rewind_curve = rewind_obj.get("curve", []) or []
        oracle_curve = oracle_obj.get("curve", []) or []
        if rewind_curve or oracle_curve:
            plt.figure()
            plt.plot(ks, match, marker="o", label="forward_prefix")
            if smooth is not None:
                plt.plot(ks, smooth, marker="x", label="forward_prefix_smooth")
            if rewind_curve:
                rks = [pt["k"] for pt in rewind_curve]
                rmatch = [pt["match_rate"] for pt in rewind_curve]
                plt.plot(rks, rmatch, marker="s", label="rewind_tail")
                rsmooth = rewind_obj.get("smooth_match_rate")
                if rsmooth is not None:
                    plt.plot(rks, rsmooth, marker="^", label="rewind_tail_smooth")
            if oracle_curve:
                oks = [pt["k"] for pt in oracle_curve]
                omatch = [pt["match_rate"] for pt in oracle_curve]
                plt.plot(oks, omatch, marker="d", label="oracle_tail")
                osmooth = oracle_obj.get("smooth_match_rate")
                if osmooth is not None:
                    plt.plot(oks, osmooth, marker="v", label="oracle_tail_smooth")
            plt.xlabel("depth / k")
            plt.ylabel("match_rate")
            plt.title(f"{base} forward vs rewind")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{base}__rewind_compare.png"), dpi=160)
            plt.close()

        trace_vs_rewind = obj.get("trace_vs_rewind")
        if not isinstance(trace_vs_rewind, dict):
            trace_obj = obj.get("trace") or {}
            rewind_trace_obj = obj.get("rewind_trace") or {}
            if isinstance(trace_obj, dict) and isinstance(rewind_trace_obj, dict) and (
                rewind_trace_obj.get("forward_order") or rewind_trace_obj.get("latest_to_earlier")
            ):
                trace_vs_rewind = compare_forward_and_rewind_steps(trace_obj.get("steps") or [], rewind_trace_obj)
        if isinstance(trace_vs_rewind, dict):
            pairs = trace_vs_rewind.get("pairwise_alignment") or []
            if pairs:
                idx = [int(p.get("index", i)) for i, p in enumerate(pairs)]
                sims = [float(p.get("similarity", 0.0) or 0.0) for p in pairs]
                plt.figure()
                plt.plot(idx, sims, marker="o")
                plt.ylim(0.0, 1.0)
                plt.xlabel("step index")
                plt.ylabel("similarity")
                plt.title(f"{base} trace vs rewind similarity")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{base}__trace_vs_rewind_similarity.png"), dpi=160)
                plt.close()

        trace_axes = obj.get("trace_axes")
        if isinstance(trace_axes, dict):
            rows = trace_axes.get("rows") or []
            if rows:
                idx = [int(row.get("index", i)) for i, row in enumerate(rows)]
                base_prm = [float(row["base_prm_similarity"]) if row.get("base_prm_similarity") is not None else float("nan") for row in rows]
                base_rewind = [float(row["base_rewind_similarity"]) if row.get("base_rewind_similarity") is not None else float("nan") for row in rows]
                prm_rewind = [float(row["prm_rewind_similarity"]) if row.get("prm_rewind_similarity") is not None else float("nan") for row in rows]
                process_reward = obj.get("trace_process_reward") or {}
                base_score_points: List[Tuple[int, float]] = []
                prm_score_points: List[Tuple[int, float]] = []
                if isinstance(process_reward, dict):
                    base_idx = int(process_reward.get("base_candidate_index", 0) or 0)
                    selected_idx = int(process_reward.get("selected_candidate_index", 0) or 0)
                    for cand in process_reward.get("candidates", []) or []:
                        if not isinstance(cand, dict) or cand.get("status") != "ok":
                            continue
                        step_scores = ((cand.get("process_reward") or {}).get("step_scores") if isinstance(cand.get("process_reward"), dict) else None) or []
                        points = [(i, float(row.get("score", 0.0) or 0.0)) for i, row in enumerate(step_scores)]
                        if int(cand.get("candidate_index", -1) or -1) == base_idx:
                            base_score_points = points
                        if int(cand.get("candidate_index", -1) or -1) == selected_idx:
                            prm_score_points = points
                plt.figure()
                plt.plot(idx, base_prm, marker="o", label="base_vs_prm")
                plt.plot(idx, base_rewind, marker="s", label="base_vs_rewind")
                plt.plot(idx, prm_rewind, marker="^", label="prm_vs_rewind")
                if base_score_points:
                    plt.plot([x for x, _ in base_score_points], [y for _, y in base_score_points], linestyle="--", marker=".", alpha=0.8, label="base_step_score")
                if prm_score_points:
                    plt.plot([x for x, _ in prm_score_points], [y for _, y in prm_score_points], linestyle="--", marker="x", alpha=0.8, label="prm_step_score")
                plt.ylim(0.0, 1.0)
                plt.xlabel("step index")
                plt.ylabel("similarity")
                plt.title(f"{base} trace axes")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{base}__trace_axes_similarity.png"), dpi=160)
                plt.close()

        rewind_axes = obj.get("rewind_axes")
        if isinstance(rewind_axes, dict):
            rows = rewind_axes.get("rows") or []
            if rows:
                idx = [int(row.get("index", i)) for i, row in enumerate(rows)]
                sims = [float(row["rewind_base_prm_similarity"]) if row.get("rewind_base_prm_similarity") is not None else float("nan") for row in rows]
                rewind_trace_obj = obj.get("rewind_trace") or {}
                base_rewind_obj = (rewind_trace_obj.get("base_rewind") or {}) if isinstance(rewind_trace_obj, dict) else {}
                base_scores = [float(row.get("score", 0.0) or 0.0) for row in (base_rewind_obj.get("process_reward_step_scores") or [])]
                prm_scores = [float(row.get("score", 0.0) or 0.0) for row in (rewind_trace_obj.get("process_reward_step_scores") or [])] if isinstance(rewind_trace_obj, dict) else []
                plt.figure()
                plt.plot(idx, sims, marker="o", label="rewind_base_vs_prm")
                if base_scores:
                    plt.plot(idx[:len(base_scores)], base_scores, linestyle="--", marker=".", alpha=0.8, label="rewind_base_step_score")
                if prm_scores:
                    plt.plot(idx[:len(prm_scores)], prm_scores, linestyle="--", marker="x", alpha=0.8, label="rewind_prm_step_score")
                plt.ylim(0.0, 1.0)
                plt.xlabel("rewind step index")
                plt.ylabel("similarity / score")
                plt.title(f"{base} rewind axes")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{base}__rewind_axes_similarity.png"), dpi=160)
                plt.close()

        process_reward = obj.get("trace_process_reward")
        if isinstance(process_reward, dict):
            candidates = [c for c in (process_reward.get("candidates") or []) if isinstance(c, dict) and c.get("status") == "ok"]
            if candidates:
                xs = [int(c.get("candidate_index", i) or i) for i, c in enumerate(candidates)]
                ys = [float(((c.get("process_reward") or {}).get("overall_score")) or 0.0) for c in candidates]
                selected_idx = int(process_reward.get("selected_candidate_index", -1) or -1)
                colors = ["tab:orange" if x == selected_idx else "tab:blue" for x in xs]
                plt.figure()
                plt.bar(xs, ys, color=colors)
                plt.xlabel("trace candidate")
                plt.ylabel("PRM-lite score")
                plt.title(f"{base} trace candidate scores")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{base}__trace_candidate_scores.png"), dpi=160)
                plt.close()

        rewind_trace_obj = obj.get("rewind_trace") or {}
        novelty_curve = rewind_trace_obj.get("novelty_curve") or []
        if not novelty_curve and (rewind_trace_obj.get("latest_to_earlier") or []):
            recovered = [str(s).strip() for s in rewind_trace_obj.get("latest_to_earlier", []) if str(s).strip()]
            tmp_rows: List[Dict[str, Any]] = []
            prior: List[str] = []
            threshold = float(rewind_trace_obj.get("novelty_threshold", 0.82) or 0.82)
            for depth, step in enumerate(recovered, start=1):
                info = rewind_step_novelty_info(step, prior, threshold)
                tmp_rows.append({
                    "depth": depth,
                    "step": step,
                    **info,
                })
                prior.append(step)
            novelty_curve = summarize_rewind_novelty(tmp_rows, threshold=threshold).get("novelty_curve") or tmp_rows
        if novelty_curve:
            depths = [int(row.get("depth", i + 1) or (i + 1)) for i, row in enumerate(novelty_curve)]
            novelty = [float(row.get("novelty", 0.0) or 0.0) for row in novelty_curve]
            max_sim = [float(row.get("max_similarity_to_previous", 0.0) or 0.0) for row in novelty_curve]
            rewind_scores = [float(row.get("score", 0.0) or 0.0) for row in (rewind_trace_obj.get("process_reward_step_scores") or [])]
            threshold = float(rewind_trace_obj.get("novelty_threshold", 0.82) or 0.82)
            fixed_depth = rewind_trace_obj.get("fixed_point_depth")
            plt.figure()
            plt.plot(depths, novelty, marker="o", label="novelty")
            plt.plot(depths, max_sim, marker="s", label="max_similarity_to_previous")
            if rewind_scores:
                plt.plot(depths[:len(rewind_scores)], rewind_scores, marker="^", linestyle="--", label="rewind_step_score")
            plt.axhline(1.0 - threshold, linestyle="--", color="tab:orange", label="novelty_threshold")
            if fixed_depth is not None:
                plt.axvline(int(fixed_depth), linestyle=":", color="tab:red", label="fixed_point_depth")
            plt.ylim(0.0, 1.05)
            plt.xlabel("rewind depth")
            plt.ylabel("score")
            plt.title(f"{base} rewind novelty")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{base}__rewind_novelty.png"), dpi=160)
            plt.close()



        # bridge-CoT plots
        for bridge_key in ["bridge_recovered", "bridge_oracle"]:
            bridge_obj = obj.get(bridge_key) or {}
            grid = bridge_obj.get("grid", []) or []
            if grid:
                valid_cells = [c for c in grid if c.get("valid")]
                if valid_cells:
                    xs = sorted(set(int(c["k"]) for c in valid_cells))
                    ys = sorted(set(int(c["depth"]) for c in valid_cells))
                    match_mat = [[float("nan") for _ in xs] for _ in ys]
                    syn_mat = [[float("nan") for _ in xs] for _ in ys]
                    x_index = {x: i for i, x in enumerate(xs)}
                    y_index = {y: i for i, y in enumerate(ys)}
                    for cell in valid_cells:
                        xi = x_index[int(cell["k"])]
                        yi = y_index[int(cell["depth"])]
                        match_mat[yi][xi] = float(cell.get("match_rate", float("nan")))
                        syn = cell.get("synergy_over_max")
                        syn_mat[yi][xi] = float(syn) if syn is not None else float("nan")

                    plt.figure()
                    plt.imshow(match_mat, origin="lower", aspect="auto")
                    plt.colorbar(label="match_rate")
                    plt.xticks(range(len(xs)), xs)
                    plt.yticks(range(len(ys)), ys)
                    plt.xlabel("prefix k")
                    plt.ylabel("rewind depth")
                    plt.title(f"{base} {bridge_key} bridge heatmap")
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f"{base}__{bridge_key}__heatmap_match.png"), dpi=160)
                    plt.close()

                    plt.figure()
                    plt.imshow(syn_mat, origin="lower", aspect="auto")
                    plt.colorbar(label="synergy_over_max")
                    plt.xticks(range(len(xs)), xs)
                    plt.yticks(range(len(ys)), ys)
                    plt.xlabel("prefix k")
                    plt.ylabel("rewind depth")
                    plt.title(f"{base} {bridge_key} synergy heatmap")
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f"{base}__{bridge_key}__heatmap_synergy.png"), dpi=160)
                    plt.close()

            best_curve = (bridge_obj.get("budget_best_curve") or {}).get("curve", []) or []
            bal_curve = (bridge_obj.get("budget_balanced_curve") or {}).get("curve", []) or []
            if best_curve or bal_curve:
                plt.figure()
                if best_curve:
                    bxs = [pt["budget"] for pt in best_curve]
                    bys = [pt["match_rate"] for pt in best_curve]
                    plt.plot(bxs, bys, marker="o", label=f"{bridge_key}_best")
                    bsmooth = (bridge_obj.get("budget_best_curve") or {}).get("smooth_match_rate")
                    if bsmooth is not None:
                        plt.plot(bxs, bsmooth, marker="x", label=f"{bridge_key}_best_smooth")
                if bal_curve:
                    bxs2 = [pt["budget"] for pt in bal_curve]
                    bys2 = [pt["match_rate"] for pt in bal_curve]
                    plt.plot(bxs2, bys2, marker="s", label=f"{bridge_key}_balanced")
                    bsmooth2 = (bridge_obj.get("budget_balanced_curve") or {}).get("smooth_match_rate")
                    if bsmooth2 is not None:
                        plt.plot(bxs2, bsmooth2, marker="^", label=f"{bridge_key}_balanced_smooth")
                plt.xlabel("bridge evidence budget (k + depth)")
                plt.ylabel("match_rate")
                plt.title(f"{base} {bridge_key} budget curves")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{base}__{bridge_key}__budget_curves.png"), dpi=160)
                plt.close()

            examples = bridge_obj.get("middle_reconstruction_examples", []) or []
            if examples:
                tags = [str(ex.get("tag", f"case_{i+1}")) for i, ex in enumerate(examples)]
                xs = list(range(len(examples)))
                lc = [float(ex.get("loop_closure_score", float("nan")) or float("nan")) for ex in examples]
                cs = [float(ex.get("core_score", float("nan")) or float("nan")) for ex in examples]
                plt.figure()
                plt.plot(xs, lc, marker="o", label="loop_closure_score")
                plt.plot(xs, cs, marker="s", label="core_score")
                plt.xticks(xs, tags, rotation=45, ha="right")
                plt.ylabel("score")
                plt.title(f"{base} {bridge_key} middle loop-closure")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{base}__{bridge_key}__middle_loop_closure.png"), dpi=160)
                plt.close()

            prototypes = bridge_obj.get("core_step_prototypes", []) or []
            if prototypes:
                top = prototypes[: min(6, len(prototypes))]
                labels = [str(p.get("representative_step", ""))[:40] for p in top]
                weights = [float(p.get("weighted_support", 0.0) or 0.0) for p in top]
                xs = list(range(len(top)))
                plt.figure()
                plt.bar(xs, weights)
                plt.xticks(xs, labels, rotation=45, ha="right")
                plt.ylabel("weighted_support")
                plt.title(f"{base} {bridge_key} core prototypes")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{base}__{bridge_key}__core_prototypes.png"), dpi=160)
                plt.close()

    print(f"[plot] saved plots to: {plots_dir}")


# ----------------------------- runners -----------------------------

def run_one_backend_one_question(
    backend: Backend,
    qid: str,
    question: str,
    settings: ExperimentSettings,
    out_dir: str,
    shared_trace: Optional[Trace] = None,
    trace_file: Optional[str] = None,
    run_seed: int = 42
) -> Dict[str, Any]:
    t_total = time.perf_counter()
    stage_timings: Dict[str, float] = {}
    generation_logger: Optional[GenerationLogger] = None
    active_backend = backend
    if settings.save_generation_log_jsonl:
        generation_logger = GenerationLogger(
            backend_name=getattr(backend, "name", "backend"),
            question_id=qid,
            question=question,
            prompt_version=settings.prompt_version,
            prompt_family=settings.prompt_family,
        )
        active_backend = LoggedBackend(backend, generation_logger)

    # trace source
    trace: Trace
    base_trace: Optional[Trace] = None
    process_reward: Optional[Dict[str, Any]] = None
    if trace_file:
        obj = read_json(trace_file)
        raw_steps = obj.get("step_records") if isinstance(obj.get("step_records"), list) else obj.get("steps", [])
        step_records = normalize_trace_step_records(raw_steps)
        trace = Trace(
            steps=[record["text"] for record in step_records],
            answer=str(obj.get("answer", "")),
            step_records=step_records,
        )
    elif shared_trace is not None:
        trace = shared_trace
        process_reward = dict((trace.metadata or {}).get("process_reward") or {}) if isinstance((trace.metadata or {}).get("process_reward"), dict) else None
    else:
        t0 = time.perf_counter()
        trace_bundle = build_trace_bundle(active_backend, question, settings, seed=run_seed + 123)
        stage_timings["trace_build_s"] = time.perf_counter() - t0
        trace = trace_bundle["selected_trace"]
        base_trace = trace_bundle.get("base_trace")
        process_reward = trace_bundle.get("process_reward")

    if base_trace is None and process_reward:
        for candidate in process_reward.get("candidates", []) or []:
            if int(candidate.get("candidate_index", -1) or -1) == int(process_reward.get("base_candidate_index", 0) or 0):
                payload = candidate.get("trace") or {}
                if isinstance(payload, dict):
                    base_trace = trace_from_payload(payload)
                break

    t0 = time.perf_counter()
    res = compute_curve(active_backend, question, trace, settings, run_seed=run_seed + 1000)
    stage_timings["forward_curve_s"] = time.perf_counter() - t0
    res["backend"] = getattr(backend, "name", "backend")
    res["runtime_backend"] = getattr(backend, "runtime_backend", getattr(active_backend, "runtime_backend", None))
    res["question_id"] = qid
    res["question"] = question
    res["trace"] = trace_to_payload(trace)
    res["trace_mode"] = settings.trace_mode
    res["trace_format"] = resolve_trace_output_format(settings.trace_format, stream_output=settings.stream_output)
    res["trace_source"] = settings.trace_source
    res["prompt_version"] = settings.prompt_version
    res["prompt_family"] = settings.prompt_family
    if process_reward is not None:
        res["trace_process_reward"] = process_reward
        if base_trace is not None:
            res["base_trace"] = trace_to_payload(base_trace)
        res["prm_trace"] = trace_to_payload(trace)

    if settings.enable_rewind:
        try:
            t0 = time.perf_counter()
            rewind_bundle = compute_rewind_bundle(active_backend, question, trace, settings, res, run_seed=run_seed + 20_000)
            stage_timings["rewind_total_s"] = time.perf_counter() - t0
            res.update(rewind_bundle)
        except Exception as e:
            res["rewind_enabled"] = False
            res["rewind_error"] = str(e)

    if settings.enable_bridge:
        try:
            t0 = time.perf_counter()
            bridge_bundle = compute_bridge_bundles(active_backend, question, trace, settings, res, run_seed=run_seed + 30_000)
            stage_timings["bridge_total_s"] = time.perf_counter() - t0
            res.update(bridge_bundle)
        except Exception as e:
            res["bridge_enabled"] = False
            res["bridge_error"] = str(e)

    # permutation tests
    if settings.permutation_tests > 0:
        t0 = time.perf_counter()
        res["permutations"] = run_permutation_tests(active_backend, question, trace, settings, run_seed=run_seed + 5000)
        stage_timings["permutation_tests_s"] = time.perf_counter() - t0

    rewind_trace_obj = res.get("rewind_trace")
    trace_vs_rewind = compare_forward_and_rewind_steps(trace.steps[: settings.max_steps], rewind_trace_obj) if isinstance(rewind_trace_obj, dict) else None
    if trace_vs_rewind is not None:
        res["trace_vs_rewind"] = trace_vs_rewind
    trace_axes = None
    if process_reward is not None and base_trace is not None and isinstance(rewind_trace_obj, dict):
        trace_axes = build_three_axis_trace_comparison(base_trace, trace, rewind_trace_obj)
        if trace_axes is not None:
            res["trace_axes"] = trace_axes
    rewind_axes = build_rewind_axis_comparison(rewind_trace_obj) if isinstance(rewind_trace_obj, dict) else None
    if rewind_axes is not None:
        res["rewind_axes"] = rewind_axes
    if "trace_build_s" not in stage_timings:
        stage_timings["trace_build_s"] = 0.0
    res["stage_timings_s"] = {
        **stage_timings,
        "total_s": time.perf_counter() - t_total,
    }

    # persist
    out_path = os.path.join(out_dir, f"{backend.name}__{qid}.json")
    write_json(out_path, res)
    if generation_logger is not None:
        log_path = os.path.join(out_dir, f"{backend.name}__{qid}__generations.jsonl")
        generation_logger.write_jsonl(log_path)
        res["generation_log_jsonl"] = os.path.basename(log_path)
        write_json(out_path, res)
    if trace_vs_rewind is not None:
        compare_path = os.path.join(out_dir, f"{backend.name}__{qid}__trace_vs_rewind.jsonl")
        write_jsonl(compare_path, trace_vs_rewind_jsonl_rows(trace_vs_rewind))
        res["trace_vs_rewind_jsonl"] = os.path.basename(compare_path)
        write_json(out_path, res)
    if trace_axes is not None:
        axes_path = os.path.join(out_dir, f"{backend.name}__{qid}__trace_axes.jsonl")
        write_jsonl(axes_path, trace_axes_jsonl_rows(trace_axes))
        res["trace_axes_jsonl"] = os.path.basename(axes_path)
        write_json(out_path, res)
    if rewind_axes is not None:
        rewind_axes_path = os.path.join(out_dir, f"{backend.name}__{qid}__rewind_axes.jsonl")
        write_jsonl(rewind_axes_path, rewind_axes_jsonl_rows(rewind_axes))
        res["rewind_axes_jsonl"] = os.path.basename(rewind_axes_path)
        write_json(out_path, res)
    return res

def run_experiment(config_path: str) -> str:
    cfg = load_config(config_path)
    out_dir = ensure_dir(os.path.join("results", f"run_{now_ts()}"))
    settings = parse_experiment_settings(cfg)
    backends = parse_backends(cfg)
    questions = parse_questions(cfg)

    # Save run meta
    meta = {
        "created_at": datetime.now().isoformat(),
        "python": sys.version,
        "platform": sys.platform,
        "config_path": os.path.abspath(config_path),
        "settings": dataclasses.asdict(settings),
        "backends": [dataclasses.asdict(b) if dataclasses.is_dataclass(b) else {"name": getattr(b, "name", "backend")} for b in backends],
        "n_questions": len(questions),
    }
    write_json(os.path.join(out_dir, "run_meta.json"), meta)

    # Decide trace sharing
    shared_trace: Optional[Trace] = None
    shared_trace_backend_name: Optional[str] = None
    if settings.trace_source and settings.trace_source != "per_backend":
        shared_trace_backend_name = settings.trace_source

    summary_rows: List[Dict[str, Any]] = []

    # Prebuild shared trace if requested
    if shared_trace_backend_name:
        # find backend
        b0 = None
        for b in backends:
            if b.name == shared_trace_backend_name:
                b0 = b
                break
        if b0 is None:
            raise RuntimeError(f"trace_source backend '{shared_trace_backend_name}' not found among backends.")
        # use first question's trace? Actually shared trace per question, not global.
        # We'll build per question below.
        pass

    for q in questions:
        qid = q["id"]
        qtext = q["question"]
        per_question_shared_trace: Optional[Trace] = None
        if shared_trace_backend_name:
            # build shared trace using the specified backend for THIS question
            b0 = next(b for b in backends if b.name == shared_trace_backend_name)
            per_question_shared_trace = build_trace_bundle(
                b0,
                qtext,
                settings,
                seed=12345 + int(sha1(qid)[:6], 16) % 100000,
            )["selected_trace"]

        for b in backends:
            # per backend seed for reproducibility
            run_seed = 42 + (int(sha1(b.name + qid)[:8], 16) % 10_000_000)

            res = run_one_backend_one_question(
                b, qid, qtext, settings, out_dir,
                shared_trace=per_question_shared_trace,
                trace_file=None,
                run_seed=run_seed,
            )

            # optionally skip unstable baselines in summary
            skipped = False
            if settings.skip_unstable_baseline and not res.get("is_baseline_stable", True):
                skipped = True

            row = {
                "backend": b.name,
                "question_id": qid,
                "runtime_backend": res.get("runtime_backend"),
                "prompt_version": settings.prompt_version,
                "prompt_family": settings.prompt_family,
                "trace_format": res.get("trace_format"),
                "L": res.get("L"),
                "A_star": res.get("A_star"),
                "baseline_confidence": res.get("baseline_confidence"),
                "baseline_entropy_bits": res.get("baseline_entropy_bits"),
                "is_baseline_stable": res.get("is_baseline_stable"),
                "unstable_reasons": "; ".join(res.get("baseline_unstable_reasons", []) or []),
                "skipped": skipped,
                "tau": res.get("tau"),
                "smooth_window": res.get("smooth_window"),
                "converge_min_run": res.get("converge_min_run"),
                "collapse_min_run": res.get("collapse_min_run"),
                "converge_k": res.get("converge_k"),
                "collapse_k": res.get("collapse_k"),
                "hysteresis": res.get("hysteresis"),
                "auc_match": res.get("auc_match"),
                "plateau_mean": res.get("plateau_mean"),
                "attractor_strength": res.get("attractor_strength"),
                "stabilization_max_slope": res.get("stabilization_max_slope"),
                "stabilization_max_slope_k": res.get("stabilization_max_slope_k"),
                "rewind_enabled": res.get("rewind_enabled", False),
                "rewind_loop_closure_mean": ((res.get("rewind_trace") or {}).get("loop_closure_mean") if isinstance(res.get("rewind_trace"), dict) else None),
                "rewind_loop_closure_tail_mean": ((res.get("rewind_trace") or {}).get("loop_closure_tail_mean") if isinstance(res.get("rewind_trace"), dict) else None),
                "rewind_auc_match": ((res.get("rewind_curve") or {}).get("auc_match") if isinstance(res.get("rewind_curve"), dict) else None),
                "rewind_converge_depth": ((res.get("rewind_curve") or {}).get("converge_depth") if isinstance(res.get("rewind_curve"), dict) else None),
                "oracle_tail_auc_match": ((res.get("oracle_tail_curve") or {}).get("auc_match") if isinstance(res.get("oracle_tail_curve"), dict) else None),
                "oracle_tail_converge_depth": ((res.get("oracle_tail_curve") or {}).get("converge_depth") if isinstance(res.get("oracle_tail_curve"), dict) else None),
                "directional_gap_auc": res.get("directional_gap_auc"),
                "oracle_recovery_gap_auc": res.get("oracle_recovery_gap_auc"),
                "bridge_enabled": res.get("bridge_enabled", False),
                "bridge_tail_source": res.get("bridge_tail_source"),
                "bridge_oracle_minus_recovered_best_auc": res.get("bridge_oracle_minus_recovered_best_auc"),
                "permutation_tests": settings.permutation_tests,
            }
            row.update(summarize_bridge_for_summary(res, "bridge_recovered"))
            row.update(summarize_bridge_for_summary(res, "bridge_oracle"))
            row.update(summarize_trace_vs_rewind_for_summary(res))
            row.update(summarize_process_reward_for_summary(res))
            row.update(summarize_trace_axes_for_summary(res))
            row.update(summarize_rewind_core_for_summary(res))
            row.update(summarize_rewind_process_reward_for_summary(res))
            row.update(summarize_rewind_axes_for_summary(res))
            row.update(summarize_runtime_for_summary(res))
            summary_rows.append(row)

    # write summary CSV
    csv_path = os.path.join(out_dir, "summary.csv")
    write_text(csv_path, to_csv(summary_rows))
    print(f"[run] saved: {out_dir}")
    return out_dir

def to_csv(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    # stable order: keys from first row
    keys = list(rows[0].keys())
    def esc(v: Any) -> str:
        if v is None:
            return ""
        s = str(v)
        if any(ch in s for ch in [",", "\n", '"']):
            s = '"' + s.replace('"', '""') + '"'
        return s
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(esc(r.get(k)) for k in keys))
    return "\n".join(lines) + "\n"

def run_quick_hf(args: argparse.Namespace) -> str:
    settings = ExperimentSettings(
        trace_mode=args.trace_mode,
        trace_source="per_backend",
        trace_format=args.trace_format,
        stream_output=args.stream_output,
        temperature_trace=args.temp_trace,
        process_reward_mode=args.process_reward_mode,
        trace_candidates=args.trace_candidates,
        trace_candidate_temperature=args.trace_candidate_temperature,
        temperature_reanswer=args.temp_reanswer,
        n_samples_baseline=args.n_baseline,
        n_samples_per_k=args.n_per_k,
        tau=args.tau,
        max_steps=args.max_steps,
        prompt_version=args.prompt_version,
        prompt_family=args.prompt_family,

        baseline_min_conf=args.baseline_min_conf,
        baseline_max_entropy=args.baseline_max_entropy,
        skip_unstable_baseline=args.skip_unstable_baseline,

        smooth_window=args.smooth_window,
        converge_min_run=args.converge_min_run,
        collapse_min_run=args.collapse_min_run,

        compute_divergence=not args.no_divergence,
        divergence_alpha=args.div_alpha,

        permutation_tests=args.permute,
        permutation_seed=args.permute_seed,

        enable_rewind=not args.no_rewind,
        rewind_step_temperature=args.rewind_step_temp,
        rewind_samples_per_depth=args.rewind_samples_per_depth,
        rewind_order=args.rewind_order,
        compute_oracle_tail=not args.no_oracle_tail,
        rewind_process_reward_mode=args.rewind_process_reward_mode,
        rewind_step_candidates=args.rewind_step_candidates,
        rewind_novelty_max_similarity=args.rewind_novelty_max_similarity,
        rewind_novelty_retries=args.rewind_novelty_retries,
        rewind_escape_temperature=args.rewind_escape_temp,
        rewind_step_top_p=args.rewind_step_top_p,
        rewind_step_repetition_penalty=args.rewind_step_repetition_penalty,

        enable_bridge=not args.no_bridge,
        bridge_tail_source=args.bridge_tail_source,
        bridge_samples_per_cell=args.bridge_samples_per_cell,
        bridge_num_prefix_points=args.bridge_num_prefix_points,
        bridge_num_tail_points=args.bridge_num_tail_points,
        bridge_allow_overlap=args.bridge_allow_overlap,
        bridge_middle_temperature=args.bridge_middle_temp,
        bridge_reconstruct_middle=not args.no_bridge_middle,
        bridge_middle_cases=args.bridge_middle_cases,
        bridge_loop_closure_samples=args.bridge_loop_closure_samples,
        bridge_core_cluster_similarity=args.bridge_core_cluster_similarity,
        bridge_core_min_loop_closure=args.bridge_core_min_loop_closure,
        bridge_core_min_match=args.bridge_core_min_match,
        bridge_core_min_synergy=args.bridge_core_min_synergy,
        bridge_core_top_prototypes=args.bridge_core_top_prototypes,
        save_raw_generations=args.save_raw_generations,
        save_generation_log_jsonl=args.save_generation_log_jsonl,
    )
    settings.prompt_version, settings.prompt_family = normalize_prompt_selector(settings.prompt_version, settings.prompt_family)
    settings.trace_format = normalize_trace_output_format(settings.trace_format)

    out_dir = ensure_dir(os.path.join("results", f"run_{now_ts()}"))
    b = HFLocalBackend(
        name=args.name,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        device=args.device,
        hf_backend=args.hf_backend,
        torch_dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        use_fast_tokenizer=(None if args.use_fast_tokenizer is None else args.use_fast_tokenizer),
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    qid = "q1"
    res = run_one_backend_one_question(
        b, qid, args.question, settings, out_dir,
        shared_trace=None,
        trace_file=args.trace_file,
        run_seed=12345,
    )
    # write summary
    summary_row = {
        "backend": b.name,
        "question_id": qid,
        "runtime_backend": res.get("runtime_backend"),
        "prompt_version": settings.prompt_version,
        "prompt_family": settings.prompt_family,
        "trace_format": res.get("trace_format"),
        "L": res.get("L"),
        "A_star": res.get("A_star"),
        "baseline_confidence": res.get("baseline_confidence"),
        "baseline_entropy_bits": res.get("baseline_entropy_bits"),
        "is_baseline_stable": res.get("is_baseline_stable"),
        "unstable_reasons": "; ".join(res.get("baseline_unstable_reasons", []) or []),
        "tau": res.get("tau"),
        "smooth_window": res.get("smooth_window"),
        "converge_min_run": res.get("converge_min_run"),
        "collapse_min_run": res.get("collapse_min_run"),
        "converge_k": res.get("converge_k"),
        "collapse_k": res.get("collapse_k"),
        "hysteresis": res.get("hysteresis"),
        "auc_match": res.get("auc_match"),
        "plateau_mean": res.get("plateau_mean"),
        "attractor_strength": res.get("attractor_strength"),
        "stabilization_max_slope": res.get("stabilization_max_slope"),
        "stabilization_max_slope_k": res.get("stabilization_max_slope_k"),
        "rewind_enabled": res.get("rewind_enabled", False),
        "rewind_loop_closure_mean": ((res.get("rewind_trace") or {}).get("loop_closure_mean") if isinstance(res.get("rewind_trace"), dict) else None),
        "rewind_loop_closure_tail_mean": ((res.get("rewind_trace") or {}).get("loop_closure_tail_mean") if isinstance(res.get("rewind_trace"), dict) else None),
        "rewind_auc_match": ((res.get("rewind_curve") or {}).get("auc_match") if isinstance(res.get("rewind_curve"), dict) else None),
        "rewind_converge_depth": ((res.get("rewind_curve") or {}).get("converge_depth") if isinstance(res.get("rewind_curve"), dict) else None),
        "oracle_tail_auc_match": ((res.get("oracle_tail_curve") or {}).get("auc_match") if isinstance(res.get("oracle_tail_curve"), dict) else None),
        "oracle_tail_converge_depth": ((res.get("oracle_tail_curve") or {}).get("converge_depth") if isinstance(res.get("oracle_tail_curve"), dict) else None),
        "directional_gap_auc": res.get("directional_gap_auc"),
        "oracle_recovery_gap_auc": res.get("oracle_recovery_gap_auc"),
        "bridge_enabled": res.get("bridge_enabled", False),
        "bridge_tail_source": res.get("bridge_tail_source"),
        "bridge_oracle_minus_recovered_best_auc": res.get("bridge_oracle_minus_recovered_best_auc"),
    }
    summary_row.update(summarize_bridge_for_summary(res, "bridge_recovered"))
    summary_row.update(summarize_bridge_for_summary(res, "bridge_oracle"))
    summary_row.update(summarize_trace_vs_rewind_for_summary(res))
    summary_row.update(summarize_process_reward_for_summary(res))
    summary_row.update(summarize_trace_axes_for_summary(res))
    summary_row.update(summarize_rewind_core_for_summary(res))
    summary_row.update(summarize_rewind_process_reward_for_summary(res))
    summary_row.update(summarize_rewind_axes_for_summary(res))
    summary_row.update(summarize_runtime_for_summary(res))
    write_text(os.path.join(out_dir, "summary.csv"), to_csv([summary_row]))
    print(f"[quick-hf] saved: {out_dir}")
    return out_dir

def run_quick_http(args: argparse.Namespace) -> str:
    settings = ExperimentSettings(
        trace_mode=args.trace_mode,
        trace_source="per_backend",
        trace_format=args.trace_format,
        stream_output=args.stream_output,
        temperature_trace=args.temp_trace,
        process_reward_mode=args.process_reward_mode,
        trace_candidates=args.trace_candidates,
        trace_candidate_temperature=args.trace_candidate_temperature,
        temperature_reanswer=args.temp_reanswer,
        n_samples_baseline=args.n_baseline,
        n_samples_per_k=args.n_per_k,
        tau=args.tau,
        max_steps=args.max_steps,
        prompt_version=args.prompt_version,
        prompt_family=args.prompt_family,

        baseline_min_conf=args.baseline_min_conf,
        baseline_max_entropy=args.baseline_max_entropy,
        skip_unstable_baseline=args.skip_unstable_baseline,

        smooth_window=args.smooth_window,
        converge_min_run=args.converge_min_run,
        collapse_min_run=args.collapse_min_run,

        compute_divergence=not args.no_divergence,
        divergence_alpha=args.div_alpha,

        permutation_tests=args.permute,
        permutation_seed=args.permute_seed,

        enable_rewind=not args.no_rewind,
        rewind_step_temperature=args.rewind_step_temp,
        rewind_samples_per_depth=args.rewind_samples_per_depth,
        rewind_order=args.rewind_order,
        compute_oracle_tail=not args.no_oracle_tail,
        rewind_process_reward_mode=args.rewind_process_reward_mode,
        rewind_step_candidates=args.rewind_step_candidates,
        rewind_novelty_max_similarity=args.rewind_novelty_max_similarity,
        rewind_novelty_retries=args.rewind_novelty_retries,
        rewind_escape_temperature=args.rewind_escape_temp,
        rewind_step_top_p=args.rewind_step_top_p,
        rewind_step_repetition_penalty=args.rewind_step_repetition_penalty,

        enable_bridge=not args.no_bridge,
        bridge_tail_source=args.bridge_tail_source,
        bridge_samples_per_cell=args.bridge_samples_per_cell,
        bridge_num_prefix_points=args.bridge_num_prefix_points,
        bridge_num_tail_points=args.bridge_num_tail_points,
        bridge_allow_overlap=args.bridge_allow_overlap,
        bridge_middle_temperature=args.bridge_middle_temp,
        bridge_reconstruct_middle=not args.no_bridge_middle,
        bridge_middle_cases=args.bridge_middle_cases,
        bridge_loop_closure_samples=args.bridge_loop_closure_samples,
        bridge_core_cluster_similarity=args.bridge_core_cluster_similarity,
        bridge_core_min_loop_closure=args.bridge_core_min_loop_closure,
        bridge_core_min_match=args.bridge_core_min_match,
        bridge_core_min_synergy=args.bridge_core_min_synergy,
        bridge_core_top_prototypes=args.bridge_core_top_prototypes,
        save_raw_generations=args.save_raw_generations,
        save_generation_log_jsonl=args.save_generation_log_jsonl,
    )
    settings.prompt_version, settings.prompt_family = normalize_prompt_selector(settings.prompt_version, settings.prompt_family)
    settings.trace_format = normalize_trace_output_format(settings.trace_format)
    out_dir = ensure_dir(os.path.join("results", f"run_{now_ts()}"))
    b = HTTPBackend(
        name=args.name,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout_s=args.timeout_s,
    )
    qid = "q1"
    res = run_one_backend_one_question(
        b, qid, args.question, settings, out_dir,
        trace_file=args.trace_file,
        run_seed=12345,
    )
    summary_row = {
        "backend": b.name,
        "question_id": qid,
        "runtime_backend": res.get("runtime_backend"),
        "prompt_version": settings.prompt_version,
        "prompt_family": settings.prompt_family,
        "trace_format": res.get("trace_format"),
        "L": res.get("L"),
        "A_star": res.get("A_star"),
        "baseline_confidence": res.get("baseline_confidence"),
        "baseline_entropy_bits": res.get("baseline_entropy_bits"),
        "is_baseline_stable": res.get("is_baseline_stable"),
        "unstable_reasons": "; ".join(res.get("baseline_unstable_reasons", []) or []),
        "tau": res.get("tau"),
        "smooth_window": res.get("smooth_window"),
        "converge_min_run": res.get("converge_min_run"),
        "collapse_min_run": res.get("collapse_min_run"),
        "converge_k": res.get("converge_k"),
        "collapse_k": res.get("collapse_k"),
        "hysteresis": res.get("hysteresis"),
        "auc_match": res.get("auc_match"),
        "plateau_mean": res.get("plateau_mean"),
        "attractor_strength": res.get("attractor_strength"),
        "stabilization_max_slope": res.get("stabilization_max_slope"),
        "stabilization_max_slope_k": res.get("stabilization_max_slope_k"),
        "rewind_enabled": res.get("rewind_enabled", False),
        "rewind_loop_closure_mean": ((res.get("rewind_trace") or {}).get("loop_closure_mean") if isinstance(res.get("rewind_trace"), dict) else None),
        "rewind_loop_closure_tail_mean": ((res.get("rewind_trace") or {}).get("loop_closure_tail_mean") if isinstance(res.get("rewind_trace"), dict) else None),
        "rewind_auc_match": ((res.get("rewind_curve") or {}).get("auc_match") if isinstance(res.get("rewind_curve"), dict) else None),
        "rewind_converge_depth": ((res.get("rewind_curve") or {}).get("converge_depth") if isinstance(res.get("rewind_curve"), dict) else None),
        "oracle_tail_auc_match": ((res.get("oracle_tail_curve") or {}).get("auc_match") if isinstance(res.get("oracle_tail_curve"), dict) else None),
        "oracle_tail_converge_depth": ((res.get("oracle_tail_curve") or {}).get("converge_depth") if isinstance(res.get("oracle_tail_curve"), dict) else None),
        "directional_gap_auc": res.get("directional_gap_auc"),
        "oracle_recovery_gap_auc": res.get("oracle_recovery_gap_auc"),
        "bridge_enabled": res.get("bridge_enabled", False),
        "bridge_tail_source": res.get("bridge_tail_source"),
        "bridge_oracle_minus_recovered_best_auc": res.get("bridge_oracle_minus_recovered_best_auc"),
    }
    summary_row.update(summarize_bridge_for_summary(res, "bridge_recovered"))
    summary_row.update(summarize_bridge_for_summary(res, "bridge_oracle"))
    summary_row.update(summarize_trace_vs_rewind_for_summary(res))
    summary_row.update(summarize_process_reward_for_summary(res))
    summary_row.update(summarize_trace_axes_for_summary(res))
    summary_row.update(summarize_rewind_core_for_summary(res))
    summary_row.update(summarize_rewind_process_reward_for_summary(res))
    summary_row.update(summarize_rewind_axes_for_summary(res))
    summary_row.update(summarize_runtime_for_summary(res))
    write_text(os.path.join(out_dir, "summary.csv"), to_csv([summary_row]))
    print(f"[quick] saved: {out_dir}")
    return out_dir

def sample_config_text() -> str:
    return """
# experiment.yaml (sample)
api:
  timeout_s: 120

backends:
  # Local HF
  - name: hf_llama
    type: hf_local
    model: "./model/llama-3.2-3b"
    # tokenizer: "./model/llama-3.2-3b"   # optional
    device: "mps"
    hf_backend: "auto"            # auto | mlx | torch
    torch_dtype: "float16"
    device_map: null
    trust_remote_code: true
    use_fast_tokenizer: null
    max_new_tokens: 256
    top_p: 1.0
    top_k: 50
    repetition_penalty: 1.0

  # OpenAI-compatible HTTP (vLLM/Ollama/LiteLLM/patapi/OpenAI...)
  # - name: local_vllm
  #   base_url: "http://localhost:8000/v1"
  #   api_key: "EMPTY"
  #   model: "YOUR_MODEL"

experiment:
  trace_mode: masked
  trace_source: per_backend        # or a backend name to share trace across backends
  trace_format: auto               # auto | json | text
  stream_output: false
  temperature_trace: 0.0
  process_reward_mode: off         # off | lite
  trace_candidates: 1
  trace_candidate_temperature: 0.4
  temperature_reanswer: 0.8
  n_samples_baseline: 9
  n_samples_per_k: 9
  tau: 0.8
  max_steps: 40
  prompt_version: v2
  prompt_family: math_reasoning   # general_reasoning | math_reasoning | logic_reasoning | story_reasoning | strict_answer_format

  # Research extras
  baseline_min_conf: 0.8           # if full-trace majority is weaker than this, mark unstable
  baseline_max_entropy: null       # e.g. 1.0
  skip_unstable_baseline: false

  smooth_window: 3                 # moving average window on match_rate
  converge_min_run: 2              # require >=tau for 2 consecutive ks
  collapse_min_run: 2              # require <tau for 2 consecutive ks (reverse scan)

  compute_divergence: true
  divergence_alpha: 1.0e-6

  permutation_tests: 0             # >0 enables permutation test runs (costly)
  permutation_seed: 1234

  # Rewind-CoT extras
  enable_rewind: true
  rewind_step_temperature: 0.4
  rewind_samples_per_depth: 0   # 0 -> reuse n_samples_per_k
  rewind_order: reverse         # reverse or forward
  compute_oracle_tail: true
  rewind_process_reward_mode: off   # off | lite
  rewind_step_candidates: 1
  rewind_novelty_max_similarity: 0.82
  rewind_novelty_retries: 3
  rewind_escape_temperature: 0.6
  rewind_step_top_p: 0.9
  rewind_step_repetition_penalty: 1.05

  # Bridge-CoT extras
  enable_bridge: true
  bridge_tail_source: recovered   # recovered | oracle | both
  bridge_samples_per_cell: 0      # 0 -> reuse n_samples_per_k
  bridge_num_prefix_points: 9     # 0 -> full 0..L
  bridge_num_tail_points: 9       # 0 -> full 0..D
  bridge_allow_overlap: false
  bridge_middle_temperature: 0.0
  bridge_reconstruct_middle: true
  bridge_middle_cases: 5
  bridge_loop_closure_samples: 3
  bridge_core_cluster_similarity: 0.72
  bridge_core_min_loop_closure: 0.75
  bridge_core_min_match: 0.75
  bridge_core_min_synergy: 0.0
  bridge_core_top_prototypes: 6
  save_raw_generations: false
  save_generation_log_jsonl: false

questions:
  - id: q1
    question: "Find the general term of the sequence: 1, 4, 9, 16, ..."
  - id: q2
    question: "Simplify: (x+1)^2 - (x-1)^2"
""".strip() + "\n"


# ----------------------------- CLI -----------------------------

def add_common_curve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trace-mode", default="masked", choices=["masked", "unmasked"])
    p.add_argument("--trace-format", default="auto", choices=["auto", "json", "text"], help="Trace/rewind prompt format. 'auto' uses text when --stream-output is enabled, otherwise json.")
    p.add_argument("--stream-output", action="store_true", help="Print streamed CoT first, then rewind recovery, while still saving JSON/JSONL/plots.")
    p.add_argument("--temp-trace", type=float, default=0.0)
    p.add_argument("--process-reward-mode", default="off", choices=["off", "lite"], help="Pseudo-PRM mode for reranking multiple forward trace candidates before downstream evaluation.")
    p.add_argument("--trace-candidates", type=int, default=1, help="How many forward trace candidates to generate before PRM-lite reranking.")
    p.add_argument("--trace-candidate-temperature", type=float, default=0.4, help="Temperature for additional forward trace candidates used in PRM-lite reranking.")
    p.add_argument("--temp-reanswer", type=float, default=0.8)
    p.add_argument("--n-baseline", type=int, default=9)
    p.add_argument("--n-per-k", type=int, default=9)
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--prompt-version", default="v1", help="Prompt template version. Built-in: v1, v2.")
    p.add_argument("--prompt-family", default="", help="Prompt family within the selected version. For v2, try general_reasoning, math_reasoning, logic_reasoning, story_reasoning, or strict_answer_format.")

    # baseline gating
    p.add_argument("--baseline-min-conf", type=float, default=None)
    p.add_argument("--baseline-max-entropy", type=float, default=None)
    p.add_argument("--skip-unstable-baseline", action="store_true")

    # smoothing & robust points
    p.add_argument("--smooth-window", type=int, default=1)
    p.add_argument("--converge-min-run", type=int, default=1)
    p.add_argument("--collapse-min-run", type=int, default=1)

    # divergence
    p.add_argument("--no-divergence", action="store_true")
    p.add_argument("--div-alpha", type=float, default=1e-6)

    # permutation tests
    p.add_argument("--permute", type=int, default=0, help="Number of permutation tests (shuffle steps).")
    p.add_argument("--permute-seed", type=int, default=1234)

    # rewind-CoT extras
    p.add_argument("--no-rewind", action="store_true", help="Disable rewind-side reconstruction / tail curves.")
    p.add_argument("--rewind-step-temp", type=float, default=0.4)
    p.add_argument("--rewind-samples-per-depth", type=int, default=0, help="0 -> reuse --n-per-k")
    p.add_argument("--rewind-order", choices=["reverse", "forward"], default="reverse")
    p.add_argument("--no-oracle-tail", action="store_true", help="Skip oracle tail curve from original final steps.")
    p.add_argument("--rewind-process-reward-mode", default="off", choices=["off", "lite"], help="Pseudo-PRM mode for reranking multiple rewind-step candidates at each depth.")
    p.add_argument("--rewind-step-candidates", type=int, default=1, help="How many rewind-step candidates to sample per depth before PRM-lite reranking.")
    p.add_argument("--rewind-novelty-max-similarity", type=float, default=0.82)
    p.add_argument("--rewind-novelty-retries", type=int, default=3)
    p.add_argument("--rewind-escape-temp", type=float, default=0.6)
    p.add_argument("--rewind-step-top-p", type=float, default=0.9)
    p.add_argument("--rewind-step-repetition-penalty", type=float, default=1.05)

    # bridge-CoT extras
    p.add_argument("--no-bridge", action="store_true", help="Disable bridge-side prefix+rewind combined evaluation.")
    p.add_argument("--bridge-tail-source", choices=["recovered", "oracle", "both"], default="recovered")
    p.add_argument("--bridge-samples-per-cell", type=int, default=0, help="0 -> reuse --n-per-k")
    p.add_argument("--bridge-num-prefix-points", type=int, default=9, help="0 -> full 0..L")
    p.add_argument("--bridge-num-tail-points", type=int, default=9, help="0 -> full 0..D")
    p.add_argument("--bridge-allow-overlap", action="store_true")
    p.add_argument("--bridge-middle-temp", type=float, default=0.0)
    p.add_argument("--no-bridge-middle", action="store_true", help="Skip explicit middle reconstruction for selected bridge cells.")
    p.add_argument("--bridge-middle-cases", type=int, default=5)
    p.add_argument("--bridge-loop-closure-samples", type=int, default=3)
    p.add_argument("--bridge-core-cluster-similarity", type=float, default=0.72)
    p.add_argument("--bridge-core-min-loop-closure", type=float, default=0.75)
    p.add_argument("--bridge-core-min-match", type=float, default=0.75)
    p.add_argument("--bridge-core-min-synergy", type=float, default=0.0)
    p.add_argument("--bridge-core-top-prototypes", type=int, default=6)
    p.add_argument("--save-raw-generations", action="store_true", help="Persist raw baseline/forward/rewind sampled generations into the result JSON for debugging.")
    p.add_argument("--save-generation-log-jsonl", action="store_true", help="Persist every backend.generate prompt/output pair into a per-question JSONL log.")

    # trace file bypass
    p.add_argument("--trace-file", default=None, help="Path to trace JSON file to skip trace generation.")

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sr_rewind_cot.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("quick", help="Quick run against an OpenAI-compatible HTTP endpoint.")
    p1.add_argument("--base-url", required=True)
    p1.add_argument("--api-key", default="")
    p1.add_argument("--model", required=True)
    p1.add_argument("--name", default="http_backend")
    p1.add_argument("--timeout-s", type=int, default=120)
    p1.add_argument("--question", required=True, help="Full problem text, not a short label like 'Solve the puzzle.'")
    add_common_curve_args(p1)

    p2 = sub.add_parser("quick-hf", help="Quick run with a local Hugging Face transformers model.")
    p2.add_argument(
        "--model",
        required=True,
        help="Local model directory or HF model id. If you pass a parent directory, this script tries to resolve the model subdirectory from --name.",
    )
    p2.add_argument("--tokenizer", default=None, help="Optional tokenizer directory/id if different. Missing local paths are ignored.")
    p2.add_argument("--name", default="hf_local")
    p2.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p2.add_argument("--hf-backend", default="auto", choices=["auto", "torch", "mlx"], help="For local HF runs, prefer MLX on Apple Silicon when available, else fall back to torch.")
    p2.add_argument("--dtype", default="auto")
    p2.add_argument("--device-map", default=None)
    p2.add_argument("--trust-remote-code", action="store_true")
    p2.add_argument("--use-fast-tokenizer", default=None, type=lambda x: (str(x).lower() == "true"),
                    help="true/false. If omitted, transformers default.")
    p2.add_argument("--max-new-tokens", type=int, default=256)
    p2.add_argument("--top-p", type=float, default=1.0)
    p2.add_argument("--top-k", type=int, default=50)
    p2.add_argument("--repetition-penalty", type=float, default=1.0)
    p2.add_argument("--question", required=True, help="Full problem text, not a short label like 'Solve the puzzle.'")
    add_common_curve_args(p2)

    p3 = sub.add_parser("run", help="Run from YAML/JSON config (multi-backend, multi-question).")
    p3.add_argument("--config", required=True)

    p4 = sub.add_parser("plot", help="Plot curves for a run directory.")
    p4.add_argument("--run-dir", required=True)

    p5 = sub.add_parser("sample-config", help="Print a sample YAML config.")

    return ap

def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if args.cmd == "sample-config":
        print(sample_config_text())
        return

    if args.cmd == "plot":
        plot_run(args.run_dir)
        return

    if args.cmd == "run":
        out_dir = run_experiment(args.config)
        try:
            plot_run(out_dir)
        except Exception as e:
            print(f"[run] plot skipped: {e}")
        return

    if args.cmd == "quick":
        out_dir = run_quick_http(args)
        try:
            plot_run(out_dir)
        except Exception as e:
            print(f"[quick] plot skipped: {e}")
        return

    if args.cmd == "quick-hf":
        out_dir = run_quick_hf(args)
        try:
            plot_run(out_dir)
        except Exception as e:
            print(f"[quick-hf] plot skipped: {e}")
        return

if __name__ == "__main__":
    main()
