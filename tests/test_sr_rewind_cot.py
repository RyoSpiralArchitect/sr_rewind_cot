import contextlib
import io
import json
import shutil
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

import sr_rewind_cot as spbc


def _tempdir():
    base = Path(__file__).resolve().parent / "_tmp2"
    base.mkdir(parents=True, exist_ok=True)

    p = base / ("t" + uuid.uuid4().hex)
    p.mkdir(parents=False, exist_ok=False)

    class _Ctx:
        def __enter__(self):
            return p

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(p, ignore_errors=True)

    return _Ctx()


def _write_model_dir(path: Path, model_type: str, arch: str, *, with_tokenizer: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "architectures": [arch],
            }
        ),
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(b"")
    if with_tokenizer:
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class TestHFLocalBackendPathResolution(unittest.TestCase):
    def test_resolves_container_model_path_from_name_hint(self) -> None:
        with _tempdir() as td:
            model_root = td / "model"
            _write_model_dir(model_root / "gpt2", "gpt2", "GPT2LMHeadModel")
            _write_model_dir(model_root / "llama-3.2-3b", "llama", "LlamaForCausalLM")
            _write_model_dir(model_root / "minilm", "bert", "BertModel")

            backend = spbc.HFLocalBackend(
                name="llama-3.2-it",
                model_path=str(model_root),
                tokenizer_path=str(model_root / "original"),
            )

            resolved_model = backend._resolve_model_path()
            resolved_tokenizer = backend._resolve_tokenizer_path()

            self.assertEqual(resolved_model, str(model_root / "llama-3.2-3b"))
            self.assertIsNone(resolved_tokenizer)

    def test_raises_when_container_model_path_is_ambiguous(self) -> None:
        with _tempdir() as td:
            model_root = td / "model"
            _write_model_dir(model_root / "gpt2", "gpt2", "GPT2LMHeadModel")
            _write_model_dir(model_root / "mistral7b", "mistral", "MistralForCausalLM")

            backend = spbc.HFLocalBackend(
                name="hf_local",
                model_path=str(model_root),
            )

            with self.assertRaisesRegex(RuntimeError, "Pass --model with the exact model subdirectory"):
                backend._resolve_model_path()

    def test_keeps_direct_model_directory(self) -> None:
        with _tempdir() as td:
            model_dir = td / "model" / "llama-3.2-3b"
            _write_model_dir(model_dir, "llama", "LlamaForCausalLM")

            backend = spbc.HFLocalBackend(
                name="llama",
                model_path=str(model_dir),
            )

            self.assertEqual(backend._resolve_model_path(), str(model_dir))


class TestNamingAndPromptAssets(unittest.TestCase):
    def test_v1_prompt_templates_render_from_assets_folder(self) -> None:
        prompt = spbc.build_trace_prompt("Find the pattern.", "masked")
        self.assertIn("STRICT JSON", prompt)
        self.assertIn("Find the pattern.", prompt)
        self.assertIn('Put the final answer only in the "answer" field.', prompt)
        self.assertTrue((Path(spbc.PROMPTS_DIR) / "trace_masked.txt").exists())

    def test_general_reasoning_question_set_exists(self) -> None:
        if spbc.yaml is None:
            self.skipTest("pyyaml not available")
        path = Path(spbc.SCRIPT_DIR) / "sr_rewind_cot_assets" / "question_sets" / "general_reasoning_observation_v1.yaml"
        self.assertTrue(path.exists())
        cfg = spbc.load_config(str(path))
        questions = cfg.get("questions") or []
        self.assertEqual(len(questions), 6)
        self.assertEqual(questions[0]["id"], "q1")
        self.assertIn("binary search", questions[0]["question"].lower())

    def test_v2_prompt_family_guidance_is_injected(self) -> None:
        prompt = spbc.build_trace_prompt(
            "Find the general term.",
            "masked",
            prompt_version="v2",
            prompt_family="math_reasoning",
        )
        self.assertIn("math_reasoning", prompt)
        self.assertIn("symbolic relations", prompt)
        self.assertTrue((Path(spbc.PROMPTS_DIR) / "v2" / "math_reasoning" / "_guidance.txt").exists())

    def test_v2_family_uses_general_template_fallback_for_bridge_prompt(self) -> None:
        prompt = spbc.build_bridge_answer_prompt(
            "Solve the puzzle.",
            ["premise A"],
            1,
            ["conclusion B"],
            1,
            order="reverse",
            prompt_version="v2",
            prompt_family="logic_reasoning",
        )
        self.assertIn("logic_reasoning", prompt)
        self.assertIn("Track premises, constraints, and exclusions explicitly.", prompt)
        self.assertIn("Known early forward prefix", prompt)

    def test_v1_normalizes_prompt_family_to_default(self) -> None:
        version, family = spbc.normalize_prompt_selector("v1", "math_reasoning")
        self.assertEqual(version, "v1")
        self.assertEqual(family, "default")

    def test_trace_output_format_auto_prefers_text_when_streaming(self) -> None:
        self.assertEqual(spbc.resolve_trace_output_format("auto", stream_output=False), "json")
        self.assertEqual(spbc.resolve_trace_output_format("auto", stream_output=True), "text")

    def test_text_trace_prompt_uses_plaintext_template(self) -> None:
        prompt = spbc.build_trace_prompt(
            "Compare Pascal's Wager and agnostic theism.",
            "unmasked",
            prompt_version="v2",
            prompt_family="logic_reasoning",
            output_format="text",
        )
        self.assertIn("Create a reasoning trace in plain text.", prompt)
        self.assertIn("Answer: ...", prompt)
        self.assertIn("logic_reasoning", prompt)


class TestQuestionValidation(unittest.TestCase):
    def test_placeholder_question_is_detected(self) -> None:
        self.assertTrue(spbc.looks_like_placeholder_question("Solve the puzzle."))
        self.assertFalse(spbc.looks_like_placeholder_question("A, B, and C each make one statement. Exactly one lies. Who lies?"))

    def test_build_trace_fails_early_for_placeholder_question(self) -> None:
        class _NoCallBackend:
            def generate(self, prompt: str, temperature: float, seed=None) -> str:
                raise AssertionError("generate() should not be called for placeholder questions")

        settings = spbc.ExperimentSettings()
        with self.assertRaisesRegex(RuntimeError, "question is underspecified"):
            spbc.build_trace(_NoCallBackend(), "Solve the puzzle.", settings, seed=123)

    def test_failure_classifier_explains_missing_task_content(self) -> None:
        hint = spbc.classify_trace_generation_failure(
            "Solve the puzzle.",
            "I'd be happy to help, but I don't see the puzzle provided. Please provide it.",
        )
        self.assertIsNotNone(hint)
        self.assertIn("complete puzzle/problem statement", hint)


class TestTraceParsing(unittest.TestCase):
    def test_normalize_trace_steps_accepts_object_steps(self) -> None:
        steps = spbc.normalize_trace_steps(
            [
                {"step": "Track the three statements.", "premises": ["A says B did it."]},
                {"text": "Assume A is truthful."},
                "Therefore test B next.",
            ]
        )
        self.assertEqual(
            steps,
            [
                "Track the three statements.",
                "Assume A is truthful.",
                "Therefore test B next.",
            ],
        )

    def test_build_trace_accepts_structured_step_objects_and_final_answer_alias(self) -> None:
        class _StructuredTraceBackend:
            def generate(self, prompt: str, temperature: float, seed=None) -> str:
                return json.dumps(
                    {
                        "steps": [
                            {"step": "Given the three statements."},
                            {"step": "Check which assignment makes exactly one liar."},
                        ],
                        "final_answer": "B is lying.",
                    }
                )

        settings = spbc.ExperimentSettings(prompt_version="v2", prompt_family="logic_reasoning")
        trace = spbc.build_trace(
            _StructuredTraceBackend(),
            "Three suspects A, B, and C each make one statement. Exactly one is lying. Who is lying?",
            settings,
            seed=123,
        )
        self.assertEqual(
            trace.steps,
            [
                "Given the three statements.",
                "Check which assignment makes exactly one liar.",
            ],
        )
        self.assertEqual(trace.answer, "B is lying.")
        self.assertEqual(
            trace.step_records,
            [
                {"step": "Given the three statements.", "id": 1, "type": "step", "text": "Given the three statements."},
                {"step": "Check which assignment makes exactly one liar.", "id": 2, "type": "step", "text": "Check which assignment makes exactly one liar."},
            ],
        )

    def test_extract_trace_payload_fallback_recovers_steps_from_json_like_output(self) -> None:
        raw = """
{
  "steps": [
    {
      "step": "Given statements: A says 'B did it.' B says 'C did it.' C says 'I did not do it.'",
      "premises": ["A says 'B did it.'", "B says 'C did it.'", "C says 'I did not do it."]
    },
    {
      "step": "Assume A is telling the truth. Then B must be the liar.",
      "premises": []
    }
  ],
  "final_answer": "B is lying."
}
""".strip()
        obj = spbc.extract_trace_payload_fallback(raw)
        self.assertIsNotNone(obj)
        self.assertEqual(
            obj["steps"],
            [
                "Given statements: A says 'B did it.' B says 'C did it.' C says 'I did not do it.'",
                "Assume A is telling the truth. Then B must be the liar.",
            ],
        )
        self.assertEqual(obj["answer"], "B is lying.")


class TestProcessRewardLite(unittest.TestCase):
    def test_prm_lite_reranks_trace_candidates_toward_atomic_trace(self) -> None:
        class _TraceBackend:
            def __init__(self) -> None:
                self._ix = 0

            def generate(self, prompt: str, temperature: float, seed=None) -> str:
                outputs = [
                    json.dumps(
                        {
                            "steps": [
                                "In summary, Pascal's Wager and agnostic theism differ because one is prudential and the other is epistemic, so the fundamental difference is pragmatic justification versus uncertain belief, which is the answer."
                            ],
                            "answer": "Pascal's Wager is a prudential argument, while agnostic theism is an epistemic stance of belief without claimed knowledge.",
                        }
                    ),
                    json.dumps(
                        {
                            "steps": [
                                {"type": "facet", "text": "Pascal's Wager argues about whether one should believe under uncertainty."},
                                {"type": "facet", "text": "Agnostic theism combines belief in God with a refusal to claim certain knowledge."},
                                {"type": "comparison", "text": "So the difference is prudential motivation for belief versus an epistemic stance about knowledge."},
                            ],
                            "answer": "Pascal's Wager is prudential, while agnostic theism is epistemic.",
                        }
                    ),
                ]
                out = outputs[min(self._ix, len(outputs) - 1)]
                self._ix += 1
                return out

        settings = spbc.ExperimentSettings(
            process_reward_mode="lite",
            trace_candidates=2,
            trace_candidate_temperature=0.6,
            prompt_version="v2",
            prompt_family="general_reasoning",
        )
        bundle = spbc.build_trace_bundle(
            _TraceBackend(),
            "What is the fundamental difference between Pascal's Wager and agnostic theism?",
            settings,
            seed=123,
        )
        prm = bundle.get("process_reward") or {}
        selected = bundle["selected_trace"]
        self.assertEqual(int(prm.get("selected_candidate_index", -1)), 1)
        self.assertGreater(float(prm.get("selected_score", 0.0) or 0.0), float(prm.get("base_score", 0.0) or 0.0))
        self.assertEqual(
            selected.steps,
            [
                "Pascal's Wager argues about whether one should believe under uncertainty.",
                "Agnostic theism combines belief in God with a refusal to claim certain knowledge.",
                "So the difference is prudential motivation for belief versus an epistemic stance about knowledge.",
            ],
        )

    def test_three_axis_trace_comparison_builds_pairwise_rows(self) -> None:
        base_trace = spbc.Trace(
            steps=[
                "Pascal's Wager concerns pragmatic reasons for belief.",
                "Agnostic theism concerns belief without claimed certainty.",
            ],
            answer="The difference is pragmatic justification versus epistemic stance.",
            step_records=spbc.normalize_trace_step_records(
                [
                    {"type": "facet", "text": "Pascal's Wager concerns pragmatic reasons for belief."},
                    {"type": "facet", "text": "Agnostic theism concerns belief without claimed certainty."},
                ]
            ),
        )
        prm_trace = spbc.Trace(
            steps=[
                "Pascal's Wager is a prudential argument about whether belief is worth adopting.",
                "Agnostic theism allows belief while withholding a claim of knowledge.",
                "So the difference is prudential motivation versus epistemic stance.",
            ],
            answer="The difference is prudential motivation versus epistemic stance.",
            step_records=spbc.normalize_trace_step_records(
                [
                    {"type": "facet", "text": "Pascal's Wager is a prudential argument about whether belief is worth adopting."},
                    {"type": "facet", "text": "Agnostic theism allows belief while withholding a claim of knowledge."},
                    {"type": "comparison", "text": "So the difference is prudential motivation versus epistemic stance."},
                ]
            ),
        )
        rewind_trace = {
            "forward_order": [
                "Pascal's Wager is a prudential argument about whether belief is worth adopting.",
                "Agnostic theism allows belief while withholding a claim of knowledge.",
                "So the difference is prudential motivation versus epistemic stance.",
            ]
        }

        comp = spbc.build_three_axis_trace_comparison(base_trace, prm_trace, rewind_trace)
        self.assertIsNotNone(comp)
        assert comp is not None
        self.assertEqual(comp["base_len"], 2)
        self.assertEqual(comp["prm_len"], 3)
        self.assertEqual(comp["rewind_len"], 3)
        self.assertEqual(len(comp["rows"]), 3)
        self.assertIn("base_prm_similarity", comp["rows"][0])
        self.assertGreater(float((comp["prm_rewind"] or {}).get("similarity_mean") or 0.0), 0.95)

    def test_extract_trace_payload_fallback_recovers_string_step_array_from_truncated_object(self) -> None:
        raw = """
{
  "steps": [
    "Premise 1: From A, B is C.",
    "Premise 2: From B, A is C.",
    "Step 3: Combine Premises 1 and 2 to find a relationship between A and C.",
    "Step 4: Since A is C from Premise 1 and A is C from Premise 2, we can conclude that C is C.",
    "Step 5: Therefore, C is C."
  ],
  "answer": "C"
""".strip()
        obj = spbc.extract_trace_payload_fallback(raw)
        self.assertIsNotNone(obj)
        self.assertEqual(
            obj["steps"],
            [
                "Premise 1: From A, B is C.",
                "Premise 2: From B, A is C.",
                "Step 3: Combine Premises 1 and 2 to find a relationship between A and C.",
                "Step 4: Since A is C from Premise 1 and A is C from Premise 2, we can conclude that C is C.",
                "Step 5: Therefore, C is C.",
            ],
        )
        self.assertEqual(obj["answer"], "C")

    def test_extract_trace_payload_from_plaintext(self) -> None:
        raw = """
Step 1: Pascal's Wager is a pragmatic argument about whether to believe.
Step 2: Agnostic theism is a position about belief together with uncertainty about knowledge.
Answer: Pascal's Wager is an argument for belief under uncertainty, while agnostic theism is a belief position that admits knowledge is uncertain.
""".strip()
        obj = spbc.extract_trace_payload_from_plaintext(raw)
        self.assertIsNotNone(obj)
        self.assertEqual(
            obj["steps"],
            [
                "Pascal's Wager is a pragmatic argument about whether to believe.",
                "Agnostic theism is a position about belief together with uncertainty about knowledge.",
            ],
        )
        self.assertIn("argument for belief under uncertainty", obj["answer"])

    def test_normalize_trace_step_records_preserves_atomic_metadata(self) -> None:
        records = spbc.normalize_trace_step_records(
            [
                {"id": 7, "type": "facet", "text": "Pascal's Wager is pragmatic."},
                {"type": "comparison", "step": "The difference is pragmatic versus epistemic."},
            ]
        )
        self.assertEqual(records[0]["id"], 7)
        self.assertEqual(records[0]["type"], "facet")
        self.assertEqual(records[1]["id"], 2)
        self.assertEqual(records[1]["type"], "comparison")
        self.assertEqual(records[1]["text"], "The difference is pragmatic versus epistemic.")

    def test_build_trace_accepts_plaintext_trace_output(self) -> None:
        class _PlaintextTraceBackend:
            def generate(self, prompt: str, temperature: float, seed=None) -> str:
                return "\n".join(
                    [
                        "Step 1: Track what Pascal's Wager is trying to justify.",
                        "Step 2: Track what agnostic theism says about belief and knowledge.",
                        "Answer: Pascal's Wager is an argument for adopting belief, while agnostic theism is a belief stance paired with uncertainty about knowing.",
                    ]
                )

        settings = spbc.ExperimentSettings(
            trace_format="text",
            prompt_version="v2",
            prompt_family="logic_reasoning",
        )
        trace = spbc.build_trace(
            _PlaintextTraceBackend(),
            "What is the fundamental difference between Pascal's Wager and agnostic theism?",
            settings,
            seed=123,
        )
        self.assertEqual(
            trace.steps,
            [
                "Track what Pascal's Wager is trying to justify.",
                "Track what agnostic theism says about belief and knowledge.",
            ],
        )
        self.assertIn("belief stance", trace.answer)


class TestLoggingAndComparison(unittest.TestCase):
    def test_logged_backend_records_prompt_and_output_rows(self) -> None:
        class _EchoBackend:
            name = "echo"

            def generate(self, prompt: str, temperature: float, seed=None) -> str:
                return '{"steps":["x"],"answer":"y"}'

        logger = spbc.GenerationLogger(
            backend_name="echo",
            question_id="q1",
            question="Find the liar.",
            prompt_version="v2",
            prompt_family="logic_reasoning",
        )
        backend = spbc.LoggedBackend(_EchoBackend(), logger)
        out = backend.generate("Create a reasoning trace as STRICT JSON:", temperature=0.0, seed=7)
        self.assertEqual(out, '{"steps":["x"],"answer":"y"}')
        self.assertEqual(len(logger.rows), 1)
        self.assertEqual(logger.rows[0]["stage"], "trace")
        self.assertEqual(logger.rows[0]["status"], "ok")
        self.assertEqual(logger.rows[0]["seed"], 7)

    def test_compare_forward_and_rewind_steps_summarizes_differences(self) -> None:
        comp = spbc.compare_forward_and_rewind_steps(
            ["premise", "branch A", "therefore B lies"],
            {"forward_order": ["premise", "branch X", "therefore B lies", "extra wrap-up"]},
        )
        self.assertIsNotNone(comp)
        self.assertEqual(comp["exact_match_count"], 2)
        self.assertEqual(comp["changed_count"] + comp["paraphrase_count"], 1)
        self.assertEqual(comp["extra_count"], 1)
        self.assertGreaterEqual(comp["preservation_rate"], 2 / 3)
        self.assertLessEqual(comp["preservation_rate"], 1.0)
        self.assertEqual(len(comp["pairwise_alignment"]), 4)

    def test_trace_vs_rewind_jsonl_rows_include_summary_and_steps(self) -> None:
        comp = {
            "original_len": 2,
            "rewind_len": 2,
            "similarity_mean": 0.5,
            "similarity_concat": 0.6,
            "preservation_rate": 0.5,
            "exact_match_count": 1,
            "paraphrase_count": 0,
            "changed_count": 1,
            "missing_count": 0,
            "extra_count": 0,
            "pairwise_alignment": [{"index": 0, "similarity": 1.0, "status": "exact"}],
        }
        rows = spbc.trace_vs_rewind_jsonl_rows(comp)
        self.assertEqual(rows[0]["record_type"], "summary")
        self.assertEqual(rows[1]["record_type"], "step")
        self.assertEqual(rows[1]["status"], "exact")

    def test_logged_backend_stream_generate_records_joined_output(self) -> None:
        class _StreamingEchoBackend:
            name = "streaming_echo"

            def stream_generate(self, prompt: str, temperature: float, seed=None):
                yield "Step 1\n"
                yield "Answer: X"

        logger = spbc.GenerationLogger(
            backend_name="streaming_echo",
            question_id="q1",
            question="Find X.",
            prompt_version="v2",
            prompt_family="logic_reasoning",
        )
        backend = spbc.LoggedBackend(_StreamingEchoBackend(), logger)
        out = "".join(backend.stream_generate("Create a reasoning trace in plain text.", temperature=0.0, seed=9))
        self.assertEqual(out, "Step 1\nAnswer: X")
        self.assertEqual(len(logger.rows), 1)
        self.assertEqual(logger.rows[0]["stage"], "trace")
        self.assertEqual(logger.rows[0]["output"], "Step 1\nAnswer: X")

    def test_emit_streamed_generation_prints_and_collects_output(self) -> None:
        class _StreamingBackend:
            def stream_generate(self, prompt: str, temperature: float, seed=None):
                yield "cot"
                yield " trace"

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = spbc.emit_streamed_generation(
                _StreamingBackend(),
                "Create a reasoning trace in plain text.",
                temperature=0.0,
                seed=1,
                enable_stream=True,
                label="demo",
            )
        self.assertEqual(out, "cot trace")
        self.assertIn("=== demo ===", buf.getvalue())
        self.assertIn("cot trace", buf.getvalue())

    def test_rewind_core_metrics_capture_fixed_point_behavior(self) -> None:
        rewind_trace = {
            "fixed_point_depth": 3,
            "fixed_point_recurrence_rate": 1.0,
            "pre_fixed_novelty_mean": 0.6,
            "loop_closure_tail_mean": 0.5,
        }
        rewind_curve = {
            "curve": [
                {"top_answer": "x"},
                {"top_answer": "a"},
                {"top_answer": "core"},
                {"top_answer": "core"},
                {"top_answer": "core"},
            ]
        }
        core = spbc.compute_rewind_core_metrics(rewind_trace, rewind_curve)
        self.assertEqual(core["fixed_point_depth"], 3)
        self.assertEqual(core["answer_reproduction_rate"], 1.0)
        self.assertAlmostEqual(core["core_strength"], 0.3)


class TestHFBackendSelection(unittest.TestCase):
    def test_hf_backend_auto_prefers_mlx_on_mps(self) -> None:
        backend = spbc.HFLocalBackend(name="x", model_path="./model", device="mps", hf_backend="auto")
        self.assertTrue(backend._should_try_mlx())

    def test_hf_backend_auto_falls_back_to_torch_when_mlx_load_fails(self) -> None:
        backend = spbc.HFLocalBackend(name="x", model_path="./model", device="mps", hf_backend="auto")
        with mock.patch.object(backend, "_resolve_model_path", return_value="./model"), \
             mock.patch.object(backend, "_resolve_tokenizer_path", return_value=None), \
             mock.patch.object(backend, "_load_mlx", side_effect=RuntimeError("mlx unavailable")), \
             mock.patch.object(backend, "_load_torch") as load_torch:
            backend._load()
        load_torch.assert_called_once()

    def test_hf_backend_forced_mlx_surfaces_load_error(self) -> None:
        backend = spbc.HFLocalBackend(name="x", model_path="./model", device="mps", hf_backend="mlx")
        with mock.patch.object(backend, "_resolve_model_path", return_value="./model"), \
             mock.patch.object(backend, "_resolve_tokenizer_path", return_value=None), \
             mock.patch.object(backend, "_load_mlx", side_effect=RuntimeError("mlx unavailable")):
            with self.assertRaisesRegex(RuntimeError, "mlx unavailable"):
                backend._load()

    def test_generate_torch_imports_torch_locally(self) -> None:
        backend = spbc.HFLocalBackend(name="x", model_path="./model", device="mps", hf_backend="torch")
        backend._model = mock.Mock()
        backend._tokenizer = mock.Mock()
        backend._loaded = True
        backend._runtime_backend = "torch"

        class _FakeTensor:
            def __init__(self, values):
                self.values = list(values)
                self.shape = (1, len(self.values))

            def to(self, device):
                return self

        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        backend._tokenizer.return_value = {"input_ids": _FakeTensor([1, 2, 3])}
        backend._tokenizer.decode.return_value = "decoded"
        backend._model.generate.return_value = [[1, 2, 3, 4, 5]]
        fake_torch = types.SimpleNamespace(no_grad=lambda: _NoGrad())

        with mock.patch.object(backend, "_seed_torch") as seed_torch, \
             mock.patch.object(backend, "_format_prompt", return_value=("PROMPT", True)), \
             mock.patch.dict(sys.modules, {"torch": fake_torch}):
            out = backend._generate_torch("hello", temperature=0.0, seed=17)

        self.assertEqual(out, "decoded")
        seed_torch.assert_called_once_with(17)
        backend._model.generate.assert_called_once()

    def test_stream_generate_mlx_uses_sampler_not_temp_kwargs(self) -> None:
        backend = spbc.HFLocalBackend(name="x", model_path="./model", device="mps", hf_backend="mlx")
        backend._model = object()
        backend._tokenizer = object()
        backend._loaded = True
        backend._runtime_backend = "mlx"

        captured: dict = {}

        fake_mlx_core = types.SimpleNamespace(
            random=types.SimpleNamespace(seed=lambda x: captured.setdefault("seed", x)),
        )

        def _make_sampler(**kwargs):
            captured["sampler_kwargs"] = kwargs
            return "sampler"

        def _make_logits_processors(**kwargs):
            captured["logits_kwargs"] = kwargs
            return ["lp"]

        def _stream_generate(model, tokenizer, **kwargs):
            captured["stream_kwargs"] = kwargs
            yield types.SimpleNamespace(text="chunk-1")
            yield types.SimpleNamespace(text="chunk-2")

        fake_sample_utils = types.SimpleNamespace(
            make_sampler=_make_sampler,
            make_logits_processors=_make_logits_processors,
        )
        fake_mlx_lm = types.SimpleNamespace(stream_generate=_stream_generate)

        with mock.patch.object(backend, "_format_prompt", return_value=("PROMPT", True)), \
             mock.patch.dict(sys.modules, {
                 "mlx.core": fake_mlx_core,
                 "mlx_lm.sample_utils": fake_sample_utils,
                 "mlx_lm": fake_mlx_lm,
             }):
            out = list(backend._stream_generate_mlx("hello", temperature=0.3, seed=17))

        self.assertEqual(out, ["chunk-1", "chunk-2"])
        self.assertEqual(captured["seed"], 17)
        self.assertEqual(captured["sampler_kwargs"]["temp"], 0.3)
        self.assertEqual(captured["sampler_kwargs"]["top_k"], backend.top_k)
        self.assertNotIn("temp", captured["stream_kwargs"])
        self.assertEqual(captured["stream_kwargs"]["sampler"], "sampler")
        self.assertEqual(captured["stream_kwargs"]["logits_processors"], ["lp"])

    def test_extract_mlx_chunk_text_ignores_repr_when_text_is_empty(self) -> None:
        backend = spbc.HFLocalBackend(name="x", model_path="./model", device="mps", hf_backend="mlx")
        chunk = types.SimpleNamespace(text="", token=123)
        self.assertEqual(backend._extract_mlx_chunk_text(chunk), "")

    def test_generate_many_mlx_reuses_prefilled_prompt_cache(self) -> None:
        backend = spbc.HFLocalBackend(name="x", model_path="./model", device="mps", hf_backend="mlx")
        backend._model = object()
        backend._tokenizer = object()
        backend._loaded = True
        backend._runtime_backend = "mlx"

        captured_prompts = []
        captured_caches = []
        seed_values = []
        base_cache = [types.SimpleNamespace(tag="base")]

        fake_mlx_core = types.SimpleNamespace(
            random=types.SimpleNamespace(seed=lambda x: seed_values.append(x)),
            clear_cache=lambda: None,
        )

        def _make_sampler(**kwargs):
            return ("sampler", kwargs)

        def _make_logits_processors(**kwargs):
            return [("lp", kwargs)]

        def _stream_generate(model, tokenizer, **kwargs):
            captured_prompts.append(kwargs["prompt"])
            captured_caches.append(kwargs.get("prompt_cache"))
            yield types.SimpleNamespace(text="part-a")
            yield types.SimpleNamespace(text="part-b")

        fake_sample_utils = types.SimpleNamespace(
            make_sampler=_make_sampler,
            make_logits_processors=_make_logits_processors,
        )
        fake_mlx_lm = types.SimpleNamespace(stream_generate=_stream_generate)

        with mock.patch.object(backend, "_format_prompt", return_value=("PROMPT", True)), \
             mock.patch.object(backend, "_tokenize_mlx_prompt", return_value=[11, 22, 33]), \
             mock.patch.object(backend, "_prepare_mlx_prompt_reuse", return_value=([33], base_cache)), \
             mock.patch.dict(sys.modules, {
                 "mlx.core": fake_mlx_core,
                 "mlx_lm.sample_utils": fake_sample_utils,
                 "mlx_lm": fake_mlx_lm,
             }):
            out = backend._generate_many_mlx("hello", temperature=0.3, seeds=[17, 18])

        self.assertEqual(out, ["part-apart-b", "part-apart-b"])
        self.assertEqual(captured_prompts, [[33], [33]])
        self.assertEqual(seed_values, [17, 18])
        self.assertEqual(len(captured_caches), 2)
        self.assertIsNot(captured_caches[0], base_cache)
        self.assertIsNot(captured_caches[1], base_cache)
        self.assertIsNot(captured_caches[0], captured_caches[1])


class _StubBackend:
    def __init__(self) -> None:
        self.name = "stub"
        self._rewind_step_ix = 0

    def generate(self, prompt: str, temperature: float, seed=None) -> str:
        if "Create a reasoning trace as STRICT JSON" in prompt:
            return json.dumps({"steps": ["step 1", "step 2"], "answer": "a_n = n^2"})
        if "Recover EXACTLY ONE short reasoning step" in prompt:
            self._rewind_step_ix += 1
            return json.dumps({"step": f"rewind step {self._rewind_step_ix}"})
        return "a_n = n^2"


class TestRawGenerationCapture(unittest.TestCase):
    def test_save_raw_generations_persists_forward_and_rewind_samples(self) -> None:
        backend = _StubBackend()
        settings = spbc.ExperimentSettings(
            n_samples_baseline=2,
            n_samples_per_k=2,
            rewind_samples_per_depth=2,
            max_steps=2,
            save_raw_generations=True,
            compute_oracle_tail=True,
        )
        trace = spbc.Trace(steps=["step 1", "step 2"], answer="a_n = n^2")

        forward = spbc.compute_curve(
            backend,
            "数列 1,4,9,16,... の一般項を求めよ。",
            trace,
            settings,
            run_seed=123,
        )
        rewind = spbc.compute_rewind_bundle(
            backend,
            "数列 1,4,9,16,... の一般項を求めよ。",
            trace,
            settings,
            forward,
            run_seed=456,
        )

        self.assertEqual(forward["baseline_raw_answers"], ["a_n = n^2", "a_n = n^2"])
        self.assertEqual(len(forward["forward_raw_answers_by_k"]), 3)
        self.assertEqual(forward["forward_raw_answers_by_k"][0]["k"], 0)
        self.assertEqual(forward["forward_raw_answers_by_k"][0]["answers"], ["a_n = n^2", "a_n = n^2"])

        self.assertIn("raw_recoveries", rewind["rewind_trace"])
        self.assertEqual(len(rewind["rewind_trace"]["raw_recoveries"]), 2)
        self.assertEqual(rewind["rewind_trace"]["raw_recoveries"][0]["parsed_step"], "rewind step 1")

        self.assertIn("raw_answers_by_depth", rewind["rewind_curve"])
        self.assertEqual(len(rewind["rewind_curve"]["raw_answers_by_depth"]), 3)
        self.assertEqual(rewind["rewind_curve"]["raw_answers_by_depth"][1]["depth"], 1)
        self.assertEqual(rewind["rewind_curve"]["raw_answers_by_depth"][1]["answers"], ["a_n = n^2", "a_n = n^2"])

        self.assertIn("raw_answers_by_depth", rewind["oracle_tail_curve"])
        self.assertEqual(len(rewind["oracle_tail_curve"]["raw_answers_by_depth"]), 3)


class TestBatchGenerationReuse(unittest.TestCase):
    def test_logged_backend_generate_many_logs_each_output(self) -> None:
        class _BatchEchoBackend:
            name = "batch_echo"

            def generate_many(self, prompt: str, temperature: float, seeds):
                return [f"out-{seed}" for seed in seeds]

        logger = spbc.GenerationLogger(
            backend_name="batch_echo",
            question_id="q1",
            question="Find X.",
            prompt_version="v2",
            prompt_family="general_reasoning",
        )
        backend = spbc.LoggedBackend(_BatchEchoBackend(), logger)
        outs = backend.generate_many("Create a reasoning trace as STRICT JSON:", temperature=0.2, seeds=[5, 6])
        self.assertEqual(outs, ["out-5", "out-6"])
        self.assertEqual(len(logger.rows), 2)
        self.assertEqual(logger.rows[0]["seed"], 5)
        self.assertEqual(logger.rows[1]["seed"], 6)
        self.assertEqual(logger.rows[0]["stage"], "trace")

    def test_build_trace_bundle_keeps_trace_candidates_sequential(self) -> None:
        class _TraceBatchBackend:
            def __init__(self) -> None:
                self.name = "trace-batch"
                self.single_calls = 0
                self.many_calls = []

            def generate(self, prompt: str, temperature: float, seed=None) -> str:
                self.single_calls += 1
                return json.dumps({"steps": ["base trace"], "answer": "base answer"})

            def generate_many(self, prompt: str, temperature: float, seeds):
                self.many_calls.append({"prompt": prompt, "temperature": temperature, "seeds": list(seeds)})
                return [
                    json.dumps({"steps": [f"candidate trace {i + 1}"], "answer": f"answer {i + 1}"})
                    for i, _ in enumerate(seeds)
                ]

        backend = _TraceBatchBackend()
        settings = spbc.ExperimentSettings(
            process_reward_mode="lite",
            trace_candidates=4,
            trace_candidate_temperature=0.6,
            prompt_version="v2",
            prompt_family="general_reasoning",
        )
        bundle = spbc.build_trace_bundle(
            backend,
            "Why does binary search require a sorted array?",
            settings,
            seed=123,
        )
        self.assertEqual(backend.single_calls, 4)
        self.assertEqual(len(backend.many_calls), 0)
        self.assertEqual(len((bundle.get("process_reward") or {}).get("candidates") or []), 4)

    def test_compute_curve_keeps_forward_answers_sequential(self) -> None:
        class _AnswerBatchBackend:
            def __init__(self) -> None:
                self.name = "answer-batch"
                self.single_calls = []
                self.many_calls = []

            def generate(self, prompt: str, temperature: float, seed=None) -> str:
                self.single_calls.append({"prompt": prompt, "temperature": temperature, "seed": seed})
                return "stable answer"

            def generate_many(self, prompt: str, temperature: float, seeds):
                self.many_calls.append({"prompt": prompt, "temperature": temperature, "seeds": list(seeds)})
                return ["stable answer" for _ in seeds]

        backend = _AnswerBatchBackend()
        settings = spbc.ExperimentSettings(
            n_samples_baseline=2,
            n_samples_per_k=2,
        )
        trace = spbc.Trace(steps=["step 1", "step 2"], answer="stable answer")
        result = spbc.compute_curve(
            backend,
            "Why does binary search require a sorted array?",
            trace,
            settings,
            run_seed=123,
        )
        self.assertEqual(len(backend.many_calls), 0)
        self.assertEqual(len(backend.single_calls), 8)
        self.assertEqual(result["A_star"], "stable answer")

    def test_only_tail_curve_uses_batch_exact_prompt_samples(self) -> None:
        class _RewindBatchBackend:
            def __init__(self) -> None:
                self.name = "rewind-batch"
                self.single_calls = []
                self.many_calls = []

            def generate(self, prompt: str, temperature: float, seed=None) -> str:
                self.single_calls.append({"prompt": prompt, "temperature": temperature, "seed": seed})
                if "Recover EXACTLY ONE short reasoning step" in prompt:
                    return json.dumps({"step": "rewind candidate"})
                return "stable answer"

            def generate_many(self, prompt: str, temperature: float, seeds):
                self.many_calls.append({"prompt": prompt, "temperature": temperature, "seeds": list(seeds)})
                return ["stable answer" for _ in seeds]

        backend = _RewindBatchBackend()
        trace = spbc.Trace(
            steps=["step 1", "step 2"],
            answer="stable answer",
        )
        settings = spbc.ExperimentSettings(
            rewind_process_reward_mode="lite",
            rewind_step_candidates=3,
            rewind_novelty_retries=0,
            rewind_samples_per_depth=2,
        )
        rewind_trace = spbc.recover_rewind_trace(
            backend,
            "Why does binary search require a sorted array?",
            trace,
            settings,
            run_seed=123,
        )
        rewind_calls = [
            call for call in backend.single_calls
            if "Recover EXACTLY ONE short reasoning step" in call["prompt"]
        ]
        self.assertEqual(len(rewind_calls), 6)
        self.assertEqual(len(backend.many_calls), 0)
        curve = spbc.compute_tail_curve(
            backend,
            "Why does binary search require a sorted array?",
            rewind_trace["latest_to_earlier"],
            settings,
            run_seed=456,
            baseline_A_star="stable answer",
            baseline_distribution={"stable answer": 2},
            label="recovered",
        )
        tail_calls = [
            call for call in backend.many_calls
            if "Partial rewind trace" in call["prompt"]
        ]
        self.assertEqual(len(tail_calls), 3)
        self.assertTrue(all(len(call["seeds"]) == 2 for call in tail_calls))
        self.assertEqual(curve["answer_calls"], 6)


class TestRewindProcessRewardLite(unittest.TestCase):
    def test_rewind_prm_lite_prefers_atomic_candidate(self) -> None:
        class _RewindBackend:
            def __init__(self) -> None:
                self.name = "rewind-prm"
                self._ix = 0

            def generate(self, prompt: str, temperature: float, seed=None) -> str:
                if "Recover EXACTLY ONE short reasoning step" not in prompt:
                    return "irrelevant"
                outputs = [
                    json.dumps({
                        "step": "In summary, the difference is that Pascal's Wager is practical and agnostic theism is epistemic, which already states the answer.",
                        "novelty": "compressed comparison",
                        "why_earlier": "",
                    }),
                    json.dumps({
                        "step": "Pascal's Wager asks whether belief is worth adopting under uncertainty.",
                        "novelty": "isolates Pascal's Wager before the comparison step",
                        "why_earlier": "It introduces one side of the comparison as a prerequisite.",
                    }),
                    json.dumps({
                        "step": "Overall, agnostic theism differs because it keeps belief while denying certainty, which again summarizes the answer.",
                        "novelty": "summary restatement",
                        "why_earlier": "",
                    }),
                    json.dumps({
                        "step": "Agnostic theism allows belief in God while withholding a claim of certain knowledge.",
                        "novelty": "isolates the epistemic stance as a separate facet",
                        "why_earlier": "It supplies the second prerequisite before the final contrast.",
                    }),
                ]
                out = outputs[min(self._ix, len(outputs) - 1)]
                self._ix += 1
                return out

        trace = spbc.Trace(
            steps=[
                "Pascal's Wager is prudential.",
                "Agnostic theism is epistemic.",
            ],
            answer="The difference is prudential motivation versus epistemic stance.",
        )
        settings = spbc.ExperimentSettings(
            rewind_process_reward_mode="lite",
            rewind_step_candidates=2,
            rewind_novelty_retries=0,
        )
        rewind_trace = spbc.recover_rewind_trace(
            _RewindBackend(),
            "What is the fundamental difference between Pascal's Wager and agnostic theism?",
            trace,
            settings,
            run_seed=123,
        )

        self.assertEqual(
            rewind_trace["latest_to_earlier"][0],
            "Pascal's Wager asks whether belief is worth adopting under uncertainty.",
        )
        self.assertEqual(
            rewind_trace["latest_to_earlier"][1],
            "Agnostic theism allows belief in God while withholding a claim of certain knowledge.",
        )
        self.assertEqual(rewind_trace["process_reward_mode"], "lite")
        self.assertEqual(rewind_trace["process_reward_step_candidates"], 2)
        self.assertGreater(float(rewind_trace["process_reward_mean"] or 0.0), 0.5)
        self.assertEqual(rewind_trace["step_records"][0]["selected_candidate_index"], 1)
        self.assertEqual(
            rewind_trace["base_rewind"]["latest_to_earlier"][0],
            "In summary, the difference is that Pascal's Wager is practical and agnostic theism is epistemic, which already states the answer.",
        )
        self.assertEqual(rewind_trace["generation_calls"], 4)
        self.assertEqual(rewind_trace["generation_attempts"], 2)
        self.assertGreater(
            float(rewind_trace["process_reward_mean"] or 0.0),
            float(rewind_trace["process_reward_base_mean"] or 0.0),
        )

        rewind_axes = spbc.build_rewind_axis_comparison(rewind_trace)
        self.assertIsNotNone(rewind_axes)
        self.assertLess(
            float((rewind_axes or {}).get("rewind_base_prm", {}).get("similarity_mean") or 1.0),
            1.0,
        )

    def test_runtime_summary_includes_rewind_workload(self) -> None:
        summary = spbc.summarize_runtime_for_summary({
            "stage_timings_s": {
                "trace_build_s": 0.11,
                "forward_curve_s": 0.22,
                "bridge_total_s": 0.55,
                "total_s": 1.23,
            },
            "rewind_timings_s": {
                "rewind_total_s": 0.66,
                "rewind_trace_s": 0.21,
                "rewind_curve_s": 0.30,
                "oracle_tail_curve_s": 0.15,
            },
            "rewind_workload": {
                "rewind_trace_generation_calls": 8,
                "rewind_trace_attempts": 4,
                "rewind_tail_answer_calls": 27,
                "oracle_tail_answer_calls": 27,
                "rewind_total_generation_calls": 62,
            },
        })
        self.assertEqual(summary["rewind_trace_generation_calls"], 8)
        self.assertEqual(summary["rewind_trace_attempts"], 4)
        self.assertEqual(summary["rewind_total_generation_calls"], 62)
        self.assertAlmostEqual(float(summary["time_rewind_total_s"] or 0.0), 0.66, places=6)


if __name__ == "__main__":
    unittest.main()
