"""CPU contracts for producer budgets, collector fallback and phase handoff.

The small tokenizer/job doubles exercise Tabby's real collector and backend
methods without importing CUDA. The engine separately tests accepted-token,
MTP, rewind, EOS and producer queue sequencing with its native Job source.
"""
from __future__ import annotations

import __future__
import ast
import asyncio
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, List
import unittest
from unittest.mock import ANY, Mock, patch

from backends.exllamav3.reasoning import (
    NativeReasoningBudget,
    ReasoningBoundaryGuard,
    encode_forced_output,
    prepare_native_reasoning_budget,
    producer_phase_end_callback,
    supports_native_reasoning_budget,
)
from endpoints.OAI.utils import chat_completion as cc
from endpoints.OAI.utils.stream_parser import Qwen3CoderStreamParser, TagStreamParser
from tests.test_qwen_tool_contract import xml
from tests.test_tool_delta_streaming import make_mc, make_request, pieces


class Ids:
    """Only the CPU indexing operations used when preparing an injection."""
    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, len(self.values))

    def __getitem__(self, item):
        row, column = item
        if isinstance(column, slice):
            return Ids(self.values[column])
        return SimpleNamespace(item=lambda: self.values[column])

    def size(self, dim):
        return self.shape[dim]


class Tokenizer:
    def __init__(self, ids=(8, 9), end_id=9, bos_id=None):
        self.ids = Ids(ids)
        self.end_id = end_id
        self.bos_token_id = bos_id
        self.bos_token = "<bos>"
        self.eos_token_id = 99
        self.encode_calls = []
        self.single_calls = []

    def single_id(self, text):
        self.single_calls.append(text)
        return self.end_id

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        return self.ids

    def get_id_to_piece_list(self, include_special_tokens):
        return ["text"] * 9 + ["</think>"]


def prepared(tokenizer=None, budget=2, **kwargs):
    return prepare_native_reasoning_budget(
        tokenizer or Tokenizer(), budget, "finish thought</think>",
        initial_reasoning=kwargs.pop("initial_reasoning", True),
        end_token=kwargs.pop("end_token", "</think>"),
        supported=kwargs.pop("supported", True), **kwargs,
    )


def backend_symbols():
    """Execute actual backend methods; only CUDA-heavy imports are excluded."""
    path = Path(__file__).parents[1] / "backends/exllamav3/model.py"
    module = ast.parse(path.read_text())
    classes = {node.name: node for node in module.body if isinstance(node, ast.ClassDef)}
    names = {
        "prepare_reasoning_budget", "set_generation_phase", "constrain_generation_output",
        "stream_generate", "generate_gen",
    }
    methods = [deepcopy(node) for node in classes["ExllamaV3Container"].body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert {node.name for node in methods} == names
    body = [deepcopy(classes["JobPhases"]), *methods]
    namespace = {
        "__name__": __name__, "dataclass": dataclass, "Any": Any, "List": List,
        "prepare_native_reasoning_budget": prepare_native_reasoning_budget,
        "encode_forced_output": encode_forced_output,
        "producer_phase_end_callback": producer_phase_end_callback,
        "supports_native_reasoning_budget": supports_native_reasoning_budget,
        "ReasoningBoundaryGuard": ReasoningBoundaryGuard,
        "xlogger": SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None),
        "CancelledError": asyncio.CancelledError,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


class PreparationTests(unittest.TestCase):
    def test_supported_initial_single_token_format_and_zero_budget(self):
        for budget in (0, 1, 24):
            with self.subTest(budget=budget):
                tokenizer = Tokenizer()
                plan = prepared(tokenizer, budget)
                self.assertIsInstance(plan, NativeReasoningBudget)
                self.assertEqual((plan.max_tokens, plan.end_token_id), (budget, 9))
                self.assertIs(plan.output_ids, tokenizer.ids)
                self.assertEqual(tokenizer.encode_calls, [
                    ("finish thought</think>", {"encode_special_tokens": True, "add_bos": False})])

    def test_unsupported_phase_engine_and_absent_or_multitoken_marker_fall_back(self):
        for options in ({"supported": False}, {"initial_reasoning": False}, {"end_token": None}):
            with self.subTest(options=options):
                tokenizer = Tokenizer()
                self.assertIsNone(prepared(tokenizer, **options))
                self.assertEqual(tokenizer.single_calls, [])
                self.assertEqual(tokenizer.encode_calls, [])
        tokenizer = Tokenizer(end_id=None)
        self.assertIsNone(prepared(tokenizer))
        self.assertEqual(tokenizer.encode_calls, [])

    def test_invalid_budgets_are_rejected(self):
        for budget in (-1, 1.5, True, None, "24"):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    prepared(budget=budget)

    def test_bos_postprocessor_is_removed_without_changing_other_ids(self):
        tokenizer = Tokenizer(ids=(77, 8, 9), bos_id=77)
        self.assertEqual(prepared(tokenizer).output_ids.values, [8, 9])
        self.assertEqual(tokenizer.ids.values, [77, 8, 9])
        self.assertEqual(encode_forced_output(Tokenizer(ids=(8, 9), bos_id=77), "x").values, [8, 9])

    def test_injection_must_end_with_the_single_closing_token(self):
        for ids in ((), (8,), (9, 8)):
            with self.subTest(ids=ids):
                self.assertIsNone(prepared(Tokenizer(ids=ids)))
        self.assertIsNone(prepared(Tokenizer(ids=(77,), bos_id=77)))
        self.assertEqual(prepared(Tokenizer(ids=(0,), end_id=0)).end_token_id, 0)

    def test_backend_negotiates_engine_and_excludes_other_reasoning_framing(self):
        ns = backend_symbols()
        mc = SimpleNamespace(tokenizer=Tokenizer(), reasoning_end_token="</think>",
                             harmony=False, muse_glimmer=False)
        for supported in (False, True):
            ns["AsyncJob"] = SimpleNamespace(set_token_budget=lambda *a, can_end=None: None) if supported else object
            for framing in (None, "harmony", "muse_glimmer"):
                for initial in (False, True):
                    with self.subTest(supported=supported, framing=framing, initial=initial):
                        mc.harmony = framing == "harmony"
                        mc.muse_glimmer = framing == "muse_glimmer"
                        plan = ns["prepare_reasoning_budget"](mc, 3, "</think>", initial)
                        self.assertEqual(plan is not None, supported and framing is None and initial)


async def collect(mc, params, *, stream, initial=True):
    queue = asyncio.Queue() if stream else None
    with patch.object(cc, "model") as model:
        model.container = mc
        result = await cc._chat_stream_collector(
            0, queue, "budget-request", "synthetic prompt", params, initial,
            streaming_mode=stream,
        )
    frames = []
    if queue is not None:
        while not queue.empty():
            frames.append(queue.get_nowait())
    return frames, result


def recording_backend(chunks, **overrides):
    mc = make_mc(chunks, reasoning_start_token="<think>", reasoning_end_token="</think>", **overrides)
    original = mc.stream_generate
    mc.options = []

    async def generate(*args, **kwargs):
        mc.options.append(kwargs)
        async for chunk in original(*args, **kwargs):
            yield chunk

    mc.stream_generate = generate
    mc.constrain_generation_output = Mock(return_value=True)
    return mc


class CollectorBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_plan_is_forwarded_and_never_injected_again_by_consumer(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                plan = prepared()
                mc = recording_backend(["reasoning words", "</think>", "answer"])
                mc.prepare_reasoning_budget = Mock(return_value=plan)
                frames, result = await collect(mc, make_request(tools=None, reasoning_budget_tokens=2), stream=stream)
                mc.prepare_reasoning_budget.assert_called_once_with(2, "</think>", True, parser=ANY)
                self.assertIs(mc.options[0]["reasoning_budget"], plan)
                mc.constrain_generation_output.assert_not_called()
                if stream:
                    self.assertFalse(any(isinstance(frame, Exception) for frame in frames))
                    self.assertEqual("".join(frame.get("delta_content", "") for frame in frames), "answer")
                    self.assertEqual("".join(frame.get("delta_reasoning_content", "") for frame in frames), "reasoning words")
                else:
                    self.assertEqual((result["content"], result["reasoning_content"]), ("answer", "reasoning words"))

    async def test_missing_capability_or_rejected_plan_retains_consumer_fallback(self):
        for has_method in (False, True):
            for stream in (False, True):
                with self.subTest(has_method=has_method, stream=stream):
                    mc = recording_backend(["reasoning words", "</think>", "answer"])
                    if has_method:
                        mc.prepare_reasoning_budget = Mock(return_value=None)
                    await collect(mc, make_request(tools=None, reasoning_budget_tokens=2), stream=stream)
                    self.assertNotIn("reasoning_budget", mc.options[0])
                    mc.constrain_generation_output.assert_called_once_with("budget-request", "</think>")

    async def test_no_budget_or_unknown_reasoning_format_does_not_negotiate(self):
        for stream in (False, True):
            for no_format in (False, True):
                with self.subTest(stream=stream, no_format=no_format):
                    mc = recording_backend(["answer"])
                    mc.prepare_reasoning_budget = Mock(side_effect=AssertionError("must not prepare"))
                    if no_format:
                        mc.reasoning = False
                    await collect(mc, make_request(tools=None, reasoning_budget_tokens=2 if no_format else None), stream=stream)
                    self.assertNotIn("reasoning_budget", mc.options[0])
                    mc.prepare_reasoning_budget.assert_not_called()
                    mc.constrain_generation_output.assert_not_called()

    async def test_continued_messages_retain_legacy_budget_path(self):
        for stream in (False, True):
            for options in ({"response_prefix": "prefix"}, {"continue_final_message": True}):
                with self.subTest(stream=stream, options=options):
                    mc = recording_backend(["reasoning words", "</think>", "answer"])
                    mc.prepare_reasoning_budget = Mock(side_effect=AssertionError("unknown initial prefix"))
                    await collect(mc, make_request(tools=None, reasoning_budget_tokens=2, **options), stream=stream)
                    mc.prepare_reasoning_budget.assert_not_called()
                    self.assertNotIn("reasoning_budget", mc.options[0])
                    mc.constrain_generation_output.assert_called_once_with("budget-request", "</think>")

    async def test_budget_resolution_and_injected_message_are_preserved(self):
        cases = [
            ({"reasoning_budget_tokens": 0, "reasoning": {"max_tokens": 3}}, 0),
            ({"reasoning_budget_tokens": -1, "reasoning": {"max_tokens": 3}}, 3),
            ({"reasoning_budget_tokens": -1, "reasoning": {"max_tokens": -1}}, 7),
        ]
        for options, expected in cases:
            with self.subTest(options=options):
                mc = recording_backend(["</think>", "answer"], reasoning_budget_tokens=7, reasoning_budget_message="default")
                mc.prepare_reasoning_budget = Mock(return_value=prepared(budget=expected))
                await collect(mc, make_request(tools=None, reasoning_budget_message="finish", **options), stream=False)
                mc.prepare_reasoning_budget.assert_called_once_with(expected, "finish</think>", True, parser=ANY)

    async def test_preparation_error_reaches_the_stream_queue_and_nonstream_caller(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                error = ValueError("synthetic preparation failure")
                mc = recording_backend([])
                mc.prepare_reasoning_budget = Mock(side_effect=error)
                frames, result = await collect(mc, make_request(reasoning_budget_tokens=2), stream=stream)
                self.assertEqual(mc.options, [])
                if stream:
                    self.assertEqual(frames, [error])
                else:
                    self.assertIs(result, error)

    async def test_content_phase_literal_tags_cannot_arm_a_native_reasoning_budget(self):
        literal = "<think>literal</think>"
        for stream in (False, True):
            with self.subTest(stream=stream):
                mc = recording_backend(pieces(xml("get_weather", [("city", literal)])))
                tokenizer = Tokenizer()
                mc.prepare_reasoning_budget = lambda n, text, initial, parser=None: prepare_native_reasoning_budget(
                    tokenizer, n, text, initial_reasoning=initial, end_token="</think>", supported=True,
                    parser=parser)
                frames, result = await collect(mc, make_request(reasoning_budget_tokens=0), stream=stream, initial=False)
                self.assertNotIn("reasoning_budget", mc.options[0])
                self.assertEqual(tokenizer.encode_calls, [])
                mc.constrain_generation_output.assert_not_called()
                if not stream:
                    self.assertEqual(result["tool_calls"][0]["function"]["arguments"], '{"city": "<think>literal</think>"}')
                else:
                    self.assertFalse(any(isinstance(frame, Exception) for frame in frames))
                    self.assertEqual(frames[-1]["finish_reason"], "tool_calls")

    async def test_native_reasoning_close_preserves_forced_tool_contract(self):
        for stream in (False, True):
            for choice in ("required", {"type": "function", "function": {"name": "get_weather"}}):
                with self.subTest(stream=stream, choice=choice):
                    raw = "thought</think>" + xml("get_weather", [("city", "Paris")])
                    mc = recording_backend(pieces(raw))
                    mc.prepare_reasoning_budget = Mock(return_value=prepared())
                    frames, result = await collect(mc, make_request(tool_choice=choice, reasoning_budget_tokens=2), stream=stream)
                    mc.constrain_generation_output.assert_not_called()
                    if stream:
                        self.assertFalse(any(isinstance(frame, Exception) for frame in frames))
                        self.assertEqual(frames[-1]["finish_reason"], "tool_calls")
                    else:
                        self.assertEqual(result["finish_reason"], "tool_calls")
                        self.assertEqual(result["tool_calls"][0]["function"]["arguments"], '{"city": "Paris"}')


class PhaseJob:
    def __init__(self, *, suspended, pending=False):
        self.job = SimpleNamespace(filters_suspended=suspended)
        self.pending = pending
        self.filters = []
        self.calls = []
        self.sampler = None
        self.banned = None

    def set_filters(self, filters):
        if self.pending:
            raise ValueError("forced output remains")
        self.calls.append(("filters", filters))
        self.filters = filters
        for item in filters:
            item.position = 0
        self.job.filters_suspended = False

    def set_sampler(self, sampler):
        self.calls.append(("sampler", sampler))
        self.sampler = sampler

    def set_banned_strings(self, banned):
        self.calls.append(("banned", banned))
        self.banned = banned


def phase_container(*, suspended, pending=False, engine_trigger=True):
    ns = backend_symbols()
    job = PhaseJob(suspended=suspended, pending=pending)
    content_filter = SimpleNamespace(trigger_token=9, position=0)
    phases = ns["JobPhases"](
        job=job, content_sampler=object(), reasoning_sampler=object(),
        content_banned=["content banned"], reasoning_banned=["thought banned"],
        content_filters=[content_filter], reasoning=True, engine_trigger=engine_trigger,
    )
    mc = SimpleNamespace(active_job_ids={"id": job}, job_phases={"id": phases})
    mc.set_generation_phase = MethodType(ns["set_generation_phase"], mc)
    return mc, phases, job, content_filter


class ProducerPhaseTests(unittest.TestCase):
    def test_forced_close_restores_all_settings_before_content_without_later_reset(self):
        mc, phases, job, content_filter = phase_container(suspended=True)
        callback = producer_phase_end_callback(mc, "id")
        self.assertIsNone(callback(job.job))
        self.assertFalse(phases.reasoning)
        self.assertIsNone(content_filter.trigger_token)
        self.assertFalse(phases.engine_trigger)
        self.assertEqual([kind for kind, _ in job.calls], ["filters", "sampler", "banned"])
        self.assertIs(job.sampler, phases.content_sampler)
        self.assertEqual(job.banned, phases.content_banned)
        content_filter.position = 3  # Producer has consumed three answer tokens.
        self.assertTrue(mc.set_generation_phase("id", False))  # Delayed consumer sees closing tag.
        self.assertEqual(content_filter.position, 3)
        self.assertEqual(len(job.calls), 3)

    def test_natural_end_respects_engine_trigger_and_preserves_filter_progress(self):
        mc, phases, job, content_filter = phase_container(suspended=False)
        content_filter.position = 1
        producer_phase_end_callback(mc, "id")(job.job)
        self.assertFalse(phases.reasoning)
        self.assertEqual(content_filter.position, 1)
        self.assertEqual([kind for kind, _ in job.calls], ["sampler", "banned"])
        self.assertTrue(mc.set_generation_phase("id", False))
        self.assertEqual(len(job.calls), 2)

    def test_native_natural_end_attaches_previously_detached_filters_exactly_once(self):
        mc, phases, job, content_filter = phase_container(suspended=False, engine_trigger=False)
        producer_phase_end_callback(mc, "id")(job.job)
        self.assertFalse(phases.reasoning)
        self.assertIsNone(content_filter.trigger_token)
        self.assertEqual([kind for kind, _ in job.calls], ["filters", "sampler", "banned"])
        content_filter.position = 2
        self.assertTrue(mc.set_generation_phase("id", False))
        self.assertEqual(content_filter.position, 2)
        self.assertEqual(len(job.calls), 3)

    def test_pending_forced_output_fails_closed_without_switching_the_sampler(self):
        mc, phases, job, _ = phase_container(suspended=True, pending=True)
        with self.assertRaisesRegex(RuntimeError, "content settings"):
            producer_phase_end_callback(mc, "id")(job.job)
        self.assertTrue(phases.reasoning)
        self.assertEqual(job.calls, [])

    def test_missing_or_replaced_request_job_cannot_change_another_job(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                mc, phases, job, _ = phase_container(suspended=True)
                if missing:
                    mc.active_job_ids.clear()
                with self.assertRaisesRegex(RuntimeError, "inactive generation job"):
                    producer_phase_end_callback(mc, "id")(job.job if missing else object())
                self.assertTrue(phases.reasoning)
                self.assertEqual(job.calls, [])

    def test_identical_phase_settings_need_no_registered_swap(self):
        native = object()
        mc = SimpleNamespace(active_job_ids={"id": SimpleNamespace(job=native)}, job_phases={})
        mc.set_generation_phase = MethodType(backend_symbols()["set_generation_phase"], mc)
        self.assertIsNone(producer_phase_end_callback(mc, "id")(native))


class BackendArmingTests(unittest.IsolatedAsyncioTestCase):
    def backend(self, setter_error=None, invalid_guard=False):
        ns = backend_symbols()
        events, jobs = [], []
        sampler = object()

        class FakeAsyncJob:
            def __init__(self, *args, **kwargs):
                self.job = SimpleNamespace(
                    filters_suspended=False, prefix_token=None, new_tokens=0, checkpoint=None,
                    sequences=[] if invalid_guard else [SimpleNamespace(sequence_ids=AcceptedIds())],
                )
                self.cancelled = False
                self.queue = asyncio.Queue()
                self.budget = None
                self.creation = kwargs
                jobs.append(self)
                events.append("enqueue")
                async def producer_ready():
                    events.append(("producer_ready", self.budget is not None, self.cancelled))
                asyncio.create_task(producer_ready())

            def set_token_budget(self, max_tokens, output, **kwargs):
                events.append("set_budget")
                self.budget = (max_tokens, output, kwargs)
                if setter_error is not None:
                    raise setter_error

            async def cancel(self):
                events.append("cancel")
                self.cancelled = True
                await asyncio.sleep(0)

            async def __aiter__(self):
                yield {"stage": "streaming", "eos": True, "text": ""}

        class Grammar:
            def __init__(self):
                self.filters = []
            def add_grammar_filter(self, *args, **kwargs):
                self.filters.append(SimpleNamespace(trigger_token=kwargs.get("trigger_token_id")))

        class Disconnect:
            async def add_cleanup_task(self, *args):
                events.append("register_cleanup")
                await asyncio.sleep(0)
            async def poll(self):
                pass
            async def finish(self, *args):
                events.append("finish")

        ns.update({
            "AsyncJob": FakeAsyncJob,
            "ExllamaV3SamplerBuilder": SimpleNamespace(from_params=lambda *a: SimpleNamespace(
                build=lambda *a: sampler, settings=[])),
            "ExLlamaV3Grammar": Grammar,
            "unwrap": lambda value, default: default if value is None else value,
            "validate_context_requirements": lambda *a: None,
            "reasoning_tag_conflicts": lambda *a: [],
            "format_settings": lambda *a: "",
            "log_prompt": lambda *a, **k: None,
            "log_request_start": lambda *a, **k: None,
            "log_generation_params": lambda *a, **k: None,
            "log_metrics": lambda *a, **k: None,
            "status_display": SimpleNamespace(add_job=lambda *a: SimpleNamespace(), remove_job=lambda *a: None),
        })
        other = object()
        mc = SimpleNamespace(
            active_job_ids={"other": other}, job_phases={}, tokenizer=Tokenizer(),
            generator=SimpleNamespace(generator=SimpleNamespace(recurrent_cache=None)),
            max_seq_len=4096, cache=SimpleNamespace(max_num_tokens=4096),
            hf_model=SimpleNamespace(add_bos_token=lambda: False, eos_tokens=lambda: [99]),
            config=SimpleNamespace(eos_token_id_list=[]),
            reasoning=True, reasoning_end_token="</think>",
            job_max_rq_tokens=lambda n: n,
            loaded=True, load_condition=asyncio.Condition(), load_lock=asyncio.Lock(),
            handle_finish_chunk=lambda *a: {"finish_reason": "stop"},
        )
        for name in ("stream_generate", "generate_gen", "set_generation_phase"):
            setattr(mc, name, MethodType(ns[name], mc))
        params = SimpleNamespace(
            temperature=0, stop=[], add_bos_token=False, max_tokens=10, min_tokens=0,
            banned_strings=[], token_healing=False, logprobs=False, top_logprobs=0,
            json_schema=None, regex_pattern=None, grammar_string='start: "answer"',
            model_dump=lambda **k: {}, reasoning_params=lambda: None,
            param_source=lambda key: "default", get_stop_on_loop=lambda: None,
        )
        return mc, params, Disconnect(), events, jobs, other

    async def test_budget_is_armed_before_first_producer_turn_and_cleanup_is_unchanged(self):
        for budget in (None, prepared(budget=0), prepared(budget=24),
                       prepared(budget=0, parser=qwen_parser())):
            with self.subTest(budget=None if budget is None else budget.max_tokens):
                mc, params, disconnect, events, jobs, other = self.backend()
                output = [item async for item in mc.stream_generate(
                    "id", "prompt", params, disconnect, reasoning_phase=True, reasoning_budget=budget)]
                self.assertEqual(output, [{"finish_reason": "stop"}])
                self.assertEqual(mc.active_job_ids, {"other": other})
                self.assertEqual(mc.job_phases, {})
                expected = budget is not None
                self.assertIn(("producer_ready", expected, False), events)
                self.assertEqual("set_budget" in events, expected)
                self.assertNotIn("cancel", events)
                if expected:
                    self.assertEqual(jobs[0].creation["filters"], [])
                else:
                    self.assertEqual(len(jobs[0].creation["filters"]), 1)
                    self.assertEqual(jobs[0].creation["filters"][0].trigger_token, 9)
                if expected:
                    self.assertLess(events.index("set_budget"), events.index("register_cleanup"))
                    self.assertEqual(jobs[0].budget[0], budget.max_tokens)
                    self.assertIs(jobs[0].budget[1], budget.output_ids)
                    self.assertEqual(jobs[0].budget[2]["end_token_id"], budget.end_token_id)
                    if budget.parser is not None:
                        guard = jobs[0].budget[2]["can_end"]
                        self.assertIsInstance(guard, ReasoningBoundaryGuard)
                        self.assertTrue(guard(jobs[0].job))

    async def test_guard_preparation_failure_cancels_its_own_enqueued_job(self):
        mc, params, disconnect, events, jobs, other = self.backend(invalid_guard=True)
        with self.assertRaisesRegex(ValueError, "one sequence"):
            _ = [item async for item in mc.stream_generate(
                "id", "prompt", params, disconnect, reasoning_phase=True,
                reasoning_budget=prepared(parser=qwen_parser()))]
        self.assertEqual(mc.active_job_ids, {"other": other})
        self.assertEqual(mc.job_phases, {})
        self.assertEqual(events.count("cancel"), 1)
        self.assertNotIn("set_budget", events)
        self.assertTrue(jobs[0].cancelled)

    async def test_setter_failure_or_cancellation_cancels_only_its_own_enqueued_job(self):
        for error in (ValueError("invalid budget"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                mc, params, disconnect, events, jobs, other = self.backend(error)
                with self.assertRaises(type(error)) as caught:
                    _ = [item async for item in mc.stream_generate(
                        "id", "prompt", params, disconnect, reasoning_phase=True, reasoning_budget=prepared())]
                self.assertIs(caught.exception, error)
                self.assertEqual(mc.active_job_ids, {"other": other})
                self.assertEqual(mc.job_phases, {})
                self.assertTrue(jobs[0].cancelled)
                self.assertEqual(events.count("cancel"), 1)
                self.assertNotIn("register_cleanup", events)
                self.assertIn(("producer_ready", True, True), events)


def qwen_parser(**kwargs):
    options = dict(reasoning_start="<think>", reasoning_end="</think>",
                   tool_start="<tool_call>", tool_end="</tool_call>",
                   start_in_reasoning=True, tool_calls_in_reasoning=True)
    options.update(kwargs)
    return Qwen3CoderStreamParser(**options)


class AcceptedIds:
    def __init__(self, values=(0, 0, 0)):
        self.values = list(values)
        self.reads = []
        self.device = "cpu"

    def __len__(self):
        return len(self.values)

    def torch_slice(self, start, end):
        self.reads.append((start, end))
        return SimpleNamespace(device=SimpleNamespace(type=self.device), reshape=lambda *_:
                               SimpleNamespace(tolist=lambda: self.values[start:end]))


def guarded(pieces, parser=None, healing_prefix=None):
    tokenizer = SimpleNamespace(get_id_to_piece_list=lambda _: ["PROMPT", *pieces])
    sequence = AcceptedIds()
    native = SimpleNamespace(sequences=[SimpleNamespace(sequence_ids=sequence)],
                             checkpoint=None, new_tokens=0, prefix_token=None)
    if healing_prefix is not None:
        native.prefix_token = [SimpleNamespace(item=lambda: healing_prefix)]
        native.new_tokens = -1
    parser = parser or qwen_parser()
    return native, ReasoningBoundaryGuard(native, tokenizer, parser), parser, sequence


def append(native, token_id, checkpoint_offset=None):
    native.sequences[0].sequence_ids.values.append(token_id)
    native.new_tokens += 1
    native.checkpoint = None if checkpoint_offset is None else {"offset": checkpoint_offset}


class BoundaryGuardTests(unittest.TestCase):
    def test_wrapped_call_and_partial_prefixes_defer_while_literal_native_end_is_data(self):
        parts = ["thought ", "<", "tool", "_call", ">", "<func", "tion=f>",
                 "<param", "eter=value>\n", "</think>", "\n</para", "meter>",
                 "</function>", "</tool_call>", " after", "</think>"]
        native, guard, parser, seq = guarded(parts)
        observed = []
        for token_id in range(1, len(parts) + 1):
            append(native, token_id)
            observed.append(guard(native))
            if token_id == 10:
                self.assertTrue(parser.in_reasoning)
                self.assertTrue(parser._in_parameter)
        self.assertEqual(observed, [True, *([False] * 12), True, True, True])
        self.assertFalse(parser.in_reasoning)
        self.assertEqual(seq.reads, [(index, index + 1) for index in range(3, 3 + len(parts))])

    def test_bare_function_prefix_and_literal_inner_wrapper_stay_protected(self):
        parts = ["<func", "tion=f><parameter=x>", "<tool_call></tool_call></think>",
                 "</parameter>", "</function>"]
        native, guard, parser, _ = guarded(parts)
        safe = []
        for token_id in range(1, len(parts) + 1):
            append(native, token_id)
            safe.append(guard(native))
        self.assertEqual(safe, [False, False, False, False, True])
        self.assertTrue(parser.in_reasoning)

    def test_due_budget_scans_only_unseen_tokens_and_has_constant_history_without_hold(self):
        parts = ["thought", "<tool_call><function=f><parameter=x>", "data", "</parameter></function></tool_call>"]
        native, guard, _, seq = guarded(parts)
        for _ in range(24):
            append(native, 1)
        self.assertTrue(guard(native))
        self.assertEqual(seq.reads, [(3, 27)])
        append(native, 2)
        self.assertFalse(guard(native))
        for _ in range(1024):
            append(native, 3)
            self.assertFalse(guard(native))
            self.assertEqual(len(guard._states), 1)
        append(native, 4)
        self.assertTrue(guard(native))
        self.assertEqual(sum(end - start for start, end in seq.reads), len(seq.values) - 3)

    def test_banned_rewind_restores_partial_opener_and_parameter_state(self):
        for prefix in ("thought ", "<tool_call><function=f><parameter=x>"):
            with self.subTest(prefix=prefix):
                parts = [prefix, "<tool", "_call></think>", " replacement"]
                native, guard, parser, seq = guarded(parts)
                append(native, 1)
                guard(native)
                before = parser.checkpoint()
                target = len(seq)
                append(native, 2, 1)
                guard(native)
                append(native, 3, 2)
                guard(native)
                self.assertLessEqual(len(guard._states), 3)
                seq.values = seq.values[:target]
                native.new_tokens -= 2
                native.checkpoint = {"offset": 0}
                guard(native)  # Engine's explicit synchronization at the rewind.
                self.assertEqual(parser.checkpoint(), before)
                append(native, 4)
                guard(native)
                fresh = qwen_parser()
                fresh.feed(prefix)
                fresh.feed(parts[3])
                self.assertEqual(parser.checkpoint(), fresh.checkpoint())

    def test_first_guard_read_retains_only_active_checkpoint_history_for_a_later_rewind(self):
        native, guard, parser, seq = guarded(["a", "<tool", "_call>", " replacement"])
        for _ in range(21):
            append(native, 1)
        target = len(seq)
        append(native, 2, 1)
        append(native, 3, 2)
        self.assertFalse(guard(native))
        self.assertEqual(set(guard._states), {target, target + 1, target + 2})
        seq.values = seq.values[:target]
        native.new_tokens -= 2
        native.checkpoint = {"offset": 0}
        self.assertTrue(guard(native))
        self.assertFalse(parser.in_tool)
        append(native, 4)
        self.assertTrue(guard(native))
        self.assertEqual(len(guard._states), 1)

    def test_requeue_keeps_original_output_offset_and_never_reparses_prompt(self):
        native, guard, parser, seq = guarded(["thought<tool_", "call><function=f>", "</function></tool_call>"])
        append(native, 1)
        self.assertFalse(guard(native))
        new_sequence = AcceptedIds(seq.values)
        native.sequences = [SimpleNamespace(sequence_ids=new_sequence)]
        native.new_tokens = 0
        native.rq_new_tokens = 1
        append(native, 2)
        self.assertFalse(guard(native))
        append(native, 3)
        self.assertTrue(guard(native))
        self.assertTrue(parser.in_reasoning)
        self.assertEqual(new_sequence.reads, [(4, 5), (5, 6)])

    def test_healing_prefix_is_excluded_but_new_partial_markup_is_tracked(self):
        native, guard, parser, _ = guarded(["PROMPT<tool_", "call><function=f>", "</function></tool_call>"], healing_prefix=0)
        append(native, 1)
        self.assertEqual(native.new_tokens, 0)
        self.assertFalse(guard(native))
        self.assertEqual(parser._pending, "<tool_")
        append(native, 2)
        self.assertFalse(guard(native))
        append(native, 3)
        self.assertTrue(guard(native))

    def test_guard_rejects_wrong_job_missing_rewind_history_and_non_cpu_ids(self):
        native, guard, _, seq = guarded(["data"])
        with self.assertRaisesRegex(RuntimeError, "another job"):
            guard(SimpleNamespace())
        append(native, 1)
        guard(native)
        seq.values.pop()
        with self.assertRaisesRegex(RuntimeError, "checkpoint was not retained"):
            guard(native)
        native, guard, _, seq = guarded(["data"])
        append(native, 1)
        seq.device = "cuda"
        with self.assertRaisesRegex(ValueError, "CPU token IDs"):
            guard(native)

    def test_no_tool_parsing_in_reasoning_preserves_that_existing_policy(self):
        native, guard, parser, _ = guarded(
            ["<tool_call><function=f><parameter=x>", "</think>"],
            parser=qwen_parser(tool_calls_in_reasoning=False))
        append(native, 1)
        self.assertTrue(guard(native))
        self.assertFalse(parser.in_tool)
        append(native, 2)
        self.assertTrue(guard(native))
        self.assertFalse(parser.in_reasoning)


class ParserCheckpointTests(unittest.TestCase):
    def test_restored_partial_prefix_parameter_and_whitespace_states_match_uninterrupted_router(self):
        for parser_type in (TagStreamParser, Qwen3CoderStreamParser):
            for prefix in ("thought<tool_", "<tool_call><function=f><parameter=x>\n<think>",
                           "<function=f><parameter=x>\n", "thought</think>\n "):
                for chunk in (1, 7, 31):
                    with self.subTest(parser_type=parser_type.__name__, prefix=prefix, chunk=chunk):
                        kwargs = dict(reasoning_start="<think>", reasoning_end="</think>",
                                      tool_start="<tool_call>", tool_end="</tool_call>", start_in_reasoning=True)
                        parser, fresh = parser_type(**kwargs), parser_type(**kwargs)
                        for piece in pieces(prefix, chunk):
                            parser.feed(piece)
                            fresh.feed(piece)
                        state = parser.checkpoint()
                        parser.feed("wrong branch</parameter></function></tool_call></think>")
                        parser.restore_checkpoint(state)
                        suffix = "x</parameter></function></tool_call></think> answer"
                        actual, expected = [], []
                        for piece in pieces(suffix, chunk):
                            actual.extend(parser.feed(piece))
                            expected.extend(fresh.feed(piece))
                        self.assertEqual(actual, expected)
                        self.assertEqual(parser.checkpoint(), fresh.checkpoint())

    def test_preparation_copies_parser_and_declines_ambiguous_injected_markup(self):
        parser = qwen_parser()
        plan = prepared(parser=parser)
        self.assertIsNot(plan.parser, parser)
        plan.parser.feed("<tool_call>")
        self.assertFalse(parser.in_tool)
        self.assertIsNone(prepare_native_reasoning_budget(
            Tokenizer(ids=(9,)), 2, "<tool_call></think>", initial_reasoning=True,
            end_token="</think>", supported=True, parser=parser))

    def test_pre_guard_engine_api_does_not_claim_safe_native_support(self):
        old = SimpleNamespace(set_token_budget=lambda max_tokens, output, end_token_id, on_end=None: None)
        new = SimpleNamespace(set_token_budget=lambda max_tokens, output, end_token_id, on_end=None, can_end=None: None)
        self.assertFalse(supports_native_reasoning_budget(old))
        self.assertTrue(supports_native_reasoning_budget(new))


if __name__ == "__main__":
    unittest.main()
