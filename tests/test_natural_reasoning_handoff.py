"""Natural-only phase handoff preserves default limits and literal tool data.

Runs the real collector/backend methods with CPU tokenizer/job fixtures. Native
accepted-token and Filter.feed composition is separately checked with the engine.
"""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, Mock

from backends.exllamav3.reasoning import (
    NativeReasoningBudget, ReasoningBoundaryGuard, prepare_native_reasoning_budget,
    supports_native_reasoning_budget,
)
from tests import test_reasoning_budget as budget_fixture
from tests.test_reasoning_budget import (
    Tokenizer, backend_symbols, collect, make_request,
    qwen_parser, recording_backend,
)
from tests.test_qwen_tool_contract import xml


def natural_plan(tokenizer=None, **kwargs):
    return prepare_native_reasoning_budget(
        tokenizer or Tokenizer(), None, None, initial_reasoning=True,
        end_token='</think>', supported=True, parser=kwargs.pop('parser', qwen_parser()), **kwargs)


class NaturalPreparationTests(unittest.TestCase):
    def test_natural_only_plan_has_no_output_and_never_encodes_an_injection(self):
        tokenizer = Tokenizer()
        parser = qwen_parser()
        plan = natural_plan(tokenizer, parser=parser)
        self.assertIsInstance(plan, NativeReasoningBudget)
        self.assertIsNone(plan.max_tokens)
        self.assertIsNone(plan.output_ids)
        self.assertEqual(plan.end_token_id, 9)
        self.assertEqual(tokenizer.encode_calls, [])
        self.assertIsNot(plan.parser, parser)
        plan.parser.feed('<tool_call>')
        self.assertFalse(parser.in_tool)

    def test_natural_only_requires_checkpoint_guard_and_never_accepts_forced_text(self):
        for parser in (None, SimpleNamespace(), SimpleNamespace(checkpoint=lambda: ())):
            with self.subTest(parser=parser):
                self.assertIsNone(natural_plan(parser=parser))
        tokenizer = Tokenizer()
        with self.assertRaisesRegex(ValueError, 'cannot include forced output'):
            prepare_native_reasoning_budget(tokenizer, None, 'force me</think>',
                initial_reasoning=True, end_token='</think>', supported=True, parser=qwen_parser())
        self.assertEqual(tokenizer.encode_calls, [])

    def test_natural_capability_is_explicit_and_finite_capability_is_unchanged(self):
        for marker in (None, False, True, 'yes'):
            with self.subTest(marker=marker):
                old = SimpleNamespace(set_token_budget=lambda n, out, *, end_token_id, on_end=None, can_end=None: None)
                if marker is not None:
                    old.supports_natural_token_budget = marker
                self.assertTrue(supports_native_reasoning_budget(old))
                self.assertEqual(supports_native_reasoning_budget(old, natural_only=True), marker is True)
        self.assertFalse(supports_native_reasoning_budget(SimpleNamespace(supports_natural_token_budget=True), natural_only=True))

    def test_actual_backend_declines_none_on_9c_and_accepts_only_new_capability(self):
        ns = backend_symbols()
        mc = SimpleNamespace(tokenizer=Tokenizer(), reasoning_end_token='</think>', harmony=False, muse_glimmer=False)
        for capability in (False, True):
            ns['AsyncJob'] = SimpleNamespace(
                set_token_budget=lambda n, out, *, end_token_id, on_end=None, can_end=None: None,
                supports_natural_token_budget=capability)
            for initial in (False, True):
                with self.subTest(capability=capability, initial=initial):
                    plan = ns['prepare_reasoning_budget'](mc, None, None, initial, parser=qwen_parser())
                    self.assertEqual(plan is not None, capability and initial)
            self.assertIsNotNone(ns['prepare_reasoning_budget'](mc, 2, '</think>', True, parser=qwen_parser()))
        self.assertEqual(len(mc.tokenizer.encode_calls), 2)  # Only the two finite plans encoded output.


class NaturalCollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_omitted_budget_is_observed_without_cutoff_or_message_injection(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                mc = recording_backend(['thought', '</think>', 'answer'], reasoning_budget_message='unused configured text')
                plan = natural_plan()
                mc.prepare_reasoning_budget = Mock(return_value=plan)
                params = make_request(tools=None, max_tokens=17, reasoning_budget_message='unused request text')
                frames, result = await collect(mc, params, stream=stream)
                mc.prepare_reasoning_budget.assert_called_once_with(None, None, True, parser=ANY)
                self.assertIs(mc.options[0]['reasoning_budget'], plan)
                self.assertIsNone(params.reasoning_budget_tokens)
                self.assertEqual(params.max_tokens, 17)
                mc.constrain_generation_output.assert_not_called()
                if stream:
                    self.assertEqual(''.join(f.get('delta_content', '') for f in frames), 'answer')
                else:
                    self.assertEqual(result['content'], 'answer')

    async def test_literal_end_in_initial_reasoning_tool_keeps_router_policy_in_both_modes(self):
        literal = '</think>literal'
        raw = 'thought ' + xml('echo', [('text', literal)]) + '</think>answer'
        tools = [{'type': 'function', 'function': {'name': 'echo', 'parameters': {
            'type': 'object', 'properties': {'text': {'type': 'string'}},
            'required': ['text'], 'additionalProperties': False}}}]
        for stream in (False, True):
            with self.subTest(stream=stream):
                mc = recording_backend([raw[i:i+3] for i in range(0, len(raw), 3)])
                plans = []
                def prepare(n, text, initial, parser):
                    plan = prepare_native_reasoning_budget(Tokenizer(), n, text,
                        initial_reasoning=initial, end_token='</think>', supported=True, parser=parser)
                    plans.append(plan)
                    return plan
                mc.prepare_reasoning_budget = prepare
                frames, result = await collect(mc, make_request(tools=tools), stream=stream)
                self.assertEqual(len(plans), 1)
                self.assertTrue(plans[0].parser.in_reasoning)
                self.assertFalse(plans[0].parser.in_tool)
                mc.constrain_generation_output.assert_not_called()
                if stream:
                    calls = [call for frame in frames for call in frame.get('delta_tool_calls', [])]
                    args = ''.join(call.get('function', {}).get('arguments', '') for call in calls)
                else:
                    args = result['tool_calls'][0]['function']['arguments']
                self.assertEqual(json.loads(args), {'text': literal})

    async def test_older_engine_and_unsupported_plan_keep_existing_unlimited_behavior(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                mc = recording_backend(['thought', '</think>', 'answer'])
                mc.prepare_reasoning_budget = Mock(return_value=None)
                await collect(mc, make_request(tools=None), stream=stream)
                self.assertNotIn('reasoning_budget', mc.options[0])
                mc.prepare_reasoning_budget.assert_called_once_with(None, None, True, parser=ANY)
                mc.constrain_generation_output.assert_not_called()

    async def test_content_continuations_and_other_formats_do_not_arm_natural_observer(self):
        for stream in (False, True):
            for initial, options in ((False, {}), (True, {'continue_final_message': True}),
                                     (True, {'response_prefix': 'continued'})):
                with self.subTest(stream=stream, initial=initial, options=options):
                    mc = recording_backend(['answer'])
                    mc.prepare_reasoning_budget = Mock(side_effect=AssertionError('not an initial reasoning phase'))
                    await collect(mc, make_request(tools=None, **options), stream=stream, initial=initial)
                    self.assertNotIn('reasoning_budget', mc.options[0])
                    mc.prepare_reasoning_budget.assert_not_called()
                    mc.constrain_generation_output.assert_not_called()

    async def test_client_output_constraints_and_schema_values_remain_unchanged(self):
        for stream in (False, True):
            for constraint in ({'grammar_string': 'start: "answer"'},
                               {'json_schema': {'type': 'string', 'const': 'answer'}}):
                with self.subTest(stream=stream, constraint=constraint):
                    mc = recording_backend(['thought', '</think>', 'answer'])
                    mc.prepare_reasoning_budget = Mock(return_value=natural_plan())
                    params = make_request(tools=None, **deepcopy(constraint))
                    before = {key: deepcopy(getattr(params, key)) for key in constraint}
                    await collect(mc, params, stream=stream)
                    self.assertEqual({key: getattr(params, key) for key in constraint}, before)
                    mc.constrain_generation_output.assert_not_called()


class NaturalBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_natural_handoff_detaches_raw_trigger_before_first_await(self):
        mc, params, disconnect, events, jobs, other = budget_fixture.BackendArmingTests().backend()
        plan = natural_plan()
        _ = [row async for row in mc.stream_generate('id', 'prompt', params, disconnect,
            reasoning_phase=True, reasoning_budget=plan)]
        job = jobs[0]
        self.assertEqual(job.creation['filters'], [])
        self.assertEqual(job.creation['max_new_tokens'], 10)
        self.assertEqual(job.budget[:2], (None, None))
        self.assertIsInstance(job.budget[2]['can_end'], ReasoningBoundaryGuard)
        self.assertLess(events.index('set_budget'), events.index('register_cleanup'))
        self.assertIn(('producer_ready', True, False), events)
        self.assertEqual(mc.active_job_ids, {'other': other})
        self.assertEqual(mc.job_phases, {})

    async def test_identical_phase_settings_keep_default_job_uninstrumented(self):
        mc, params, disconnect, events, jobs, other = budget_fixture.BackendArmingTests().backend()
        params.grammar_string = None
        _ = [row async for row in mc.stream_generate('id', 'prompt', params, disconnect,
            reasoning_phase=True, reasoning_budget=natural_plan())]
        self.assertIsNone(jobs[0].budget)
        self.assertEqual(jobs[0].creation['max_new_tokens'], 10)
        self.assertEqual(jobs[0].creation['filters'], [])
        self.assertNotIn('set_budget', events)
        self.assertIn(('producer_ready', False, False), events)
        self.assertEqual(mc.active_job_ids, {'other': other})
        self.assertEqual(mc.job_phases, {})


if __name__ == '__main__':
    unittest.main()
