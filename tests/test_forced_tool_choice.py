"""Real llguidance grammar and fail-closed API contracts for forced Qwen calls."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from common.errors import ToolCallParseError
from endpoints.OAI.utils import chat_completion as cc
from endpoints.OAI.utils.tool_choice import (
    ForcedToolChoice,
    prepare_forced_tool_choice,
    resolve_forced_tool_choice,
)
from tests.test_qwen_tool_contract import xml
from tests.test_tool_delta_streaming import make_mc, make_request, pieces, run_collector


def named(name):
    return {"type": "function", "function": {"name": name}}


def request(**kwargs):
    kwargs.setdefault("tool_choice", "required")
    kwargs.setdefault(
        "tools",
        [
            {"type": "function", "function": {"name": "ping"}},
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            },
        ],
    )
    return make_request(**kwargs)


class ForcedChoiceRequestTests(unittest.TestCase):
    def assert_request_error(self, data, text, tool_format="qwen3_coder"):
        with self.assertRaises(HTTPException) as caught:
            prepare_forced_tool_choice(data, tool_format)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn(text, caught.exception.detail)

    def test_auto_none_and_unset_leave_custom_grammar_untouched(self):
        for choice in [None, "auto", "none"]:
            data = request(tool_choice=choice, grammar_string='start: "hi"')
            self.assertIsNone(prepare_forced_tool_choice(data, "harmony"))
            self.assertEqual(data.grammar_string, 'start: "hi"')

    def test_required_all_declared_names(self):
        data = request()
        choice = prepare_forced_tool_choice(data, "qwen3_coder")
        self.assertEqual(choice.names, ("ping", "get_weather"))
        self.assertTrue(choice.parallel)
        self.assertEqual(data.grammar_string, choice.grammar())

    def test_named_choice_restricts_name(self):
        data = request(tool_choice=named("ping"), parallel_tool_calls=False)
        choice = prepare_forced_tool_choice(data, "qwen3_5")
        self.assertEqual(choice.names, ("ping",))
        self.assertFalse(choice.parallel)
        self.assertNotIn("get_weather", data.grammar_string)

    def test_legacy_functions(self):
        data = request(tools=None, functions=[{"name": "legacy"}])
        self.assertEqual(resolve_forced_tool_choice(data).names, ("legacy",))

    def test_missing_unknown_duplicate_and_ambiguous_names_rejected(self):
        self.assert_request_error(request(tools=[]), "at least one")
        self.assert_request_error(request(tool_choice=named("absent")), "not present")
        self.assert_request_error(request(tools=[named("same"), named("same")]), "unique")
        for name in ["", "a b", "a>b", "a<b", "a=b", "x\ny"]:
            with self.subTest(name=name):
                self.assert_request_error(request(tools=[named(name)]), "function names")
        self.assert_request_error(request(tools=None, functions=[{"name": 42}]), "function names")

    def test_other_formats_rejected_instead_of_ignoring_choice(self):
        self.assert_request_error(request(), "tool_choice='auto'", "harmony")
        self.assert_request_error(request(), "tool_choice='auto'", None)

    def test_competing_grammar_constraints_rejected(self):
        for key, value in [
            ("grammar_string", 'start: "hello"'),
            ("regex_pattern", "[a-z]+"),
            ("json_schema", {"type": "object"}),
            ("response_format", {"type": "json_object"}),
            ("response_format", {"type": "json_schema", "json_schema": {"type": "object"}}),
        ]:
            with self.subTest(key=key):
                self.assert_request_error(request(**{key: value}), key)

    def test_continuation_or_response_prefix_rejected(self):
        self.assert_request_error(request(response_prefix="<tool_call>"), "response_prefix")
        self.assert_request_error(request(continue_final_message=True), "continue_final_message")


class ForcedChoiceGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper
        except ImportError:
            raise unittest.SkipTest("llguidance is required for real grammar acceptance tests")

        # CPU-only, exact byte tokens: no network, downloaded model or CUDA.
        class ByteTokenizer:
            eos_token_id = 256
            bos_token_id = None
            tokens = [bytes([i]) for i in range(256)] + [b"<eos>"]
            special_token_ids = [256]

            def __call__(self, text):
                return list(text if isinstance(text, bytes) else text.encode("utf-8"))

        cls.matcher = LLMatcher
        cls.tokenizer = LLTokenizer(TokenizerWrapper(ByteTokenizer()), slices=[])

    def accepts(self, text, names=("ping", "get_weather"), parallel=True):
        grammar = ForcedToolChoice(names, parallel).grammar()
        self.assertEqual(self.matcher.validate_grammar(grammar), "")
        matcher = self.matcher(self.tokenizer, grammar, log_level=0)
        valid_prefix = matcher.consume_tokens(self.tokenizer.tokenize_str(text))
        return valid_prefix and matcher.is_accepting()

    def test_required_accepts_complete_single_and_parallel_calls(self):
        a = xml("ping")
        b = xml("get_weather", [("city", "Paris")])
        for text in [a, b, "\n\n" + a + "\n", a + b, b + "\n" + a]:
            with self.subTest(text=text):
                self.assertTrue(self.accepts(text))

    def test_rejects_no_call_wrong_name_and_incomplete_structure(self):
        for text in [
            "",
            "\n",
            "READY",
            "Please call the tool. " + xml("ping"),
            xml("absent"),
            "<tool_call><function=ping>",
            "<tool_call><function=ping></function>",
            "<tool_call><function=ping><parameter=x>bad</function></tool_call>",
            xml("ping") + " done",
        ]:
            with self.subTest(text=text):
                self.assertFalse(self.accepts(text))

    def test_named_and_parallel_false_are_enforced_by_grammar(self):
        ping, weather = xml("ping"), xml("get_weather")
        self.assertTrue(self.accepts(weather, names=("get_weather",), parallel=False))
        self.assertFalse(self.accepts(ping, names=("get_weather",)))
        self.assertFalse(self.accepts(ping + ping, parallel=False))
        self.assertFalse(self.accepts(ping + weather, parallel=False))
        self.assertTrue(self.accepts(ping + ping, names=("ping",), parallel=True))

    def test_parameter_values_preserve_literal_xml_unicode_and_empty_strings(self):
        for value in [
            "",
            "123",
            "true",
            '"quoted"',
            '{"nested": [1, false]}',
            "    return 1\n",
            "\r\n",
            "London / 東京 ☀️",
            'print("<think>x</think> </function> <tool_call></tool_call>")',
            "</paramete> </parameterX> << / /parameters>",
            "x" * 10000,
        ]:
            with self.subTest(value=value[:60]):
                self.assertTrue(self.accepts(xml("ping", [("code", value)])))

    def test_multiple_parameters_and_bounded_layout_whitespace(self):
        text = xml("ping", [("a", "1"), ("b", "true"), ("c", "{}")])
        self.assertTrue(self.accepts(text))
        self.assertFalse(self.accepts(" " * 30 + text))


class ForcedChoiceTemplateTests(unittest.IsolatedAsyncioTestCase):
    async def test_named_template_sees_only_selected_function(self):
        data = request(tool_choice=named("ping"))
        mc = SimpleNamespace(
            tool_format="qwen3_coder", template_vars_default={}, template_vars_force={}
        )
        formatter = AsyncMock(return_value=("PROMPT", None, {}))
        with (
            patch.object(cc, "model") as mocked,
            patch.object(cc, "format_messages_with_template", formatter),
        ):
            mocked.container = mc
            prompt, _ = await cc.apply_chat_template(data)
        self.assertEqual(prompt, "PROMPT")
        self.assertEqual([t["function"]["name"] for t in data.template_vars["tools"]], ["ping"])
        self.assertEqual(len(data.tools), 2, "the caller's declarations remain available")
        self.assertEqual(data.template_vars["tool_choice"], named("ping"))
        self.assertIn('"<function=ping>"', data.grammar_string)

    async def test_none_removes_tool_declarations_from_template(self):
        data = request(tool_choice="none", functions=[{"name": "legacy"}])
        mc = SimpleNamespace(template_vars_default={}, template_vars_force={})
        formatter = AsyncMock(return_value=("PROMPT", None, {}))
        with (
            patch.object(cc, "model") as mocked,
            patch.object(cc, "format_messages_with_template", formatter),
        ):
            mocked.container = mc
            await cc.apply_chat_template(data)
        self.assertIsNone(data.template_vars["tools"])
        self.assertIsNone(data.template_vars["functions"])
        self.assertIsNone(data.grammar_string)


class ForcedChoiceCollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_absent_or_wrong_forced_call_cannot_report_success(self):
        for stream in [False, True]:
            for raw in ["READY", xml("ping"), xml("absent")]:
                with self.subTest(stream=stream, raw=raw):
                    frames, result = await run_collector(
                        make_mc(pieces(raw)),
                        request(tool_choice=named("get_weather")),
                        streaming=stream,
                    )
                    final = frames[-1] if stream else result
                    self.assertIsInstance(final, ToolCallParseError)
                    if stream:
                        self.assertFalse(any(f.get("delta_tool_calls") for f in frames[:-1]))

    async def test_forced_parallel_false_does_not_silently_drop_extra_call(self):
        for stream in [False, True]:
            frames, result = await run_collector(
                make_mc(pieces(xml("ping") + xml("get_weather"))),
                request(parallel_tool_calls=False),
                streaming=stream,
            )
            final = frames[-1] if stream else result
            self.assertIsInstance(final, ToolCallParseError)
            if stream:
                names = [
                    d["function"]["name"]
                    for f in frames[:-1]
                    for d in f.get("delta_tool_calls", [])
                    if d["function"].get("name")
                ]
                self.assertEqual(names, ["ping"])

    async def test_thought_examples_cannot_bypass_content_grammar(self):
        raw = "<think>Example: " + xml("ping") + "</think>" + xml("get_weather")
        for stream in [False, True]:
            frames, result = await run_collector(
                make_mc(
                    pieces(raw), reasoning_start_token="<think>", reasoning_end_token="</think>"
                ),
                request(tool_choice=named("get_weather")),
                streaming=stream,
            )
            final = frames[-1] if stream else result
            self.assertEqual(final["finish_reason"], "tool_calls")
            if stream:
                names = [
                    d["function"]["name"]
                    for f in frames
                    for d in f.get("delta_tool_calls", [])
                    if d["function"].get("name")
                ]
            else:
                names = [call["function"]["name"] for call in result["tool_calls"]]
                self.assertIn("Example:", result["reasoning_content"])
            self.assertEqual(names, ["get_weather"])

    async def test_token_budget_stop_stays_length_even_without_complete_call(self):
        for stream in [False, True]:
            mc = make_mc([])

            async def backend(*args, **kwargs):
                yield {"text": "<tool_call><function=ping>", "finish_reason": "length"}

            mc.stream_generate = backend
            frames, result = await run_collector(mc, request(), streaming=stream)
            final = frames[-1] if stream else result
            self.assertEqual(final["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()
