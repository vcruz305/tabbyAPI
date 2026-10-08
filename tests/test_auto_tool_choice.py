"""Auto Qwen tools permit zero calls without an escape from an opened call.

Use real llguidance masks on byte and added-token tokenizers. Model inference is
covered by the standalone real-tokenizer check and the live recipe fixtures.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from endpoints.OAI.utils import chat_completion as cc
from endpoints.OAI.utils.tool_choice import auto_tool_grammar, prepare_forced_tool_choice
from tests.test_forced_tool_choice import named, request
from tests.test_qwen_tool_contract import xml
from tests.test_tool_delta_streaming import RE, RS, make_mc, pieces, run_collector


def bare(name="ping", params=()):
    return xml(name, params).removeprefix("<tool_call>").removesuffix("</tool_call>")


class AutoChoiceRequestTests(unittest.TestCase):
    def test_auto_and_default_install_grammar_without_forcing_choice(self):
        for choice in [None, "auto"]:
            for tool_format in ["qwen3_coder", "qwen3_5"]:
                with self.subTest(choice=choice, tool_format=tool_format):
                    data = request(tool_choice=choice)
                    before = data.model_dump()
                    self.assertIsNone(prepare_forced_tool_choice(data, tool_format))
                    self.assertTrue(data.grammar_string.startswith("start: auto_0_0\n"))
                    before["grammar_string"] = data.grammar_string
                    self.assertEqual(data.model_dump(), before)

    def test_none_no_declarations_and_other_formats_are_unchanged(self):
        for kwargs, tool_format in [
            ({"tool_choice": "none"}, "qwen3_coder"),
            ({"tools": None}, "qwen3_coder"),
            ({"tools": []}, "qwen3_coder"),
            ({}, "harmony"),
            ({}, None),
        ]:
            with self.subTest(kwargs=kwargs, tool_format=tool_format):
                data = request(tool_choice=kwargs.pop("tool_choice", "auto"), **kwargs)
                before = data.model_dump()
                self.assertIsNone(prepare_forced_tool_choice(data, tool_format))
                self.assertEqual(data.model_dump(), before)

    def test_explicit_constraints_and_continuations_are_preserved(self):
        for key, value in [
            ("grammar_string", 'start: "hello"'),
            ("regex_pattern", "[a-z]+"),
            ("json_schema", {"type": "object"}),
            ("response_format", {"type": "json_object"}),
            ("response_format", {"type": "json_schema", "json_schema": {"type": "object"}}),
            ("response_prefix", "<tool_call>"),
            ("continue_final_message", True),
        ]:
            with self.subTest(key=key):
                data = request(tool_choice="auto", **{key: value})
                before = data.model_dump()
                prepare_forced_tool_choice(data, "qwen3_coder")
                self.assertEqual(data.model_dump(), before)

    def test_text_response_format_and_legacy_functions(self):
        data = request(
            tool_choice="auto",
            tools=None,
            functions=[{"name": "legacy"}],
            response_format={"type": "text"},
            parallel_tool_calls=False,
        )
        prepare_forced_tool_choice(data, "qwen3_coder")
        self.assertEqual(data.grammar_string, auto_tool_grammar(("legacy",), False))

    def test_invalid_names_fail_before_constructing_grammar(self):
        for declarations in [[named("same"), named("same")], [named("a>b")], [named("")]]:
            with self.subTest(declarations=declarations):
                data = request(tool_choice="auto", tools=declarations)
                with self.assertRaises(HTTPException) as caught:
                    prepare_forced_tool_choice(data, "qwen3_coder")
                self.assertEqual(caught.exception.status_code, 400)


class AutoChoiceGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper
        except ImportError:
            raise unittest.SkipTest("llguidance is required for grammar mask tests")
        cls.matcher = LLMatcher
        cls.tokenizers = []
        # Include native opening/closing tags as well as ordinary byte spellings.
        # In particular, LLGuidance distinguishes all added tokens from regex text.
        added_tags = [
            "<tool_call>",
            "</tool_call>",
            "<function=",
            "</function>",
            "<parameter=",
            "</parameter>",
            "<think>",
            "</think>",
        ]
        for markers in [[], added_tags, added_tags + ["<function=ping>", "<function=absent>"]]:
            ids = {text: index + 257 for index, text in enumerate(markers)}
            marker_bytes = {text.encode(): token_id for text, token_id in ids.items()}

            class ByteTokenizer:
                eos_token_id = 256
                bos_token_id = None
                tokens = [bytes([i]) for i in range(256)] + [b"<eos>"] + list(marker_bytes)
                special_token_ids = list(range(256, len(tokens)))

                def __init__(self, added):
                    self.added = dict(sorted(added.items(), key=lambda item: -len(item[0])))

                def __call__(self, text):
                    raw = text if isinstance(text, bytes) else text.encode("utf-8")
                    result, pos = [], 0
                    while pos < len(raw):
                        for marker, token_id in self.added.items():
                            if raw.startswith(marker, pos):
                                result.append(token_id)
                                pos += len(marker)
                                break
                        else:
                            result.append(raw[pos])
                            pos += 1
                    return result

            tok = LLTokenizer(TokenizerWrapper(ByteTokenizer(marker_bytes)), slices=[])
            cls.tokenizers.append((tok, ids))

    def accepts(self, text, tokenizer, ids, parallel=True):
        grammar = auto_tool_grammar(("ping", "get_weather"), parallel, ids)
        self.assertEqual(self.matcher.validate_grammar(grammar, tokenizer), "")
        matcher = self.matcher(tokenizer, grammar, log_level=0)
        for token in tokenizer.tokenize_str(text):
            mask = matcher.compute_bitmask()
            if not mask[token // 8] & (1 << (token % 8)) or not matcher.consume_token(token):
                return False
        mask = matcher.compute_bitmask()
        return matcher.is_accepting() and bool(mask[256 // 8] & (1 << (256 % 8)))

    def check_cases(self, cases, parallel=True):
        cases = list(cases)
        for tokenizer, ids in self.tokenizers:
            for text, expected in cases:
                with self.subTest(added=bool(ids), parallel=parallel, text=text[:120]):
                    self.assertEqual(self.accepts(text, tokenizer, ids, parallel), expected)

    def test_zero_calls_plain_unicode_and_partial_openers(self):
        self.check_cases(
            (text, True)
            for text in [
                "",
                "READY",
                "\n" * 30,
                "Here is ordinary text. " * 100,
                "The price is 20 €; 東京 <test> ☀️.",
                "<tool_cal",
                "<function",
                "<<tool_cal<function<other>",
                "<think>plain literal markup</think>",
                "<tool_callX> <functionX>",
                "<<<<tool\n<function \n",
            ]
        )

    def test_wrapped_bare_preamble_and_parallel_calls(self):
        a, b = xml("ping"), bare("get_weather", [("city", "Paris")])
        self.check_cases(
            (text, True)
            for text in [
                a,
                b,
                "I will call it.\n" + a,
                a + "\nDone.",
                a + b,
                b + "\n" + a,
                "<" + a,
                "<tool_cal" + a,
                "Before\n" + b + "\nBetween\n" + a + "\nAfter",
            ]
        )

    def test_complete_opener_cannot_escape_into_free_text(self):
        self.check_cases(
            (text, False)
            for text in [
                xml("absent"),
                bare("absent"),
                "x<" + xml("absent"),
                "<tool_call>",
                "<tool_call></tool_call>",
                "<function=",
                "<tool_call><function=ping>",
                "<function=ping>",
                "<tool_call>ordinary content",
                "<function=ordinary content",
                "<tool_call><function=ping><parameter=x>bad</function></tool_call>",
                xml("ping") + "<tool_call>",
                xml("ping") + "<function=ping>",
                xml("ping") + "<tool_call><function=ping></tool_call>",
            ]
        )

    def test_parallel_false_keeps_text_but_allows_only_one_call(self):
        a, b = xml("ping"), bare("get_weather")
        self.check_cases(
            [
                (a, True),
                (b, True),
                ("READY", True),
                ("Before" + a + "After", True),
                (a + a, False),
                (a + b, False),
                (b + a, False),
                (b + b, False),
                (a + "\n<" + a, False),
            ],
            parallel=False,
        )

    def test_literal_values_and_first_parameter_close(self):
        self.check_cases(
            (xml("ping", [("code", value)]), True)
            for value in [
                "",
                "123",
                "true",
                '{"nested": [1, false]}',
                "    return 1\n",
                "東京 ☀️",
                "<think>literal</think>",
                'print("</function> <tool_call></tool_call> <function=x>")',
                "</paramete> </parameterX> < << </",
                "x" * 1000,
            ]
        )
        self.check_cases(
            [
                (xml("ping", [("code", "before</parameter>after")]), False),
                (xml("ping", [("code", "before</parameter>after<think>x")]), False),
            ]
        )

    def test_eos_is_blocked_through_every_open_call_prefix(self):
        for tokenizer, ids in self.tokenizers:
            for text, opener in [
                (xml("ping", [("code", "literal")]), "<tool_call>"),
                (bare("ping"), "<function="),
            ]:
                for end in range(len(opener), len(text)):
                    with self.subTest(added=bool(ids), end=end, text=text[:end]):
                        self.assertFalse(self.accepts(text[:end], tokenizer, ids))
                self.assertTrue(self.accepts(text, tokenizer, ids))


class AutoChoiceTemplateAndCollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_template_keeps_all_tools_choice_and_reasoning_settings(self):
        data = request(
            tool_choice="auto",
            enable_thinking=True,
            reasoning_effort="high",
            reasoning_budget_tokens=256,
            temperature=0.6,
            top_p=0.9,
        )
        mc = SimpleNamespace(
            tool_format="qwen3_coder", template_vars_default={}, template_vars_force={}
        )
        formatter = AsyncMock(return_value=("PROMPT", None, {}))
        before = data.model_dump()
        with (
            patch.object(cc, "model") as mocked,
            patch.object(cc, "format_messages_with_template", formatter),
        ):
            mocked.container = mc
            await cc.apply_chat_template(data)
        self.assertEqual(data.template_vars["tools"], before["tools"])
        self.assertEqual(data.template_vars["tool_choice"], "auto")
        self.assertTrue(data.template_vars["enable_thinking"])
        self.assertEqual(data.template_vars["reasoning_effort"], "high")
        for key in ["messages", "tools", "temperature", "top_p", "reasoning_budget_tokens"]:
            self.assertEqual(data.model_dump()[key], before[key])

    async def test_plain_and_tool_content_preserve_streaming_and_reasoning(self):
        for reasoning in ["", RS + "Choose freely." + RE]:
            for content, expected_reason in [
                ("READY", "stop"),
                ("", "stop"),
                ("Before" + xml("ping") + "After", "tool_calls"),
            ]:
                for streaming in [False, True]:
                    with self.subTest(
                        reasoning=bool(reasoning), content=content, streaming=streaming
                    ):
                        params = request(tool_choice="auto")
                        prepare_forced_tool_choice(params, "qwen3_coder")
                        mc = make_mc(pieces(reasoning + content))
                        frames, result = await run_collector(mc, params, streaming=streaming)
                        final = frames[-1] if streaming else result
                        self.assertEqual(final["finish_reason"], expected_reason)
                        self.assertEqual(params.tool_choice, "auto")
                        self.assertTrue(mc.tool_calls_in_reasoning)
                        if streaming:
                            actual_text = "".join(f.get("delta_content", "") for f in frames)
                            thought = "".join(f.get("delta_reasoning_content", "") for f in frames)
                        else:
                            actual_text = result["content"]
                            thought = result["reasoning_content"]
                        self.assertEqual(
                            actual_text or "",
                            "BeforeAfter" if expected_reason == "tool_calls" else content,
                        )
                        self.assertEqual(thought or "", "Choose freely." if reasoning else "")


if __name__ == "__main__":
    unittest.main()
