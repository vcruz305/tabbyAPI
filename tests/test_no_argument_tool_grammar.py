"""A closed empty parameter object must not permit invented Qwen parameters.

Exercise the request preparation path and actual next-token masks with normal
byte tokens, native delimiters, and native complete function-opening tokens.
"""

from copy import deepcopy
from types import SimpleNamespace
import unittest

from endpoints.OAI.utils.tool_choice import prepare_forced_tool_choice
from tests import test_auto_tool_choice as auto_tests
from tests.test_forced_tool_choice import named, request
from tests.test_qwen_tool_contract import xml


CLOSED_EMPTY = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}


def declaration(name, schema):
    return {"type": "function", "function": {"name": name, "parameters": deepcopy(schema)}}


def request_with_schema(choice, schema=CLOSED_EMPTY, legacy=False, parallel=True):
    tools = [
        declaration("ping", schema),
        declaration(
            "get_weather",
            {"type": "object", "properties": {"city": {"type": "string"}}},
        ),
    ]
    declarations = (
        {"tools": None, "functions": [tool["function"] for tool in tools]}
        if legacy
        else {"tools": tools}
    )
    return request(tool_choice=choice, parallel_tool_calls=parallel, **declarations)


class ClosedEmptyToolGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the existing three independent tokenizer fixtures; do not
        # simulate llguidance or substitute string matching for its masks.
        auto_tests.AutoChoiceGrammarTests.setUpClass()
        cls.matcher = auto_tests.AutoChoiceGrammarTests.matcher
        cls.tokenizers = auto_tests.AutoChoiceGrammarTests.tokenizers

    def accepts(self, data, text, tokenizer, ids):
        added = {
            token_id: SimpleNamespace(content=content, special=False)
            for content, token_id in ids.items()
        }
        before = data.model_dump()
        prepare_forced_tool_choice(
            data,
            "qwen3_coder",
            SimpleNamespace(get_added_tokens_decoder=lambda: added),
        )
        # Installing a grammar must not rewrite schemas, request history,
        # tool_choice, or string values in the caller's request.
        before["grammar_string"] = data.grammar_string
        self.assertEqual(data.model_dump(), before)
        self.assertEqual(self.matcher.validate_grammar(data.grammar_string, tokenizer), "")
        matcher = self.matcher(tokenizer, data.grammar_string, log_level=0)
        for token in tokenizer.tokenize_str(text):
            mask = matcher.compute_bitmask()
            if not mask[token // 8] & (1 << (token % 8)):
                return False
            if not matcher.consume_token(token):
                return False
        mask = matcher.compute_bitmask()
        return matcher.is_accepting() and bool(mask[256 // 8] & (1 << (256 % 8)))

    def test_closed_empty_object_accepts_no_args_and_rejects_invented_parameters(self):
        for choice in (None, "auto", "required", named("ping")):
            for legacy in (False, True):
                for tokenizer, ids in self.tokenizers:
                    for text, expected in (
                        (xml("ping"), True),
                        (xml("ping", [("__noargs", "")]), False),
                        (xml("ping", [("x", "123")]), False),
                    ):
                        with self.subTest(
                            choice=choice, legacy=legacy, native=tuple(ids), text=text
                        ):
                            self.assertEqual(
                                self.accepts(
                                    request_with_schema(choice, legacy=legacy),
                                    text,
                                    tokenizer,
                                    ids,
                                ),
                                expected,
                            )

    def test_auto_bare_and_native_function_openers_share_the_empty_body_rule(self):
        for tokenizer, ids in self.tokenizers:
            for text, expected in (
                ("READY", True),
                ("", True),
                ("Before " + auto_tests.bare("ping") + " after.", True),
                (auto_tests.bare("ping", [("__noargs", "")]), False),
                ("Before " + auto_tests.bare("ping", [("x", "wrong")]), False),
                (xml("ping") + auto_tests.bare("ping"), True),
            ):
                with self.subTest(native=tuple(ids), text=text):
                    self.assertEqual(
                        self.accepts(request_with_schema("auto"), text, tokenizer, ids), expected
                    )

    def test_restriction_is_per_function_and_preserves_literal_values_elsewhere(self):
        weather = xml(
            "get_weather",
            [("city", '<think>literal</think> <function=ping> </function> 東京')],
        )
        for tokenizer, ids in self.tokenizers:
            for choice in ("auto", "required", named("get_weather")):
                for text in (weather,):
                    with self.subTest(native=tuple(ids), choice=choice):
                        self.assertTrue(
                            self.accepts(request_with_schema(choice), text, tokenizer, ids)
                        )
            for choice in ("auto", "required"):
                self.assertTrue(
                    self.accepts(
                        request_with_schema(choice), weather + xml("ping"), tokenizer, ids
                    )
                )
                self.assertFalse(
                    self.accepts(
                        request_with_schema(choice, parallel=False),
                        xml("ping") + weather,
                        tokenizer,
                        ids,
                    )
                )
            self.assertFalse(
                self.accepts(request_with_schema(named("ping")), weather, tokenizer, ids)
            )

    def test_open_or_pattern_schemas_are_not_mistaken_for_no_argument_tools(self):
        schemas = [
            {},
            {"type": "object", "properties": {}},
            {"type": "object", "properties": {}, "additionalProperties": True},
            {
                "type": "object",
                "properties": {},
                "additionalProperties": {"type": "string"},
            },
            {
                "type": "object",
                "properties": {},
                "patternProperties": {"^x$": {"type": "string"}},
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "additionalProperties": False,
            },
        ]
        for schema in schemas:
            for choice in ("auto", "required", named("ping")):
                for tokenizer, ids in self.tokenizers:
                    with self.subTest(schema=schema, choice=choice, native=tuple(ids)):
                        self.assertTrue(
                            self.accepts(
                                request_with_schema(choice, schema),
                                xml("ping", [("x", "literal")]),
                                tokenizer,
                                ids,
                            )
                        )

    def test_no_argument_call_still_requires_its_closing_structure(self):
        for tokenizer, ids in self.tokenizers:
            for choice in ("auto", "required", named("ping")):
                for text in (
                    "<tool_call>",
                    "<tool_call><function=ping>",
                    "<tool_call><function=ping>\n",
                    "<tool_call><function=ping></function>",
                ):
                    with self.subTest(native=tuple(ids), choice=choice, prefix=text):
                        self.assertFalse(
                            self.accepts(request_with_schema(choice), text, tokenizer, ids)
                        )


if __name__ == "__main__":
    unittest.main()
