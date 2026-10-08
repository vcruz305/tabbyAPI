"""Prompt-only nullable guidance; no schema, parser, or choice policy changes."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from common.templating import PromptTemplate
from endpoints.OAI.utils import chat_completion as cc
from endpoints.OAI.utils.qwen_tool_guidance import (
    NULLABLE_XML_GUIDANCE,
    nullable_guidance_eligible,
    with_nullable_xml_guidance,
)
from endpoints.OAI.utils.toolcall_formats.qwen3_coder import ToolSchemas
from tests.test_forced_tool_choice import named
from tests.test_tool_delta_streaming import make_mc, make_request, pieces, run_collector


def declaration(schema=None, name="save_value", parameter="value", description="Choose values from the request."):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {parameter: schema if schema is not None else {"type": ["string", "null"]}},
                "required": [parameter],
                "additionalProperties": False,
            },
        },
    }


def container(tool_format="qwen3_coder", raw_template=None):
    # Exercise the real message normalization and PromptTemplate render path.
    template = raw_template or (
        "{{ {'tools': tools, 'functions': functions, 'messages': messages, "
        "'choice': tool_choice, 'parallel': parallel_tool_calls}|tojson }}"
    )
    return SimpleNamespace(
        tool_format=tool_format,
        template_vars_default={},
        template_vars_force={},
        use_vision=False,
        get_special_tokens=lambda: {},
        prompt_template=PromptTemplate("nullable-guidance-test", template),
    )


async def render(data, mc=None):
    with patch.object(cc, "model") as mocked:
        mocked.container = mc or container()
        return await cc.apply_chat_template(data)


class NullableGuidanceUnitTests(unittest.TestCase):
    def test_relevant_primitive_unions_and_local_refs_use_existing_resolver(self):
        for schema in (
            {"type": ["null", "string"]},
            {"type": "null"},
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
            {"oneOf": [{"type": "integer"}, {"type": "null"}]},
            {"allOf": [{"type": ["string", "null"]}, {"type": ["null", "number"]}]},
            {"$ref": "#/$defs/maybe"},
        ):
            with self.subTest(schema=schema):
                tool = declaration(schema, name="another_function", parameter="arbitrary_property")
                tool["function"]["parameters"]["$defs"] = {"maybe": {"type": ["string", "null"]}}
                original = deepcopy(tool)
                result = with_nullable_xml_guidance([tool])
                self.assertIn(NULLABLE_XML_GUIDANCE, result[0]["function"]["description"])
                self.assertEqual(result[0]["function"]["parameters"], original["function"]["parameters"])
                self.assertEqual(tool, original)

    def test_only_function_description_changes_on_detached_copy(self):
        tools = [declaration({"type": "string"}, name="plain"), declaration()]
        before = deepcopy(tools)
        result = with_nullable_xml_guidance(tools)
        self.assertEqual(tools, before)
        self.assertIsNot(result, tools)
        self.assertEqual(result[0], tools[0])
        changed = result[1]["function"].pop("description")
        self.assertEqual(changed, tools[1]["function"]["description"] + "\n\n" + NULLABLE_XML_GUIDANCE)
        expected = deepcopy(tools[1]); expected["function"].pop("description")
        self.assertEqual(result[1], expected)
        result[1]["function"]["parameters"]["properties"]["value"]["type"].append("integer")
        self.assertEqual(tools, before, "the caller's nested schema must not alias the template copy")

    def test_legacy_functions_empty_description_and_idempotence(self):
        for description in (None, "", "Existing meaning."):
            functions = [declaration(description=description)["function"]]
            before = deepcopy(functions)
            once = with_nullable_xml_guidance(functions)
            twice = with_nullable_xml_guidance(once)
            self.assertEqual(once, twice)
            self.assertEqual(functions, before)
            self.assertEqual(once[0]["description"].count(NULLABLE_XML_GUIDANCE), 1)

    def test_unrecognized_or_irrelevant_schema_shapes_are_noop(self):
        for schema in (
            {"type": "string"},
            {"type": ["string", "number"]},
            {},
            {"anyOf": [{"type": "null"}, {}]},
            {"anyOf": 7},
            {"$ref": "#/$defs/missing"},
            {"$ref": "https://example.invalid/schema"},
            {"type": "object", "properties": {"nested": {"type": ["string", "null"]}}},
        ):
            with self.subTest(schema=schema):
                tools = [declaration(schema)]
                tools[0]["function"]["parameters"]["$defs"] = {"unused": {"type": "null"}}
                self.assertIs(with_nullable_xml_guidance(tools), tools)
        for value in (None, []):
            self.assertIs(with_nullable_xml_guidance(value), value)
        legacy = [declaration()["function"]]
        legacy[0]["description"] = 7
        self.assertIs(with_nullable_xml_guidance(legacy), legacy)

    def test_scope_uses_qwen_aliases_and_preserves_explicit_constraints(self):
        for tool_format in ("qwen3_coder", "qwen3_5"):
            for choice in (None, "auto", "required", named("save_value")):
                self.assertTrue(nullable_guidance_eligible(
                    make_request(tools=[declaration()], tool_choice=choice), tool_format))
        for kwargs in (
            {"tool_choice": "none"},
            {"tools": []},
            {"grammar_string": 'start: "READY"'},
            {"regex_pattern": "READY"},
            {"json_schema": {"type": "object"}},
            {"response_format": {"type": "json_object"}},
            {"response_prefix": "<tool_call>"},
            {"continue_final_message": True},
        ):
            options = {"tools": [declaration()], **kwargs}
            self.assertFalse(nullable_guidance_eligible(make_request(**options), "qwen3_coder"))
        for tool_format in (None, "harmony", "llama3"):
            self.assertFalse(nullable_guidance_eligible(
                make_request(tools=[declaration()]), tool_format))

    def test_coercion_remains_null_empty_and_exact_whitespace(self):
        schemas = ToolSchemas([declaration()])
        for raw, expected in (
            ("\nnull\n", None), ("\n\n", ""), ("\n123\n", "123"),
            ("\ntrue\n", "true"), ("\n[1,2]\n", "[1,2]"),
            ("\n    line\n\n", "    line\n"), ("\n\n    line\r\n\n", "\n    line\r\n"),
            ("\n<think>literal</think>\n", "<think>literal</think>"),
        ):
            with self.subTest(raw=raw):
                actual = schemas.coerce(raw, "save_value", "value")
                self.assertEqual(actual, expected)
                self.assertIs(type(actual), type(expected))
        self.assertIsNone(schemas.coerce("null", "save_value", "value"),
                          "raw string null remains inherently ambiguous; this is not a codec")


class NullableGuidanceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_formatting_path_preserves_request_and_template_controls(self):
        data = make_request(
            tools=[declaration()], enable_thinking=True, temperature=.6, top_p=.9,
            template_vars={"custom_flag": "keep", "enable_thinking": False},
            reasoning_budget_tokens=64,
        )
        before = data.model_dump()
        prompt, embeddings = await render(data)
        rendered = json.loads(prompt)
        self.assertIsNone(embeddings)
        self.assertIn(NULLABLE_XML_GUIDANCE, rendered["tools"][0]["function"]["description"])
        self.assertEqual(rendered["choice"], "auto")
        self.assertEqual(rendered["parallel"], data.parallel_tool_calls)
        for key in ("messages", "tools", "functions", "tool_choice", "temperature",
                    "top_p", "reasoning_budget_tokens"):
            self.assertEqual(data.model_dump()[key], before[key])
        self.assertFalse(data.template_vars["enable_thinking"])
        self.assertEqual(data.template_vars["custom_flag"], "keep")

    async def test_auto_retains_zero_call_collector_behavior(self):
        data = make_request(tools=[declaration()])
        await render(data)
        self.assertEqual(data.tool_choice, "auto")
        self.assertIn("auto", data.grammar_string)
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                frames, result = await run_collector(
                    make_mc(pieces("READY")), data.model_copy(deep=True), streaming)
                if streaming:
                    self.assertEqual(frames[-1]["finish_reason"], "stop")
                else:
                    self.assertEqual(result["finish_reason"], "stop")
                    self.assertEqual(result["content"], "READY")
                    self.assertFalse(result.get("tool_calls"))

    async def test_named_choice_only_hints_selected_declaration(self):
        data = make_request(tools=[declaration(), declaration(name="unselected")],
                            tool_choice=named("save_value"))
        before = data.model_dump()
        prompt, _ = await render(data)
        rendered = json.loads(prompt)
        self.assertEqual(len(rendered["tools"]), 1)
        self.assertIn(NULLABLE_XML_GUIDANCE, rendered["tools"][0]["function"]["description"])
        self.assertEqual(rendered["choice"], named("save_value"))
        self.assertEqual(data.model_dump()["tools"], before["tools"])

    async def test_legacy_functions_are_copied_but_history_values_are_not_rewritten(self):
        functions = [declaration()["function"]]
        messages = [
            {"role": "user", "content": "Use the requested value."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "fixture-call", "type": "function", "function": {
                    "name": "save_value", "arguments": '{"value":null}'}}]},
            {"role": "tool", "tool_call_id": "fixture-call", "content": "accepted"},
        ]
        data = make_request(tools=None, functions=functions, messages=messages)
        before = data.model_dump()
        prompt, _ = await render(data)
        rendered = json.loads(prompt)
        self.assertIn(NULLABLE_XML_GUIDANCE, rendered["functions"][0]["description"])
        self.assertEqual(data.model_dump()["functions"], before["functions"])
        self.assertEqual(data.model_dump()["messages"], before["messages"])
        self.assertIsNone(rendered["messages"][1]["tool_calls"][0]["function"]["arguments"]["value"])

    async def test_explicit_constraints_other_formats_and_none_are_unchanged(self):
        for options, tool_format in (
            ({"grammar_string": 'start: "READY"'}, "qwen3_coder"),
            ({"regex_pattern": "READY"}, "qwen3_coder"),
            ({"json_schema": {"type": "object"}}, "qwen3_coder"),
            ({"response_format": {"type": "json_object"}}, "qwen3_coder"),
            ({}, "harmony"),
            ({"tool_choice": "none"}, "qwen3_coder"),
        ):
            with self.subTest(options=options, tool_format=tool_format):
                data = make_request(tools=[declaration()], **options)
                before = data.model_dump()
                prompt, _ = await render(data, container(tool_format))
                self.assertNotIn(NULLABLE_XML_GUIDANCE, prompt)
                self.assertEqual(data.model_dump()["tools"], before["tools"])
                for key in options:
                    self.assertEqual(data.model_dump()[key], before[key])


if __name__ == "__main__":
    unittest.main()
