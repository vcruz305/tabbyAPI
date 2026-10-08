"""Regression contracts for Qwen tool types, literal markup and SSE ordering."""

import json
import unittest

from endpoints.OAI.types.tools import ToolSpec
from common.errors import ToolCallParseError
from endpoints.OAI.utils import chat_completion as cc
from endpoints.OAI.utils.stream_parser import TagStreamParser, TOOL
from endpoints.OAI.utils.toolcall_formats import qwen3_coder
from endpoints.OAI.utils.toolcall_stream import QwenToolCallDeltaStreamer
from tests.test_tool_delta_streaming import make_mc, make_request, pieces, run_collector


def xml(name="f", params=()):
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>" for key, value in params)
    return f"<tool_call><function={name}>{body}</function></tool_call>"


def tools(properties):
    return [
        ToolSpec(
            type="function",
            function={"name": "f", "parameters": {"type": "object", "properties": properties}},
        )
    ]


class QwenSchemaTests(unittest.TestCase):
    def test_string_arguments_keep_json_looking_text(self):
        values = [
            "123",
            "true",
            "false",
            "null",
            '{"nested": [1, false]}',
            '"literal quotes"',
            "00123",
            "",
            "NaN",
            "Infinity",
        ]
        for value in values:
            with self.subTest(value=value):
                calls = qwen3_coder.parse_toolcalls(
                    xml(params=[("s", value)]), tools=tools({"s": {"type": "string"}})
                )
                self.assertEqual(json.loads(calls[0].function.arguments), {"s": value})

    def test_typed_values(self):
        schema = {
            "i": {"type": "integer"},
            "n": {"type": "number"},
            "b": {"type": "boolean"},
            "a": {"type": "array"},
            "o": {"type": "object"},
            "z": {"type": ["string", "null"]},
        }
        calls = qwen3_coder.parse_toolcalls(
            xml(
                params=[
                    ("i", "3"),
                    ("n", "1.5"),
                    ("b", "true"),
                    ("a", '[1, "2"]'),
                    ("o", '{"n": false}'),
                    ("z", "null"),
                ]
            ),
            tools=tools(schema),
        )
        self.assertEqual(
            json.loads(calls[0].function.arguments),
            {"i": 3, "n": 1.5, "b": True, "a": [1, "2"], "o": {"n": False}, "z": None},
        )

    def test_string_unions_reject_forbidden_coercions_and_preserve_text(self):
        cases = [
            (["string", "null"], "null", None),
            (["string", "null"], "123", "123"),
            (["string", "null"], "true", "true"),
            (["string", "null"], "[1,2]", "[1,2]"),
            (["string", "null"], '{"n":1}', '{"n":1}'),
            (["string", "null"], '"quoted"', '"quoted"'),
            (["string", "null"], "    return 1\n", "    return 1\n"),
            (["string", "null"], "\ttruex\r\n", "\ttruex\r\n"),
            (["string", "integer"], "123", 123),
            (["string", "integer"], "1.5", "1.5"),
            (["string", "integer"], "true", "true"),
            (["string", "object"], '{"n":1}', {"n": 1}),
            (["string", "object"], "[1,2]", "[1,2]"),
            (["string", "object"], "null", "null"),
        ]
        for types, value, expected in cases:
            declarations = tools({"text": {"type": types}})
            raw = xml(params=[("text", value)])
            parsed = qwen3_coder.parse_toolcalls(raw, tools=declarations)
            self.assertEqual(json.loads(parsed[0].function.arguments), {"text": expected})
            for split in range(len(raw) + 1):
                with self.subTest(types=types, value=value, split=split):
                    stream = QwenToolCallDeltaStreamer(tools=declarations)
                    deltas = stream.feed(raw[:split]) + stream.feed(raw[split:])
                    args = "".join(d["function"].get("arguments", "") for d in deltas)
                    self.assertEqual(json.loads(args), {"text": expected})

    def test_local_refs_and_string_enums(self):
        declarations = tools(
            {
                "r": {"$ref": "#/$defs/text"},
                "e": {"enum": ["true", "false"]},
                "a": {"anyOf": [{"type": "string"}, {"enum": ["null"]}]},
            }
        )
        declarations[0].function.parameters["$defs"] = {"text": {"type": "string"}}
        calls = qwen3_coder.parse_toolcalls(
            xml(params=[("r", "123"), ("e", "true"), ("a", "null")]), tools=declarations
        )
        self.assertEqual(
            json.loads(calls[0].function.arguments), {"r": "123", "e": "true", "a": "null"}
        )

    def test_streaming_and_nonstreaming_types_match_every_split(self):
        declarations = tools({"text": {"type": "string"}, "count": {"type": "integer"}})
        raw = xml(params=[("text", '{"a":true}'), ("count", "7")])
        expected = qwen3_coder.parse_toolcalls(raw, tools=declarations)[0].function.arguments
        for split in range(len(raw) + 1):
            with self.subTest(split=split):
                stream = QwenToolCallDeltaStreamer(tools=declarations)
                deltas = stream.feed(raw[:split]) + stream.feed(raw[split:])
                args = "".join(d["function"].get("arguments", "") for d in deltas)
                self.assertEqual(args, expected)

    def test_string_whitespace_and_crlf_survive_every_split(self):
        declarations = tools({"text": {"type": "string"}})
        for value in ["    return 1\n", "\t x ", "foo\r", "\r", "foo\r\n", "\n\n"]:
            raw = xml(params=[("text", value)])
            for split in range(len(raw) + 1):
                with self.subTest(value=value, split=split):
                    stream = QwenToolCallDeltaStreamer(tools=declarations)
                    deltas = stream.feed(raw[:split]) + stream.feed(raw[split:])
                    args = "".join(d["function"].get("arguments", "") for d in deltas)
                    self.assertEqual(json.loads(args), {"text": value})

    def test_schema_free_keyword_and_whitespace_splits_match_final_parse(self):
        for value in [
            "true",
            "false",
            "null",
            "t",
            "f",
            "n",
            "three",
            "nullish",
            "true ",
            " true \n",
            "tru  e",
            "t\n\tr",
            "null" + " " * 4000 + "x",
            " " * 4000 + "hello" + "\t" * 4000 + "world",
            " \t\n" * 4000,
        ]:
            raw = xml(params=[("text", value)])
            expected = qwen3_coder.parse_toolcalls(raw)[0].function.arguments
            for width in [1, 7, 64, len(raw)]:
                with self.subTest(value=value[:30], width=width):
                    stream = QwenToolCallDeltaStreamer()
                    args = []
                    for chunk in pieces(raw, width):
                        args.extend(d["function"].get("arguments", "") for d in stream.feed(chunk))
                    self.assertEqual("".join(args), expected)

    def test_function_and_wrapper_literals_in_parameter_survive(self):
        value = 'print("<think>x</think> </function> <tool_call></tool_call>")'
        calls = qwen3_coder.parse_toolcalls(xml(params=[("code", value)]))
        self.assertEqual(json.loads(calls[0].function.arguments), {"code": value})

    def test_incomplete_function_never_becomes_empty_call(self):
        self.assertEqual(
            qwen3_coder.parse_toolcalls(
                "<tool_call><function=f><parameter=x>unfinished</function></tool_call>"
            ),
            [],
        )

    def test_legacy_schema_free_coercion_still_works(self):
        calls = qwen3_coder.parse_toolcalls(xml(params=[("x", "true")]))
        self.assertEqual(json.loads(calls[0].function.arguments), {"x": True})


class QwenTagTests(unittest.TestCase):
    def test_literal_reasoning_tags_survive_in_tool_channel(self):
        raw = xml(params=[("code", 'print("<think>hello</think>")')])
        for width in [1, 3, len(raw)]:
            parser = TagStreamParser(
                reasoning_start="<think>",
                reasoning_end="</think>",
                tool_start="<tool_call>",
                tool_end="</tool_call>",
            )
            events = []
            for chunk in pieces(raw, width):
                events.extend(parser.feed(chunk))
            events.extend(parser.finish())
            self.assertEqual("".join(text for channel, text in events if channel == TOOL), raw)


class QwenCollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_chunk_reasoning_precedes_tool_deltas(self):
        raw = "<think>plan</think>" + xml("get_weather", [("city", "Paris")])
        mc = make_mc([raw], reasoning_start_token="<think>", reasoning_end_token="</think>")
        frames, _ = await run_collector(mc, make_request())
        nonempty = [
            f for f in frames if f.get("delta_reasoning_content") or f.get("delta_tool_calls")
        ]
        self.assertEqual(nonempty[0].get("delta_reasoning_content"), "plan")
        self.assertTrue(nonempty[1].get("delta_tool_calls"))

    async def test_one_chunk_content_precedes_tool_deltas(self):
        mc = make_mc(["checking now" + xml("get_weather", [("city", "Paris")])])
        frames, _ = await run_collector(mc, make_request())
        nonempty = [f for f in frames if f.get("delta_content") or f.get("delta_tool_calls")]
        self.assertEqual(nonempty[0].get("delta_content"), "checking now")
        self.assertTrue(nonempty[1].get("delta_tool_calls"))

    async def test_malformed_tool_does_not_claim_success(self):
        for stream in [False, True]:
            mc = make_mc(["<tool_call>invalid</tool_call>"])
            frames, result = await run_collector(mc, make_request(), streaming=stream)
            final = frames[-1] if stream else result
            self.assertIsInstance(final, ToolCallParseError)

    async def test_incomplete_trailing_calls_fail_equally_in_both_modes(self):
        prefix = xml("get_weather", [("city", "Paris")])
        tails = [
            "<tool_call><function=get_weather><parameter=city>London",
            "<tool_call><function=get_weather><parameter=city>London</parameter></tool_call>",
            "<tool_call>",
            "<tool_call></tool_call>",
            "<function=get_weather",
            "<tool_call><function=get_weather><function=get_weather></function></tool_call>",
        ]
        for tail in tails:
            for stream in [False, True]:
                for choice in ["auto", "required"]:
                    for finish in ["stop", "length"]:
                        with self.subTest(tail=tail, stream=stream, choice=choice, finish=finish):
                            mc = make_mc([])

                            async def backend(*args, **kwargs):
                                for chunk in pieces(prefix + tail, 3):
                                    yield {"text": chunk}
                                yield {"text": "", "finish_reason": finish}

                            mc.stream_generate = backend
                            frames, result = await run_collector(
                                mc, make_request(tool_choice=choice), streaming=stream
                            )
                            final = frames[-1] if stream else result
                            if finish == "stop":
                                self.assertIsInstance(final, ToolCallParseError)
                            else:
                                self.assertEqual(final.get("finish_reason"), "length")

    async def test_length_finish_is_preserved_for_partial_call(self):
        for stream in [False, True]:
            mc = make_mc([])

            async def backend(*args, **kwargs):
                yield {
                    "text": "<tool_call><function=get_weather><parameter=city>Par",
                    "finish_reason": "length",
                    "eos_reason": "max_new_tokens",
                }

            mc.stream_generate = backend
            frames, result = await run_collector(mc, make_request(), streaming=stream)
            final = frames[-1] if stream else result
            self.assertEqual(final.get("finish_reason"), "length")

    async def test_literal_tags_survive_collector(self):
        value = 'print("<think>x</think> </function> <tool_call></tool_call>")'
        for stream in [False, True]:
            request = make_request(tools=tools({"code": {"type": "string"}}))
            mc = make_mc(
                pieces(xml(params=[("code", value)]), 2),
                reasoning_start_token="<think>",
                reasoning_end_token="</think>",
            )
            frames, result = await run_collector(mc, request, streaming=stream)
            if stream:
                args = "".join(
                    d["function"].get("arguments", "")
                    for f in frames
                    for d in f.get("delta_tool_calls", [])
                )
            else:
                args = result["tool_calls"][0]["function"]["arguments"]
            self.assertEqual(json.loads(args), {"code": value})

    async def test_nonstreaming_tool_has_no_stream_index(self):
        _, result = await run_collector(
            make_mc([xml("get_weather", [("city", "Paris")])]), make_request(), streaming=False
        )
        self.assertNotIn("index", result["tool_calls"][0])


if __name__ == "__main__":
    unittest.main()
