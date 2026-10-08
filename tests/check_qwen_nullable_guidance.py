"""CPU check against a downloaded Qwen tokenizer_config.json; never loads a model.

The actual template and Tabby's real apply_chat_template/message formatting
path must expose the generic hint without mutating caller schemas/history.
Canonical history values then pass through the unchanged XML parser.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from common.templating import PromptTemplate
from endpoints.OAI.utils.qwen_tool_guidance import NULLABLE_XML_GUIDANCE
from endpoints.OAI.utils.toolcall_formats.qwen3_coder import ToolSchemas, parse_toolcalls
from tests.test_qwen_nullable_guidance import container, declaration, render
from tests.test_tool_delta_streaming import make_request


CASES = [
    ("empty", "", {"type": ["string", "null"]}, ""),
    ("null", None, {"type": ["string", "null"]}, None),
    ("numeric_text", "123", {"type": ["string", "null"]}, "123"),
    ("boolean_text", "true", {"type": ["string", "null"]}, "true"),
    ("json_text", '{"a":[1,false]}', {"type": ["string", "null"]}, '{"a":[1,false]}'),
    ("quoted", '"quoted"', {"type": ["string", "null"]}, '"quoted"'),
    ("code_lf", "  x = 1\n", {"type": ["string", "null"]}, "  x = 1\n"),
    ("code_two_lf", "  x = 1\n\n", {"type": ["string", "null"]}, "  x = 1\n\n"),
    ("leading_lf", "\n  x", {"type": ["string", "null"]}, "\n  x"),
    ("ending_cr", "line\r", {"type": ["string", "null"]}, "line\r"),
    ("crlf", "line\r\n", {"type": ["string", "null"]}, "line\r\n"),
    ("markup", "<think>x</think> </function> <tool_call> </tool_call>",
     {"type": ["string", "null"]}, "<think>x</think> </function> <tool_call> </tool_call>"),
    ("null_text_string_only", "null", {"type": "string"}, "null"),
    ("null_text_union_ambiguous", "null", {"type": ["string", "null"]}, None),
    ("object", {"a": [1, False]}, {"type": ["string", "object"]}, {"a": [1, False]}),
]


async def check(config):
    template_bytes = config.read_bytes()
    raw = json.loads(template_bytes)["chat_template"]
    if not isinstance(raw, str):
        raise ValueError("This diagnostic expects one Qwen chat_template string")
    mc = container()
    mc.prompt_template = PromptTemplate("actual-qwen-nullable", raw)
    rows = []
    for name, value, schema, expected in CASES:
        tools = [declaration(deepcopy(schema))]
        messages = [
            {"role": "user", "content": "Record the requested value."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "fixture-call", "type": "function", "function": {
                    "name": "save_value", "arguments": json.dumps({"value": value})}}]},
        ]
        data = make_request(tools=tools, messages=messages, add_generation_prompt=False,
                            enable_thinking=False)
        before = data.model_dump()
        prompt, _ = await render(data, mc)
        hinted = NULLABLE_XML_GUIDANCE in prompt
        expected_hint = "null" in ToolSchemas(tools).types("save_value", "value")
        if hinted != expected_hint:
            raise AssertionError(f"Wrong hint scope for {name}")
        for key in ("tools", "messages", "tool_choice"):
            if data.model_dump()[key] != before[key]:
                raise AssertionError(f"Caller {key} mutated for {name}")
        start = prompt.rfind("<function=save_value>")
        end = prompt.rfind("</tool_call>") + len("</tool_call>")
        if start < 0 or end <= start:
            raise AssertionError(f"Actual Qwen template did not render the fixture call: {name}")
        wire = prompt[start:end]
        parsed = parse_toolcalls(wire, tools=data.tools, strict=True)
        if len(parsed) != 1:
            raise AssertionError(f"Wrong parsed call count for {name}")
        actual = json.loads(parsed[0].function.arguments)["value"]
        if actual != expected or type(actual) is not type(expected):
            raise AssertionError(f"Parser contract changed for {name}: {actual!r}")
        rows.append({"case": name, "schema": schema, "original": value, "parsed": actual,
                     "roundtrip_exact": actual == value and type(actual) is type(value),
                     "hint_present": hinted, "wire": wire})
    root = Path(__file__).resolve().parents[1]
    files = ("endpoints/OAI/utils/qwen_tool_guidance.py",
             "endpoints/OAI/utils/chat_completion.py",
             "endpoints/OAI/utils/toolcall_formats/qwen3_coder.py",
             "tests/check_qwen_nullable_guidance.py",
             "tests/test_qwen_nullable_guidance.py")
    return {
        "scope": "CPU prompt/template/parser contract check; no inference or semantic-obedience claim",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "tracked_changes": subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip(),
        "source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
        "template_config_sha256": hashlib.sha256(template_bytes).hexdigest(),
        "cases": rows,
        "contract_checks": len(rows),
        "exact_roundtrips": sum(row["roundtrip_exact"] for row in rows),
        "known_ambiguities": [row["case"] for row in rows if not row["roundtrip_exact"]],
    }


def main():
    import asyncio
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer_config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error("Refusing an existing output path")
    report = asyncio.run(check(args.tokenizer_config))
    if args.output:
        with args.output.open("x") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
            file.write("\n")
    print(json.dumps({key: report[key] for key in (
        "contract_checks", "exact_roundtrips", "known_ambiguities", "source_commit")}))


if __name__ == "__main__":
    main()
