"""Qwen3.5 / Qwen3-Coder pseudo-XML tool calls.

The request schema disambiguates string arguments from JSON values. Literal
markup inside a parameter is data, including </function> and </tool_call>.
The first </parameter> still closes a value: that delimiter is inherently
ambiguous in this wire format and cannot be reconstructed by a parser.
"""

import json
import re

from common.errors import ToolCallParseError
from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import FormatSignature

TOOLCALL_START = "<tool_call>"
TOOLCALL_END = "</tool_call>"

DETECT = FormatSignature(
    template_markers=("<tool_call>", "<function="),
    special_tokens=("<tool_call>", "</tool_call>"),
    architectures=("Qwen3_5", "Qwen3Next", "Qwen4", "Qwen3Coder", "Step3", "NemotronH"),
    reasoning_tags=("<think>", "</think>"),
)

_FUNC_OPEN = re.compile(r"<function=([^>\s]+)[^>]*>")
_PARAM_OPEN = re.compile(r"<parameter=([^>\s]+)[^>]*>")
_FUNC_CLOSE = "</function>"
_PARAM_CLOSE = "</parameter>"


def normalize_string(raw: str) -> str:
    """Remove only the template's single surrounding line break.

    strip() silently removes indentation and final blank lines from code and
    file contents. A declared string keeps those bytes, and quotes and
    JSON-looking text are literal string data.
    """
    # Qwen's template inserts LF, not CRLF. Consuming a preceding CR here
    # would corrupt a legitimate string ending in a carriage return.
    return raw.removeprefix("\n").removesuffix("\n")


def _schema_types(schema, root, seen=frozenset()) -> set[str]:
    """Read unambiguous primitive types, including local refs and unions."""
    if not isinstance(schema, dict) or id(schema) in seen:
        return set()
    seen = seen | {id(schema)}
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        target = root
        try:
            for part in ref[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
            referenced = _schema_types(target, root, seen)
            if referenced:
                return referenced
        except (KeyError, TypeError):
            pass
    kind = schema.get("type")
    if isinstance(kind, str):
        return {kind}
    if isinstance(kind, list):
        return {t for t in kind if isinstance(t, str)}
    for union in ("anyOf", "oneOf"):
        if union in schema:
            branches = [_schema_types(s, root, seen) for s in schema[union]]
            # An unconstrained branch makes the union unconstrained.
            return set().union(*branches) if branches and all(branches) else set()
    if "allOf" in schema:
        branches = [s for s in (_schema_types(s, root, seen) for s in schema["allOf"]) if s]
        return set.intersection(*branches) if branches else set()
    values = schema.get("enum")
    if "const" in schema:
        values = [schema["const"]]
    if isinstance(values, list) and values and all(isinstance(v, str) for v in values):
        return {"string"}
    return set()


def _reject_constant(value):
    raise ValueError(f"Non-JSON number: {value}")


def coerce_value(raw: str):
    """Schema-free compatibility: safe JSON literals, otherwise plain text."""
    value = raw.strip()
    if not value:
        return ""
    try:
        return json.loads(value, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError):
        return value


class ToolSchemas:
    """Request-local parameter type lookup, shared by full and delta parsers."""

    def __init__(self, tools=None):
        self.functions = {}
        self._types = {}
        for tool in tools or []:
            if hasattr(tool, "model_dump"):
                tool = tool.model_dump()
            function = tool.get("function", tool)
            if isinstance(function, dict):
                self.functions[function.get("name")] = function.get("parameters") or {}

    def types(self, function: str, parameter: str) -> set[str]:
        key = (function, parameter)
        if key not in self._types:
            root = self.functions.get(function, {})
            schema = root.get("properties", {}).get(parameter, {})
            self._types[key] = _schema_types(schema, root)
        return self._types[key]

    def is_string(self, function: str, parameter: str) -> bool:
        return self.types(function, parameter) == {"string"}

    def coerce(self, raw: str, function: str, parameter: str):
        types = self.types(function, parameter)
        if types == {"string"}:
            return normalize_string(raw)
        value = coerce_value(raw)
        if "string" in types:
            # For a string union, only decode non-string JSON when its type
            # is actually allowed. Qwen writes strings raw, so a string branch
            # preserves quotes, indentation and final line breaks as data.
            value_types = set()
            if value is None:
                value_types = {"null"}
            elif isinstance(value, bool):
                value_types = {"boolean"}
            elif isinstance(value, int):
                value_types = {"integer", "number"}
            elif isinstance(value, float):
                value_types = {"number"}
                if value.is_integer():
                    value_types.add("integer")
            elif isinstance(value, list):
                value_types = {"array"}
            elif isinstance(value, dict):
                value_types = {"object"}
            if not types.intersection(value_types):
                return normalize_string(raw)
        return value


def parse_toolcalls(text: str, tools=None, strict: bool = False) -> list[ToolCall]:
    """Parse complete function blocks without interpreting markup in values.

    Incomplete functions or parameters are never converted into empty calls.
    The outer wrapper is optional; the same parser handles wrapped and bare
    function blocks. With strict=True, incomplete trailing calls and empty
    wrappers raise rather than silently returning only a completed prefix.
    No generated text is evaluated or executed.
    """
    schemas = tools if isinstance(tools, ToolSchemas) else ToolSchemas(tools)
    results = []
    position = 0
    while True:
        function = _FUNC_OPEN.search(text, position)
        if strict:
            gap = text[position : function.start() if function else len(text)]
            wrapper = gap.find(TOOLCALL_START)
            if wrapper >= 0 and gap.find(TOOLCALL_END, wrapper) >= 0:
                raise ToolCallParseError(
                    "The model emitted a tool wrapper without a function call."
                )
            if "<function=" in gap:
                raise ToolCallParseError("The model stopped inside a tool function header.")
        if function is None:
            break
        name = function.group(1)
        position = function.end()
        args = {}
        complete = False
        while position < len(text):
            parameter = _PARAM_OPEN.search(text, position)
            close = text.find(_FUNC_CLOSE, position)
            if strict:
                # Skip parameter bodies wholesale before looking for structure,
                # so literal wrapper/function tags inside code remain data.
                next_structure = min(
                    [p for p in (parameter.start() if parameter else -1, close) if p >= 0],
                    default=len(text),
                )
                barriers = [
                    text.find(tag, position) for tag in ("<function=", TOOLCALL_START, TOOLCALL_END)
                ]
                if any(0 <= barrier < next_structure for barrier in barriers):
                    raise ToolCallParseError(
                        "The model emitted an unclosed or nested tool function."
                    )
            if parameter is not None and (close < 0 or parameter.start() < close):
                parameter_end = text.find(_PARAM_CLOSE, parameter.end())
                if parameter_end < 0:
                    break
                key = parameter.group(1).strip()
                raw = text[parameter.end() : parameter_end]
                args[key] = schemas.coerce(raw, name, key)
                position = parameter_end + len(_PARAM_CLOSE)
            elif close >= 0:
                position = close + len(_FUNC_CLOSE)
                complete = True
                break
            else:
                break
        if not complete:
            if strict:
                raise ToolCallParseError("The model stopped inside a tool function or parameter.")
            break
        results.append(
            ToolCall(
                function=Tool(
                    name=name, arguments=json.dumps(args, ensure_ascii=False, allow_nan=False)
                )
            )
        )
    xlogger.debug(
        f"qwen3_coder: Parsed {len(results)} tool calls", {"raw_text": text, "results": results}
    )
    return results
