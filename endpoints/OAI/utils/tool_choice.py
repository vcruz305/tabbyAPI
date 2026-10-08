"""Qwen forced tool choices backed by the existing llguidance output filter.

The grammar constrains call structure, function names and call count. It does
not claim strict JSON Schema validation of the pseudo-XML parameter values.
"""

from dataclasses import dataclass
import json
import re

from fastapi import HTTPException

from common.errors import ToolCallParseError
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils.tools import canonical_format_name


def function_name(tool) -> str:
    """Read a function name from a tool spec or legacy function declaration."""

    if hasattr(tool, "model_dump"):
        tool = tool.model_dump()
    function = tool.get("function", tool)
    return function.get("name", "")


@dataclass(frozen=True)
class ForcedToolChoice:
    names: tuple[str, ...]
    parallel: bool

    def grammar(self) -> str:
        # llguidance's suffix lexeme stops at the first parameter close tag.
        # A greedy regex with a separate close terminal would consume the
        # close-tag prefix and dead-end instead. Other XML in a value is literal.
        names = " | ".join(json.dumps(f"<function={name}>") for name in self.names)
        repetition = " (WS tool_call)*" if self.parallel else ""
        return (
            f"start: WS tool_call{repetition} WS\n"
            'tool_call: "<tool_call>" WS function WS "</tool_call>"\n'
            f'function: ({names}) WS parameter* "</function>"\n'
            'parameter: "<parameter=" NAME ">" value WS\n'
            'value[suffix="</parameter>"]: /(?s:.*)/\n'
            "WS: /[ \\t\\r\\n]{0,8}/\n"
            "NAME: /[^<>\\s=]+/\n"
        )

    def validate_name(self, name: str, index: int = 0):
        """Validate before streaming a call's name or any argument fragments."""

        if name not in self.names:
            raise ToolCallParseError(
                f"The model emitted tool {name!r}, which is not allowed by tool_choice."
            )
        if not self.parallel and index > 0:
            raise ToolCallParseError(
                "The model emitted multiple calls despite parallel_tool_calls=false."
            )

    def validate_calls(self, calls: list, require_call: bool = True):
        # Recheck at EOS for non-streaming, and guard any backend/parser mismatch.
        # A token-budget stop may have no complete call and remains `length`.
        if require_call and not calls:
            raise ToolCallParseError(
                "The model stopped without a complete call required by tool_choice. "
                "Retry the request or inspect the model output."
            )
        for index, call in enumerate(calls):
            self.validate_name(call["function"]["name"], index)


def resolve_forced_tool_choice(data: ChatCompletionRequest) -> ForcedToolChoice | None:
    choice = data.tool_choice
    if choice is None or choice in ("auto", "none"):
        return None

    declarations = data.tools or data.functions or []
    if not declarations:
        raise HTTPException(400, "tool_choice requires at least one tool declaration.")

    names = tuple(function_name(tool) for tool in declarations)
    if any(not isinstance(name, str) or not name or re.search(r"[\s<>=]", name) for name in names):
        raise HTTPException(
            400,
            "Forced tool_choice requires nonempty function names without whitespace, <, > or =.",
        )
    if len(set(names)) != len(names):
        raise HTTPException(400, "Forced tool_choice requires unique function names.")

    if choice != "required":
        selected = choice.function.name
        if selected not in names:
            raise HTTPException(
                400, f"tool_choice names {selected!r}, which is not present in tools."
            )
        names = (selected,)

    return ForcedToolChoice(names, parallel=data.parallel_tool_calls is not False)


def prepare_forced_tool_choice(
    data: ChatCompletionRequest, tool_format: str | None
) -> ForcedToolChoice | None:
    """Validate forcing before generation and install the request's XML grammar."""

    choice = resolve_forced_tool_choice(data)
    if choice is None:
        return None
    if canonical_format_name(tool_format) != "qwen3_coder":
        raise HTTPException(
            400,
            "Forced tool_choice is currently supported for the Qwen pseudo-XML tool format "
            "(qwen3_coder and its aliases). Use tool_choice='auto' for this model.",
        )

    conflicts = [
        key for key in ("grammar_string", "regex_pattern", "json_schema") if getattr(data, key)
    ]
    if data.response_format and data.response_format.type != "text":
        conflicts.append("response_format")
    if conflicts:
        raise HTTPException(
            400,
            "Forced tool_choice cannot be combined with another output constraint: "
            + ", ".join(conflicts)
            + ". The server supplies the tool-call grammar.",
        )
    if data.continue_final_message or data.response_prefix:
        raise HTTPException(
            400,
            "Forced tool_choice cannot be combined with continue_final_message or "
            "response_prefix because its grammar starts at a new tool call.",
        )

    data.grammar_string = choice.grammar()
    return choice
