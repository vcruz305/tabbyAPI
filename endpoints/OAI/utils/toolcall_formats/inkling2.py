import json

from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import FormatSignature

"""
Inkling 2 - structural message format with JSON tool call bodies

Tool calls are typed content messages. The Inkling 2 stream parser emits each
one on the tool channel as:

    __NAME__<|content_invoke_tool_json|>{"name":"__NAME__","args":{...}}<|end_message|>

The header before the content token is the function name. The JSON body is
authoritative for the name when it has one; the header is the fallback. The
`args` object is serialized as the call's arguments.

This format requires the Inkling 2 stream parser and is selected automatically
for Inkling 2 models; TOOLCALL_START/END are None since there are no tags for
TagStreamParser to scan for.
"""

TOOLCALL_START = None
TOOLCALL_END = None

DETECT = FormatSignature(
    template_markers=("<|message_model|>", "<|end_message|>"),
    special_tokens=(
        "<|message_model|>",
        "<|content_thinking|>",
        "<|content_text|>",
        "<|end_message|>",
    ),
    architectures=("Inkling2",),
    structured=True,
)

_TOOL_OPEN = "<|content_invoke_tool_json|>"
_MESSAGE_END = "<|end_message|>"


def _parse_call(header: str, body: str) -> ToolCall | None:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        payload = None

    if not isinstance(payload, dict):
        xlogger.warning(
            "inkling2: Tool message body is not a JSON object, skipping",
            {"raw_text": body},
        )
        return None

    name = payload.get("name") or header.strip()
    args = payload.get("args", {})
    if not isinstance(name, str) or not name or not isinstance(args, dict):
        xlogger.warning(
            "inkling2: Tool message needs a name and an args object, skipping",
            {"raw_text": body},
        )
        return None

    args_json = json.dumps(args, ensure_ascii=False)
    return ToolCall(function=Tool(name=name, arguments=args_json))


def parse_toolcalls(text: str) -> list[ToolCall]:
    # Every message on the tool channel is a tool call; they are split on
    # their terminator, since consecutive tool messages share a channel
    results = []
    for segment in text.split(_MESSAGE_END):
        if _TOOL_OPEN not in segment:
            continue
        header, body = segment.split(_TOOL_OPEN, 1)
        call = _parse_call(header, body)
        if call is not None:
            results.append(call)

    xlogger.debug(
        f"inkling2: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results},
    )
    return results
