"""Request-local guidance for Qwen's raw XML parameter values.

This is an encoding hint, not a schema constraint or a promise that the model
will choose the requested value. Empty strings stay strings; parser coercion
and the inherent raw-string "null"/JSON-null ambiguity are unchanged.
"""
from copy import deepcopy

from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils.tool_choice import _output_constraint_conflicts
from endpoints.OAI.utils.toolcall_formats.qwen3_coder import ToolSchemas
from endpoints.OAI.utils.tools import canonical_format_name


NULLABLE_XML_GUIDANCE = (
    "Qwen XML argument encoding: a null value is the literal unquoted JSON text null "
    "inside its parameter tag. An empty string has empty string content. "
    "Preserve every character of strings, including spaces and newlines. "
    "The single LF immediately after the opening parameter tag and the single LF "
    "immediately before </parameter> are framing. A string that starts or ends in "
    "LF needs an additional LF for each such string character. "
    "These encoding rules do not choose whether to call a tool or which values to use."
)


def nullable_guidance_eligible(data: ChatCompletionRequest, tool_format: str | None) -> bool:
    """Evaluate before prepare_forced_tool_choice installs a server grammar."""
    return bool(
        data.tool_choice != "none"
        and (data.tools or data.functions)
        and canonical_format_name(tool_format) == "qwen3_coder"
        and not _output_constraint_conflicts(data)
        and not data.continue_final_message
        and not data.response_prefix
    )


def with_nullable_xml_guidance(declarations):
    """Append a hint only to copied, exposed function descriptions that need it.

    Reuse the parser's existing top-level type resolver, including the local
    references and unions it supports. Unresolved or malformed optional schema
    shapes keep their original prompt representation; this helper is not a new
    schema-validation boundary.
    """
    result = declarations
    for index, declaration in enumerate(declarations or []):
        function = declaration.get("function", declaration)
        if not isinstance(function, dict):
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            continue
        description = function.get("description")
        if description is not None and not isinstance(description, str):
            continue
        try:
            schemas = ToolSchemas([declaration])
            nullable = any(
                "null" in schemas.types(function.get("name"), name) for name in properties
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            nullable = False
        if not nullable or NULLABLE_XML_GUIDANCE in (description or ""):
            continue
        if result is declarations:
            result = deepcopy(declarations)
        target = result[index].get("function", result[index])
        target["description"] = (
            description + "\n\n" + NULLABLE_XML_GUIDANCE if description else NULLABLE_XML_GUIDANCE
        )
    return result
