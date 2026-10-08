"""Qwen tool choices backed by the existing llguidance output filter.

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


# These tags can occur literally in code even when a tokenizer marks them
# special. Other special tokens (EOS, message boundaries, image/audio markers)
# remain governed by the model's normal stop/control-token handling.
_LITERAL_MARKERS = {
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
}


def literal_token_ids(tokenizer) -> dict[str, int]:
    """Added tokens llguidance represents as token IDs instead of UTF-8 text.

    This includes HF added tokens with special=False. Reading only the model's
    special-token flags would miss Qwen's tool wrappers and reasoning markers.
    Accept the backend tokenizer or its underlying Hugging Face tokenizer.
    """

    hf = getattr(tokenizer, "tokenizer", tokenizer)
    decoder = getattr(hf, "get_added_tokens_decoder", None)
    if decoder is None:
        return {}
    return {
        token.content: token_id
        for token_id, token in decoder().items()
        if not token.special or token.content in _LITERAL_MARKERS
    }


@dataclass(frozen=True)
class ForcedToolChoice:
    names: tuple[str, ...]
    parallel: bool

    def grammar(self, token_ids: dict[str, int] | None = None) -> str:
        token_ids = token_ids or {}

        def marker(text):
            quoted = json.dumps(text)
            if text in token_ids:
                return f"({quoted} | <[{token_ids[text]}]>)"
            return quoted

        # Text uses a lazy suffix to end at the first parameter close. Added
        # tokens interrupt text lexemes; permit literal ones between text runs.
        # They must be grammar rules, since special token IDs cannot occur
        # inside regex terminals. Quoted markers remain valid as normal text.
        literals = [token_id for text, token_id in token_ids.items() if text != "</parameter>"]
        if literals:
            alternatives = " | ".join(f"<[{token_id}]>" for token_id in sorted(set(literals)))
            value = "value: (TEXT special_literal)* value_end\n"
            value += f"special_literal: {alternatives}\nTEXT: /(?s:.*)/\n"
        else:
            value = "value: value_end\n"
        value += 'value_end[suffix="</parameter>"]: /(?s:.*)/\n'
        if "</parameter>" in token_ids:
            # A parameter delimiter can itself be an added token in compatible
            # tokenizers. It closes the value rather than becoming literal data.
            value = value.replace(
                "value_end\n", f"(value_end | TEXT <[{token_ids['</parameter>']}]>)\n", 1
            )
            if not literals:
                value += "TEXT: /(?s:.*)/\n"

        names = " | ".join(marker(f"<function={name}>") for name in self.names)
        if "<function=" in token_ids:
            tails = " | ".join(json.dumps(name + ">") for name in self.names)
            names += f" | <[{token_ids['<function=']}]> ({tails})"
        repetition = " (WS tool_call)*" if self.parallel else ""
        return (
            f"start: WS tool_call{repetition} WS\n"
            f"tool_call: {marker('<tool_call>')} WS function WS {marker('</tool_call>')}\n"
            f"function: ({names}) WS parameter* {marker('</function>')}\n"
            f'parameter: {marker("<parameter=")} NAME ">" value WS\n'
            + value
            + "WS: /[ \\t\\r\\n]{0,8}/\n"
            + "NAME: /[^<>\\s=]+/\n"
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


def _declared_tool_names(data: ChatCompletionRequest) -> tuple[str, ...]:
    declarations = data.tools or data.functions or []
    if not declarations:
        raise HTTPException(400, "tool_choice requires at least one tool declaration.")
    names = tuple(function_name(tool) for tool in declarations)
    if any(not isinstance(name, str) or not name or re.search(r"[\s<>=]", name) for name in names):
        raise HTTPException(
            400,
            "Qwen tool_choice requires nonempty function names without whitespace, <, > or =.",
        )
    if len(set(names)) != len(names):
        raise HTTPException(400, "Qwen tool_choice requires unique function names.")
    return names


def auto_tool_grammar(names: tuple[str, ...], parallel: bool, token_ids=None) -> str:
    """Allow free content and zero calls, then constrain every Qwen call opener.

    The free-content prefix states matter even for byte-tokenized openers. A
    greedy text regex can consume half an opener before it knows it must end;
    a catch-all alternative can instead bypass the tool grammar entirely.
    Explicit states keep both choices alive until the opener is complete.
    """
    token_ids = token_ids or {}

    def marker(text):
        quoted = json.dumps(text)
        return f"({quoted} | <[{token_ids[text]}]>)" if text in token_ids else quoted

    # Keep the parameter/value protocol identical to forced tool_choice.
    definitions = ForcedToolChoice(names, parallel).grammar(token_ids).split("\n", 1)[1]
    name_tail = " | ".join(json.dumps(name + ">") for name in names)
    definitions += f"auto_function_tail: ({name_tail}) WS parameter* {marker('</function>')}\n"
    openers = ("<tool_call>", "<function=")
    function_tokens = {
        name: token_ids[f"<function={name}>"] for name in names if f"<function={name}>" in token_ids
    }
    literals = [
        token_id
        for text, token_id in token_ids.items()
        if text not in openers and not text.startswith("<function=")
    ]
    if literals:
        definitions += (
            "auto_literal: "
            + " | ".join(f"<[{token_id}]>" for token_id in sorted(set(literals)))
            + "\n"
        )

    # Each nonterminal records the still-possible prefix of a tool opener.
    # Every prefix is also an accepting plain-text endpoint, except a complete
    # opener, which must finish a valid call before EOS becomes available.
    trie = {}
    for opener in openers:
        node = trie
        for char in opener:
            node = node.setdefault(char, {})
        node[None] = opener
    nodes = []

    def index_nodes(node):
        node["index"] = len(nodes)
        nodes.append(node)
        for char, child in node.items():
            if char not in ("index", None) and None not in child:
                index_nodes(child)

    index_nodes(trie)
    rules = ["start: auto_0_0"]
    for seen in range(1 if parallel else 2):
        root = f"auto_{seen}_0"
        next_root = "auto_0_0" if parallel else "auto_1_0"
        can_call = parallel or not seen

        def call_tail(opener, next_root=next_root):
            body = (
                "WS function WS " + marker("</tool_call>")
                if opener == "<tool_call>"
                else "auto_function_tail"
            )
            return body + " " + next_root

        for node in nodes:
            children = {char: child for char, child in node.items() if char not in ("index", None)}
            alternatives = []
            for char, child in children.items():
                if None in child:
                    if can_call:
                        alternatives.append(json.dumps(char) + " " + call_tail(child[None]))
                else:
                    alternatives.append(json.dumps(char) + f" auto_{seen}_{child['index']}")
            excluded = set(children) | {"<"}
            mismatch = "[^" + "".join(re.escape(char) for char in sorted(excluded)) + "]"
            # Ordinary text is consumed in runs; opener prefixes use one
            # character of lookahead before returning to the content state.
            if node is trie:
                mismatch += "+"
            alternatives.append("/" + mismatch + "/ " + root)
            if "<" not in children:
                alternatives.append(f'"<" auto_{seen}_{trie["<"]["index"]}')
            if literals:
                alternatives.append("auto_literal " + root)
            if can_call:
                for opener in openers:
                    if opener in token_ids:
                        alternatives.append(f"<[{token_ids[opener]}]> " + call_tail(opener))
                for token_id in function_tokens.values():
                    alternatives.append(
                        f"<[{token_id}]> WS parameter* " + marker("</function>") + " " + next_root
                    )
            rules.append(f"auto_{seen}_{node['index']}: (" + " | ".join(alternatives) + ")?")
    return "\n".join(rules) + "\n" + definitions


def _output_constraint_conflicts(data):
    conflicts = [
        key for key in ("grammar_string", "regex_pattern", "json_schema") if getattr(data, key)
    ]
    if data.response_format and data.response_format.type != "text":
        conflicts.append("response_format")
    return conflicts


def resolve_forced_tool_choice(data: ChatCompletionRequest) -> ForcedToolChoice | None:
    choice = data.tool_choice
    if choice is None or choice in ("auto", "none"):
        return None

    names = _declared_tool_names(data)

    if choice != "required":
        selected = choice.function.name
        if selected not in names:
            raise HTTPException(
                400, f"tool_choice names {selected!r}, which is not present in tools."
            )
        names = (selected,)

    return ForcedToolChoice(names, parallel=data.parallel_tool_calls is not False)


def prepare_forced_tool_choice(
    data: ChatCompletionRequest, tool_format: str | None, tokenizer=None
) -> ForcedToolChoice | None:
    """Install the Qwen content grammar, preserving auto selection and explicit constraints."""

    choice = resolve_forced_tool_choice(data)
    if choice is None:
        if (
            data.tool_choice != "none"
            and (data.tools or data.functions)
            and canonical_format_name(tool_format) == "qwen3_coder"
            and not _output_constraint_conflicts(data)
            and not data.continue_final_message
            and not data.response_prefix
        ):
            data.grammar_string = auto_tool_grammar(
                _declared_tool_names(data),
                parallel=data.parallel_tool_calls is not False,
                token_ids=literal_token_ids(tokenizer),
            )
        return None
    if canonical_format_name(tool_format) != "qwen3_coder":
        raise HTTPException(
            400,
            "Forced tool_choice is currently supported for the Qwen pseudo-XML tool format "
            "(qwen3_coder and its aliases). Use tool_choice='auto' for this model.",
        )

    conflicts = _output_constraint_conflicts(data)
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

    data.grammar_string = choice.grammar(literal_token_ids(tokenizer))
    return choice
