"""
Incremental streaming of OpenAI tool_calls deltas for the qwen3_coder
pseudo-XML tool call format.

The end-of-stream parser (toolcall_formats/qwen3_coder.py) collects the full
tool text and parses it in one pass, so streaming clients receive the
complete tool call as a single delta after generation finishes and live UIs
cannot show tool generation as it happens.

QwenToolCallDeltaStreamer consumes TOOL-channel text as it arrives (from
TagStreamParser, which keeps the format's own wrapper tags, TOOLCALL_START
and TOOLCALL_END, in the tool channel) and emits OpenAI-style tool_calls
delta fragments:

  - when a function opens: {index, id, type, function: {name, arguments: "{"}}
  - when a parameter closes: {index, function: {arguments: <json fragment>}}
  - when the function closes: {index, function: {arguments: "}"}}

Declared string parameter values stream as JSON strings while they are
generated, including values that look like numbers, booleans or JSON source.
Without an unambiguous string schema, text streams as soon as it provably
cannot parse as a JSON literal (see _is_streamable); other values are buffered
until their parameter closes. Raw values and JSON argument fragments are
accumulated in lists so long code payloads do not require repeated full copies.

Guarantee: for well-formed calls, the concatenation of all "arguments"
fragments per index is byte-identical to the arguments string produced by
the end-of-stream parser, so the assembled tool call is unchanged. Malformed
model output is handled the way the end-of-stream regexes read it (stray
close tags and parameters outside a function block are dropped, a nested
function open belongs to the outer call). Fragments cannot be retracted, so
three inputs diverge on purpose and are reported by verify():

  - a call that generation never closes (cut off by max_tokens): the regex
    drops the whole call, the client already has the partial arguments
  - duplicate parameter keys: both are streamed, the parser keeps the last
  - a value containing a literal parameter close tag: both sides close the
    value at the first occurrence, but the remainder is read differently
"""

import json
import re
from typing import Optional
from uuid import uuid4

from common.logger import xlogger
from endpoints.OAI.utils.toolcall_formats import qwen3_coder

_FUNC_OPEN = re.compile(r"<function=([^>\s]+)[^>]*>")
_PARAM_OPEN = re.compile(r"<parameter=([^>\s]+)[^>]*>")

_FUNC_CLOSE = "</function>"
_PARAM_CLOSE = "</parameter>"

# First characters whose stripped value may still json.loads() to something
# other than the exact input string, per ToolSchemas.coerce(). Such values
# are emitted whole at parameter close instead of being streamed.
_JSONY_START = set('"{[0123456789-')

# t/f/n: only the exact keywords parse as JSON; streaming may start as soon
# as the accumulated value diverges from them.
_JSON_KEYWORDS = ("true", "false", "null")


def _is_streamable(stripped: str) -> bool:
    """
    True once the stripped value accumulated so far provably cannot parse as
    a JSON literal, i.e. ToolSchemas.coerce() will return it as a plain
    string and json.dumps() will quote it exactly as it is streamed.
    """

    if not stripped:
        return False
    if stripped[0] in _JSONY_START:
        return False
    for keyword in _JSON_KEYWORDS:
        if keyword.startswith(stripped):
            return False
    return True


def _esc(text: str) -> str:
    """JSON-escape a text fragment without surrounding quotes."""

    return json.dumps(text, ensure_ascii=False)[1:-1]


class QwenToolCallDeltaStreamer:
    """
    Emits OAI tool_calls deltas incrementally for qwen3_coder pseudo-XML.

    feed() accepts TOOL-channel text and returns a list of delta dicts to be
    sent as one streaming frame (empty list when nothing is ready). emitted
    is True once any fragment was produced; verify() should be called at the
    end of the stream with the full tool text to cross-check the assembled
    calls against the authoritative end-of-stream parser.
    """

    _OUT = 0
    _VALUE = 1

    def __init__(self, tools=None, max_calls=None, validate_name=None):
        self.schemas = qwen3_coder.ToolSchemas(tools)
        self.max_calls = max_calls
        self.validate_name = validate_name
        self.completed = 0
        self.emitted = False

        self._state = self._OUT
        self._buf = ""

        self._index = -1
        self._in_func = False
        self._first_param = True
        self._keys: set[str] = set()

        self._param_key: Optional[str] = None
        self._reset_value()

        self._names: list[str] = []
        self._argument_fragments: list[list[str]] = []

    @property
    def _assembled(self) -> list[str]:
        # Materialize once at verification, not once per generated fragment.
        return ["".join(parts) for parts in self._argument_fragments]

    @_assembled.setter
    def _assembled(self, values: list[str]):
        self._argument_fragments = [[value] for value in values]

    # -- emission helpers

    def _open_function(self, name: str) -> dict:
        if self.validate_name is not None:
            self.validate_name(name, self._index + 1)
        self._index += 1
        self._first_param = True
        self._keys = set()
        self._argument_fragments.append(["{"])
        self._names.append(name)
        return {
            "index": self._index,
            "id": f"call_{uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": "{"},
        }

    def _param_prefix(self) -> str:
        key = json.dumps(self._param_key, ensure_ascii=False)
        if self._first_param:
            self._first_param = False
            return key + ": "
        return ", " + key + ": "

    def _arg_fragment(self, fragment: str) -> dict:
        self._argument_fragments[self._index].append(fragment)
        return {"index": self._index, "function": {"arguments": fragment}}

    def _emit(self, deltas: list, delta: dict):
        if self.max_calls is not None and delta["index"] >= self.max_calls:
            return
        self.emitted = True

        # Merge consecutive fragments for the same index into one entry so a
        # frame never carries two deltas for the same tool call.
        if deltas and deltas[-1].get("index") == delta.get("index"):
            prev = deltas[-1]
            prev_fn = prev.get("function", {})
            new_fn = delta.get("function", {})
            merged_fn = dict(prev_fn)
            for field, value in new_fn.items():
                if field == "arguments":
                    merged_fn["arguments"] = merged_fn.get("arguments", "") + value
                else:
                    merged_fn[field] = value
            merged = dict(prev)
            merged["function"] = merged_fn
            deltas[-1] = merged
        else:
            deltas.append(delta)

    # -- buffer management

    @staticmethod
    def _partial_open_hold(buf: str) -> int:
        """Length of a trailing incomplete tag that must be kept in OUT."""

        for literal in ("<function=", "<parameter="):
            pos = buf.rfind(literal)
            if pos >= 0 and buf.find(">", pos) < 0:
                return len(buf) - pos
        for literal in ("<function=", "<parameter=", _FUNC_CLOSE, _PARAM_CLOSE):
            for k in range(min(len(literal) - 1, len(buf)), 0, -1):
                if literal.startswith(buf[-k:]):
                    return k
        return 0

    @staticmethod
    def _partial_close_hold(buf: str) -> int:
        """Length of a trailing text run that could still become a close tag."""

        for k in range(min(len(_PARAM_CLOSE) - 1, len(buf)), 0, -1):
            if _PARAM_CLOSE.startswith(buf[-k:]):
                return k
        return 0

    # -- main entry point

    def feed(self, text: str) -> list:
        deltas: list = []
        self._buf += text

        while True:
            if self._state == self._VALUE:
                close = self._buf.find(_PARAM_CLOSE)
                if close < 0:
                    hold = self._partial_close_hold(self._buf)
                    consume = self._buf[: len(self._buf) - hold] if hold else self._buf
                    self._buf = self._buf[len(self._buf) - hold :] if hold else ""
                    self._consume_value(consume, deltas)
                    break

                self._consume_value(self._buf[:close], deltas)
                self._buf = self._buf[close + len(_PARAM_CLOSE) :]
                self._close_param(deltas)
                self._state = self._OUT
                continue

            # _OUT: advance to the earliest structural tag
            func = _FUNC_OPEN.search(self._buf)
            param = _PARAM_OPEN.search(self._buf)
            fclose = self._buf.find(_FUNC_CLOSE)

            candidates = []
            if func is not None and (fclose < 0 or func.start() < fclose):
                candidates.append(("func", func.start(), func.end(), func.group(1)))
            if param is not None and (fclose < 0 or param.start() < fclose):
                candidates.append(("param", param.start(), param.end(), param.group(1)))
            if fclose >= 0:
                candidates.append(("fclose", fclose, fclose + len(_FUNC_CLOSE), None))

            if not candidates:
                hold = self._partial_open_hold(self._buf)
                self._buf = self._buf[len(self._buf) - hold :] if hold else ""
                break

            kind, _start, end, name = min(candidates, key=lambda c: c[1])
            self._buf = self._buf[end:]

            if kind == "func":
                if self._in_func:
                    # A nested open is body text of the outer call for the
                    # end-of-stream regex, whose name is the outer one. Keep
                    # the outer call and drop this tag.
                    continue
                self._in_func = True
                self._emit(deltas, self._open_function(name))
            elif kind == "fclose":
                if not self._in_func:
                    # Stray close: outside a function block the end-of-stream
                    # regex drops it, so consume it silently here too.
                    continue
                self._emit(deltas, self._arg_fragment("}"))
                self._in_func = False
                self.completed += 1
                self._state = self._OUT
            elif not self._in_func:
                # A parameter outside a function block is dropped by the
                # end-of-stream regex; consume it silently.
                continue
            else:
                self._param_key = name.strip()
                if self._param_key in self._keys:
                    # The end-of-stream parser keeps the last occurrence; the
                    # duplicate key we stream still parses to the same dict.
                    xlogger.debug(
                        "Duplicate parameter in tool call stream",
                        {"function": self._names[self._index], "key": self._param_key},
                    )
                self._keys.add(self._param_key)
                self._reset_value()
                types = self.schemas.types(self._names[self._index], self._param_key)
                self._value_is_string = types == {"string"}
                self._value_allows_string = "string" in types
                self._state = self._VALUE

        return deltas

    # -- value handling

    def _reset_value(self):
        self._raw_parts: list[str] = []
        self._sent_parts: list[str] = []
        self._streaming = False
        self._value_is_string = False
        self._value_allows_string = False
        self._value_started = False
        self._string_tail = ""
        self._pending_whitespace: list[str] = []
        self._probe = ""
        self._probe_whitespace = False
        self._json_value = False

    def _stream_text(self, text: str, deltas: list):
        if not text:
            return
        fragment = _esc(text)
        if not self._streaming:
            self._streaming = True
            fragment = self._param_prefix() + '"' + fragment
        self._sent_parts.append(text)
        self._emit(deltas, self._arg_fragment(fragment))

    def _consume_string_text(self, text: str, deltas: list):
        # normalize_string removes exactly one initial/final LF. Only the
        # possible final LF needs holdback; spaces, indentation and CR are data.
        if not self._value_started:
            self._value_started = True
            text = text.removeprefix("\n")
        text = self._string_tail + text
        self._string_tail = "\n" if text.endswith("\n") else ""
        if self._string_tail:
            text = text[:-1]
        self._stream_text(text, deltas)

    def _consume_value(self, text: str, deltas: list):
        if not text:
            return
        self._raw_parts.append(text)

        if self._value_is_string:
            self._consume_string_text(text, deltas)
            return

        if self._streaming:
            # Schema-free strings use strip(). Keep trailing whitespace until
            # another non-whitespace character makes it interior to the value.
            core = text.rstrip()
            core_len = len(core)
            if core:
                if self._pending_whitespace:
                    core = "".join(self._pending_whitespace) + core
                    self._pending_whitespace = []
                self._stream_text(core, deltas)
            tail = text[core_len:]
            if tail:
                self._pending_whitespace.append(tail)
            return

        if self._json_value:
            return  # JSON-looking values are buffered once, parsed at close.

        # An unresolved prefix is empty or one of t/tr/tru/true/f/.../null.
        # Track it and whether whitespace followed it using bounded state.
        # Numbers, arrays, objects and quoted JSON remain buffered. Once text
        # diverges from a JSON keyword, it can never become JSON later.
        interior_whitespace = self._probe_whitespace and bool(text.strip())
        candidate = (self._probe + text).strip()
        if not candidate:
            return
        if interior_whitespace or _is_streamable(candidate):
            raw = "".join(self._raw_parts)
            if self._value_allows_string:
                self._value_is_string = True
                self._consume_string_text(raw, deltas)
            else:
                self._stream_text(raw.strip(), deltas)
                tail = raw[len(raw.rstrip()) :]
                self._pending_whitespace = [tail] if tail else []
        elif candidate[0] in _JSONY_START:
            self._json_value = True
        else:
            self._probe = candidate
            self._probe_whitespace = text[-1].isspace()

    def _close_param(self, deltas: list):
        raw = "".join(self._raw_parts)
        value = self.schemas.coerce(raw, self._names[self._index], self._param_key)
        if self._streaming:
            sent = "".join(self._sent_parts)
            if value.startswith(sent) and len(value) > len(sent):
                self._stream_text(value[len(sent) :], deltas)
                sent = value
            if value != sent:
                xlogger.error(
                    "Tool-call delta stream diverged from value at parameter close",
                    {"function": self._names[self._index], "key": self._param_key},
                )
            self._emit(deltas, self._arg_fragment('"'))
        else:
            self._emit(
                deltas,
                self._arg_fragment(
                    self._param_prefix() + json.dumps(value, ensure_ascii=False, allow_nan=False)
                ),
            )

        self._param_key = None
        self._reset_value()

    # -- end-of-stream cross-check

    def verify(self, full_tool: str, request_id: str):
        """
        Compare the streamed assembly against the authoritative end-of-stream
        parse. Log-only: streamed fragments cannot be retracted.

        Byte-identical for well-formed calls. Degenerate inputs (duplicate
        parameter keys) may differ in bytes but must parse to the same calls,
        and a call that generation was cut off inside is reported as an error,
        because the parser drops a call that never closes.
        """

        try:
            authoritative = qwen3_coder.parse_toolcalls(full_tool, tools=self.schemas)
            # strict=False: a length mismatch here is itself a divergence,
            # and is reported by the comparison below rather than raised here
            mine = list(zip(self._names, self._assembled, strict=False))
            theirs = [(c.function.name, c.function.arguments) for c in authoritative]
            if mine == theirs:
                return

            parsed_equal = len(mine) == len(theirs) and all(
                n1 == n2 and json.loads(a1) == json.loads(a2)
                for (n1, a1), (n2, a2) in zip(mine, theirs, strict=False)
            )
            if parsed_equal:
                xlogger.debug(
                    f"Tool-call delta stream differs byte-wise but parses equal "
                    f"for request {request_id} (duplicate parameter keys?)",
                    {"streamed": mine, "authoritative": theirs},
                )
            else:
                # The common cause in practice: generation was cut off inside a
                # call, and the end-of-stream regex drops a function block that
                # never closes. Say so, because the fragments were already sent.
                hint = ""
                if len(mine) > len(theirs):
                    hint = (
                        " (more calls streamed than parsed: the last call was "
                        "probably truncated, and the end-of-stream parser drops "
                        "a function block without a closing tag)"
                    )
                xlogger.error(
                    f"Tool-call delta stream differs from end-of-stream parse "
                    f"for request {request_id}{hint}",
                    {"streamed": mine, "authoritative": theirs},
                )
        except Exception as exc:  # never fail the stream on verification
            xlogger.debug(f"Tool-call delta verification skipped: {exc}")
