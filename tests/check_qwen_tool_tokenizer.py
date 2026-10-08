"""Verify forced and auto Qwen grammars with real tokenizer sampling masks.

Example: python -m tests.check_qwen_tool_tokenizer /path/to/tokenizer.json
No model weights or GPU are used. Output records the tokenizer fingerprint.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from llguidance import LLMatcher, LLTokenizer
from tokenizers import Tokenizer

from endpoints.OAI.utils.tool_choice import ForcedToolChoice, auto_tool_grammar, literal_token_ids


def xml(name, value=None):
    parameters = "" if value is None else f"<parameter=code>\n{value}\n</parameter>"
    return f"<tool_call><function={name}>{parameters}</function></tool_call>"


def check(path, eos_text="<|im_end|>"):
    raw = Path(path).read_text()
    hf = Tokenizer.from_str(raw)
    eos_id = hf.token_to_id(eos_text)
    if eos_id is None:
        raise ValueError(f"EOS token {eos_text!r} not found")
    tokenizer = LLTokenizer(raw, eos_token=eos_id)
    added_ids = literal_token_ids(hf)
    fixtures = [
        ("no_args", xml("ping"), ("ping",), False, True),
        ("parallel", xml("ping") + "\n" + xml("weather", "Paris"), ("ping", "weather"), True, True),
        ("parallel_disabled", xml("ping") + xml("ping"), ("ping",), False, False),
        ("wrong_named", xml("ping"), ("weather",), False, False),
        ("no_call", "READY", ("ping",), False, False),
        (
            "literal_markup",
            xml("ping", "<think>x</think> </function> <tool_call></tool_call>"),
            ("ping",),
            False,
            True,
        ),
        ("unicode", xml("ping", "東京 ☀️"), ("ping",), False, True),
        (
            "partial_tag_prefixes",
            xml("ping", "< << </ </paramete> </parameterX>"),
            ("ping",),
            False,
            True,
        ),
        (
            "first_parameter_close",
            xml("ping", "before</parameter>after<think>x"),
            ("ping",),
            False,
            False,
        ),
    ]
    fixtures = [("required", *case) for case in fixtures]
    auto_cases = [
        ("auto_empty", "", ("ping",), True, True),
        ("auto_no_call", "READY", ("ping",), True, True),
        ("auto_unicode_text", "The price is 20 €; 東京 <test> ☀️.", ("ping",), True, True),
        ("auto_long_text", "Here is ordinary text. " * 100, ("ping",), True, True),
        ("auto_partial_opener", "<tool_cal", ("ping",), True, True),
        ("auto_overlap_prefix", "<<tool_cal<function<other>", ("ping",), True, True),
        ("auto_preamble", "I will call it.\n" + xml("ping"), ("ping",), True, True),
        ("auto_postamble", xml("ping") + "\nDone.", ("ping",), True, True),
        ("auto_bare", "<function=ping></function>", ("ping",), True, True),
        ("auto_parallel", xml("ping") + xml("weather"), ("ping", "weather"), True, True),
        ("auto_parallel_disabled", xml("ping") + xml("ping"), ("ping",), False, False),
        (
            "auto_bare_parallel_disabled",
            xml("ping") + "<function=ping></function>",
            ("ping",),
            False,
            False,
        ),
        (
            "auto_literal_markup",
            xml("ping", "<think>literal</think> </function> <tool_call></tool_call>"),
            ("ping",),
            True,
            True,
        ),
        ("auto_wrong_name", xml("absent"), ("ping",), True, False),
        ("auto_overlap_wrong_name", "x<" + xml("absent"), ("ping",), True, False),
        ("auto_incomplete_wrapper", xml("ping") + "<tool_call>", ("ping",), True, False),
        ("auto_incomplete_function", "<function=ping>", ("ping",), True, False),
        ("auto_native_message_token", xml("ping", "<|im_start|>literal"), ("ping",), True, False),
    ]
    fixtures += [("auto", *case) for case in auto_cases]
    fixtures = [(*case, frozenset()) for case in fixtures]
    # Closed empty object schemas must prevent invented parameters. All other
    # function/value paths remain identical to the existing protocol above.
    closed_cases = [
        ("required", "closed_no_args", xml("ping"), ("ping",), False, True),
        ("required", "closed_invented_arg", xml("ping", ""), ("ping",), False, False),
        (
            "required", "closed_and_open_functions", xml("ping") + xml("weather", "Paris"),
            ("ping", "weather"), True, True,
        ),
        ("named", "closed_named_no_args", xml("ping"), ("ping",), False, True),
        ("named", "closed_named_invented_arg", xml("ping", "wrong"), ("ping",), False, False),
        ("auto", "closed_auto_no_args", xml("ping"), ("ping",), True, True),
        ("auto", "closed_auto_invented_arg", xml("ping", ""), ("ping",), True, False),
        ("auto", "closed_auto_bare", "<function=ping></function>", ("ping",), True, True),
        (
            "auto", "closed_auto_bare_invented_arg",
            "<function=ping><parameter=__noargs></parameter></function>",
            ("ping",), True, False,
        ),
        ("auto", "closed_auto_zero_calls", "READY", ("ping",), False, True),
    ]
    fixtures += [(*case, frozenset({"ping"})) for case in closed_cases]
    results = []
    for mode, name, text, names, parallel, expected, no_argument_names in fixtures:
        grammar = (
            auto_tool_grammar(names, parallel, added_ids, no_argument_names=no_argument_names)
            if mode == "auto"
            else ForcedToolChoice(names, parallel, no_argument_names).grammar(added_ids)
        )
        error = LLMatcher.validate_grammar(grammar, tokenizer)
        if error:
            raise RuntimeError(error)
        matcher = LLMatcher(tokenizer, grammar, log_level=0)
        tokens = tokenizer.tokenize_str(text)
        start = perf_counter()
        valid = True
        for token_id in tokens:
            # Check the mask the sampler will actually see, not just parser acceptance.
            mask = matcher.compute_bitmask()
            if not (mask[token_id // 8] & (1 << (token_id % 8))):
                valid = False
                break
            if not matcher.consume_token(token_id):
                valid = False
                break
        valid = valid and matcher.is_accepting()
        if valid:
            mask = matcher.compute_bitmask()
            if not (mask[eos_id // 8] & (1 << (eos_id % 8))):
                raise AssertionError(f"{name}: complete call does not allow EOS")
            if not matcher.consume_token(eos_id) or not matcher.is_stopped():
                raise AssertionError(f"{name}: EOS did not finish the grammar")
        if valid != expected:
            raise AssertionError(
                f"{name}: expected acceptance={expected}, got {valid}; {matcher.get_error()}"
            )
        results.append(
            {
                "case": name,
                "tool_choice": mode,
                "no_argument_names": sorted(no_argument_names),
                "accepted": valid,
                "expected": expected,
                "tokens": len(tokens),
                "seconds": perf_counter() - start,
            }
        )
    # Reproduce the exact kind of prefix from the live failing synthetic call.
    # Literal think markers must remain legal while conversation boundaries and
    # premature EOS are masked out. Checking strings alone misses native IDs.
    prefix = (
        "<tool_call>\n<function=record_strings>\n"
        "<parameter=number_text>\n123\n</parameter>\n"
        "<parameter=bool_text>\ntrue\n</parameter>\n"
        '<parameter=json_text>\n{"nested": [1, false]}\n</parameter>\n'
        "<parameter=tag_text>\n"
    )
    grammar = auto_tool_grammar(("record_strings",), True, added_ids)
    matcher = LLMatcher(tokenizer, grammar, log_level=0)
    if not matcher.consume_tokens(tokenizer.tokenize_str(prefix)):
        raise AssertionError("captured synthetic argument prefix is not allowed")
    mask = matcher.compute_bitmask()
    prefix_masks = []
    for text, expected in [("<think>", True), ("<|im_start|>", False), (eos_text, False)]:
        token_id = hf.token_to_id(text)
        if token_id is None:
            raise ValueError(f"Required diagnostic token {text!r} not found")
        allowed = bool(mask[token_id // 8] & (1 << (token_id % 8)))
        if allowed != expected:
            raise AssertionError(f"captured prefix: {text} allowed={allowed}, expected {expected}")
        prefix_masks.append(
            {"token": text, "token_id": token_id, "allowed": allowed, "expected": expected}
        )
    return {
        "tokenizer_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": eos_id,
        "added_literal_tokens": added_ids,
        "cases": results,
        "captured_prefix_masks": prefix_masks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer")
    parser.add_argument("--eos-token", default="<|im_end|>")
    args = parser.parse_args()
    print(json.dumps(check(args.tokenizer, args.eos_token), indent=2))


if __name__ == "__main__":
    main()
