"""Verify forced Qwen grammar with a real tokenizer, including its token masks.

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

from endpoints.OAI.utils.tool_choice import ForcedToolChoice, literal_token_ids


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
    results = []
    for name, text, names, parallel, expected in fixtures:
        grammar = ForcedToolChoice(names, parallel).grammar(added_ids)
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
                "accepted": valid,
                "expected": expected,
                "tokens": len(tokens),
                "seconds": perf_counter() - start,
            }
        )
    return {
        "tokenizer_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": eos_id,
        "added_literal_tokens": added_ids,
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer")
    parser.add_argument("--eos-token", default="<|im_end|>")
    args = parser.parse_args()
    print(json.dumps(check(args.tokenizer, args.eos_token), indent=2))


if __name__ == "__main__":
    main()
