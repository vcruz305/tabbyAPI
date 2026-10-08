"""Producer-side reasoning budgets with an explicit legacy fallback boundary."""
from copy import deepcopy
from dataclasses import dataclass
import inspect
from typing import Any


@dataclass(frozen=True)
class NativeReasoningBudget:
    max_tokens: int
    output_ids: Any
    end_token_id: int
    parser: Any = None


def supports_native_reasoning_budget(job_type):
    setter = getattr(job_type, "set_token_budget", None)
    if not callable(setter):
        return False
    try:
        return "can_end" in inspect.signature(setter).parameters
    except (ValueError, TypeError):
        return False


def encode_forced_output(tokenizer, text):
    # Some post-processors prepend BOS even when add_bos=False.
    ids = tokenizer.encode(text, encode_special_tokens=True, add_bos=False)
    if ids.shape[-1] > 0 and ids[0, 0].item() == tokenizer.bos_token_id:
        ids = ids[:, 1:]
    return ids


def prepare_native_reasoning_budget(
    tokenizer, max_tokens, text, *, initial_reasoning, end_token, supported, parser=None
):
    """Only arm an already-active phase with one unambiguous native end token.

    Later literal reasoning markers may occur inside tool argument strings.
    They must not arm a producer budget for a request that starts in content.
    Older engines and other framing formats retain the collector fallback.
    """
    if not supported or not initial_reasoning or not end_token:
        return None
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
        raise ValueError("A native reasoning budget must be a non-negative integer")
    end_id = tokenizer.single_id(end_token)
    if end_id is None:
        return None
    ids = encode_forced_output(tokenizer, text)
    if ids.shape[-1] == 0 or ids[0, -1].item() != end_id:
        return None
    if parser is not None:
        if not callable(getattr(parser, "checkpoint", None)) or not callable(
            getattr(parser, "restore_checkpoint", None)
        ):
            return None
        ending = deepcopy(parser)
        ending.feed(text)
        if ending.in_reasoning or ending.in_tool or ending._pending:
            return None
    return NativeReasoningBudget(max_tokens, ids, end_id, deepcopy(parser))


def producer_phase_end_callback(container, request_id):
    """Switch settings in the producer before it samples any final content."""
    def on_end(native_job):
        active = container.active_job_ids.get(request_id)
        if active is None or getattr(active, "job", None) is not native_job:
            raise RuntimeError("Reasoning budget ended for an inactive generation job")
        # This existing operation respects naturally armed grammar triggers;
        # after forced output it reinstalls suspended content filters. Updating
        # the phase here prevents the later consumer from resetting that grammar.
        if not container.set_generation_phase(request_id, False):
            raise RuntimeError("Could not apply content settings at the reasoning budget boundary")
    return on_end


class ReasoningBoundaryGuard:
    """Track only newly accepted CPU tokens through the authoritative router.

    The engine calls this at a possible phase end or budget injection and also
    immediately after a banned-string rewind. Snapshots are retained only as
    far back as the currently reversible checkpoint, so a long tool argument
    neither triggers full-text rescans nor retains its entire parser history.
    Requeues preserve the full accepted sequence and the original offset.
    """

    def __init__(self, native_job, tokenizer, parser):
        if len(native_job.sequences) != 1:
            raise ValueError("A reasoning boundary guard requires one sequence")
        self._job = native_job
        self._parser = parser
        self._pieces = tokenizer.get_id_to_piece_list(True)
        self._origin = len(native_job.sequences[0].sequence_ids)
        self._cursor = self._origin
        self._states = {self._origin: parser.checkpoint()}
        self._healing_chars = 0
        if native_job.prefix_token is not None and native_job.new_tokens == -1:
            prefix_id = native_job.prefix_token[0].item()
            self._healing_chars = len(self._pieces[prefix_id])

    def __call__(self, native_job):
        if native_job is not self._job:
            raise RuntimeError("Reasoning boundary guard belongs to another job")
        sequence = native_job.sequences[0].sequence_ids
        end = len(sequence)
        if end < self._origin:
            raise RuntimeError("Generation rewound before the guarded output prefix")
        if end < self._cursor:
            state = self._states.get(end)
            if state is None:
                raise RuntimeError("Reasoning parser checkpoint was not retained for this rewind")
            self._parser.restore_checkpoint(state)
            self._cursor = end
            self._states = {position: value for position, value in self._states.items()
                            if position <= end}

        checkpoint = native_job.checkpoint
        keep_from = end - checkpoint["offset"] if checkpoint is not None else end
        if keep_from < self._origin:
            raise RuntimeError("Banned-string checkpoint precedes the guarded output prefix")
        if self._cursor < end:
            ids = sequence.torch_slice(self._cursor, end)
            if ids.device.type != "cpu":
                raise ValueError("Reasoning guard requires already accepted CPU token IDs")
            for offset, token_id in enumerate(ids.reshape(-1).tolist(), self._cursor):
                piece = self._pieces[token_id]
                if offset == self._origin and self._healing_chars:
                    piece = piece[self._healing_chars:]
                self._parser.feed(piece)
                position = offset + 1
                if position >= keep_from:
                    self._states[position] = self._parser.checkpoint()
            self._cursor = end
        self._states = {position: value for position, value in self._states.items()
                        if position >= keep_from}
        # Even an incomplete '<tool_' or '<func' must finish before an injection
        # can be placed. Parameter prefixes are protected by in_tool as well.
        return not self._parser.in_tool and not self._parser._pending
