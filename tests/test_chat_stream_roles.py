"""Every streamed choice starts an assistant message that SDKs can round-trip."""

import asyncio
import unittest
from unittest.mock import patch

from endpoints.OAI.utils import chat_completion as cc
from tests.test_qwen_tool_contract import xml
from tests.test_tool_delta_streaming import make_mc, make_request, pieces, run_collector


class ChatStreamRoleTests(unittest.IsolatedAsyncioTestCase):
    async def test_role_precedes_empty_plain_reasoning_and_tool_completions(self):
        cases = [
            [],
            ["Hello!"],
            pieces(xml("get_weather", [("city", "Paris")])),
            pieces("<think>plan</think>" + xml("get_weather", [("city", "Paris")])),
        ]
        for chunks in cases:
            with self.subTest(chunks=chunks[:2]):
                mc = make_mc(
                    chunks, reasoning_start_token="<think>", reasoning_end_token="</think>"
                )
                frames, _ = await run_collector(mc, make_request())
                payloads = [
                    cc._compose_serialize_stream_chunk("req", frame, "model")[1] for frame in frames
                ]
                deltas = [payload["choices"][0]["delta"] for payload in payloads]
                self.assertEqual(deltas[0], {"role": "assistant"})
                self.assertEqual(sum("role" in delta for delta in deltas), 1)
                self.assertTrue(payloads[-1]["choices"][0]["finish_reason"])

    async def test_each_parallel_choice_gets_one_role_before_its_content(self):
        queue = asyncio.Queue()
        mc = make_mc(["hello"])
        with patch.object(cc, "model") as mocked:
            mocked.container = mc
            await asyncio.gather(
                *[
                    cc._chat_stream_collector(
                        index,
                        queue,
                        f"req-{index}",
                        "PROMPT",
                        make_request(),
                        False,
                        None,
                        True,
                        None,
                    )
                    for index in range(3)
                ]
            )
        by_index = {}
        while not queue.empty():
            frame = queue.get_nowait()
            payload = cc._compose_serialize_stream_chunk("req", frame, "model")[1]
            choice = payload["choices"][0]
            by_index.setdefault(choice["index"], []).append(choice["delta"])
        self.assertEqual(set(by_index), {0, 1, 2})
        for deltas in by_index.values():
            self.assertEqual(deltas[0], {"role": "assistant"})
            self.assertEqual(sum("role" in delta for delta in deltas), 1)

    async def test_role_precedes_prefill_progress(self):
        mc = make_mc([])

        async def backend(*args, **kwargs):
            yield {"_prefill_progress": {"total": 100, "processed": 10}}
            yield {"text": "", "finish_reason": "stop"}

        mc.stream_generate = backend
        frames, _ = await run_collector(mc, make_request())
        self.assertEqual(frames[0]["delta_role"], "assistant")
        self.assertIn("_prefill_progress", frames[1])


if __name__ == "__main__":
    unittest.main()
