"""Accounting checks for errors and incomplete NDJSON streams."""

import contextlib
import io
import json
import time
import unittest

import httpx
from scripts.client import attempt


class AccountingTests(unittest.IsolatedAsyncioTestCase):
    async def send(self, response, timeout=1):
        transport = httpx.MockTransport(lambda request: response)
        clock = time.time() - time.monotonic()
        payload = {"prompt": "test", "max_new_tokens": 2, "stream": True}
        log = io.StringIO()
        async with httpx.AsyncClient(transport=transport) as client:
            with contextlib.redirect_stdout(log):
                await attempt(client, "http://test", {"id": "case", "planned_at": time.time(),
                                                       "payload": payload, "timeout": timeout}, clock)
        return json.loads(log.getvalue())

    async def test_refusal_has_no_chunk_or_ttfc(self):
        row = await self.send(httpx.Response(429, headers={"X-Rejection": "overloaded"}))
        self.assertEqual((row["outcome"], row["parts"], row["ttfc_s"], row["rejection"]),
                         ("rejected", 0, None, "overloaded"))

    async def test_200_without_terminal_frame_is_aborted(self):
        partial = {"request_id": "case", "index": 0, "text": "first",
                   "queue_seconds": .1, "peer": "10.0.0.1", "cache_hit": False}
        row = await self.send(httpx.Response(200, text=json.dumps(partial) + "\n"))
        self.assertEqual((row["outcome"], row["parts"], row["complete"]), ("aborted", 1, False))
        self.assertIsNotNone(row["ttfc_s"])

    async def test_completed_frame_requires_the_declared_number_of_parts(self):
        wrong_terminal = {"request_id": "case", "done": True, "parts": 2}
        row = await self.send(httpx.Response(200, text=json.dumps(wrong_terminal) + "\n"))
        self.assertEqual(row["outcome"], "aborted")
