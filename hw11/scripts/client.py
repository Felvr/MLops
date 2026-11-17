"""Open-loop NDJSON client. Each row describes one attempt with no retries."""
import argparse
import asyncio
import json
import sys
import time

import httpx


def classify(status, complete, timeout=False, partial=False):
    if timeout:
        return "timed_out"
    if status is not None and status >= 400 and not partial:
        return "rejected"
    return "completed" if status == 200 and complete else "aborted"


async def attempt(client, target, row, clock_offset):
    due = row["planned_at"] - clock_offset
    await asyncio.sleep(max(0, due - time.monotonic()))
    sent = time.monotonic()
    record = dict(row, target=target, sent_at=sent + clock_offset, send_lag_s=sent - due,
                  sent_monotonic=sent, first_monotonic=None, chunk_monotonic=[],
                  status=None, pod=None, model_id=None, runtime_sha256=None, peer=None,
                  backend_mode=None, queue_seconds=None, cache_hit=None, text=None,
                  rejection=None, error=None, complete=False)
    timed_out = False
    try:
        async with asyncio.timeout(row.get("timeout", 32)):
            async with client.stream("POST", target + "/generate", json=row["payload"],
                                     headers={"X-Request-ID": row["id"]}) as response:
                record.update(status=response.status_code, pod=response.headers.get("x-pod"),
                              model_id=response.headers.get("x-model-id"),
                              runtime_sha256=response.headers.get("x-model-sha256"),
                              backend_mode=response.headers.get("x-backend-mode"),
                              rejection=response.headers.get("x-rejection"))
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        frame = json.loads(line)
                        if frame.get("request_id") != row["id"]:
                            raise ValueError("request ID mismatch")
                        if frame.get("done"):
                            expected = row["payload"].get("max_new_tokens", 4)
                            record["complete"] = (frame["parts"] == expected
                                                  == len(record["chunk_monotonic"]))
                            break
                        now = time.monotonic()
                        if frame["index"] != len(record["chunk_monotonic"]):
                            raise ValueError("unexpected chunk index")
                        record["chunk_monotonic"].append(now)
                        if record["first_monotonic"] is None:
                            record.update(first_monotonic=now, queue_seconds=frame["queue_seconds"],
                                          peer=frame["peer"], cache_hit=frame["cache_hit"],
                                          text=frame["text"])
                else:
                    await response.aread()
    except (TimeoutError, httpx.TimeoutException) as exc:
        timed_out = True
        record["error"] = type(exc).__name__
    except (ValueError, KeyError, httpx.HTTPError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    end = time.monotonic()
    first = record["first_monotonic"]
    record.update(end_monotonic=end, ended_at=end + clock_offset, elapsed_s=end - sent,
                  first_at=None if first is None else first + clock_offset,
                  ttfc_s=None if first is None else first - sent,
                  parts=len(record["chunk_monotonic"]),
                  outcome=classify(record["status"], record["complete"], timed_out,
                                   partial=first is not None))
    print(json.dumps(record, ensure_ascii=False), flush=True)


async def run(target, schedule):
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=0)
    offset = time.time() - time.monotonic()
    async with httpx.AsyncClient(timeout=None, limits=limits, trust_env=False) as client:
        await asyncio.gather(*(attempt(client, target, row, offset) for row in schedule))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target")
    args = parser.parse_args()
    asyncio.run(run(args.target, json.load(sys.stdin)))
