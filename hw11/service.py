"""HW8–10 FastAPI contract with an explicit, bounded simulated backend."""

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
import hashlib
import json
import os
import signal
from pathlib import Path
import time
from typing import Literal
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field, StringConstraints
from typing_extensions import Annotated

BUCKETS = (.025, .05, .075, .1, .15, .2, .3, .45, .65, .85, 1.1, 1.5,
           2, 3, 5, 8, 12, 20, 35)
TTFC = Histogram("hw11_ttfc_seconds", "First chunk on completed responses", buckets=BUCKETS)
TOTAL = Histogram("hw11_total_seconds", "Duration of completed responses", buckets=BUCKETS)
QUEUE = Histogram("hw11_queue_seconds", "Queue duration on completed responses", buckets=BUCKETS)
OUTCOME = Counter("hw11_requests", "Request outcomes", ["outcome"])
RUNNING = Gauge("hw11_running", "Processing slots in use")
WAITING = Gauge("hw11_waiting", "Requests awaiting processing")
WARM = Gauge("hw11_warm", "Verified artifact and completed warmup")
for outcome in ("ok", "overloaded", "warming", "aborted"):
    OUTCOME.labels(outcome)

Text = Annotated[str, StringConstraints(min_length=1, max_length=16000)]


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Text


class Payload(BaseModel):
    prompt: Text
    history: list[Message] = Field(default_factory=list, max_length=16)
    session_id: str = Field(default="", max_length=80)
    stream: bool = False
    max_new_tokens: int = Field(default=4, ge=1, le=32)
    chunk_ms: int = Field(default=120, ge=0, le=750)
    cpu_ms: int = Field(default=0, ge=0, le=400)


def burn(milliseconds: int):
    deadline = time.thread_time() + milliseconds / 1000
    data = b"hw11-cpu"
    while time.thread_time() < deadline:
        for _ in range(128):
            data = hashlib.sha256(data).digest()


def emit(event, **fields):
    print(json.dumps({"ts": time.time(), "event": event, **fields}), flush=True)


class SimulatedBackend:
    def __init__(self):
        self.version = None
        self.sha = None
        self.warm = False
        self.pod = os.getenv("POD_NAME", "local")
        self.probe_mode = os.getenv("PROBE_MODE", "correct")
        self.slots = asyncio.Semaphore(int(os.getenv("SLOTS", "1")))
        self.capacity = int(os.getenv("SLOTS", "1")) + int(os.getenv("QUEUE_LIMIT", "5"))
        self.admitted = 0
        self.cache = OrderedDict()
        self.cache_size = int(os.getenv("CACHE_SIZE", "32"))
        self.hit_seconds = float(os.getenv("HIT_MS", "35")) / 1000
        self.miss_seconds = float(os.getenv("MISS_MS", "180")) / 1000

    def identity(self):
        return {"pod": self.pod, "model_id": self.version, "runtime_sha256": self.sha,
                "backend_mode": "simulated"}

    async def load(self):
        try:
            artifact = Path(os.getenv("MODEL_PATH", "chart/models/A.json")).read_bytes()
            digest = hashlib.sha256(artifact).hexdigest()
            if digest != os.getenv("MODEL_SHA256", digest):
                raise ValueError("artifact checksum mismatch")
            model = json.loads(artifact)
            if model["model_id"] != os.getenv("MODEL_VERSION", model["model_id"]):
                raise ValueError("artifact version mismatch")
            self.version, self.sha = model["model_id"], digest
            emit("loaded", **self.identity())
            await asyncio.sleep(float(os.getenv("WARMUP_SECONDS", "20")))
            self.warm = True
            WARM.set(1)
            emit("warm", **self.identity())
        except (OSError, ValueError, KeyError) as exc:
            emit("load_failed", error=str(exc), **self.identity())

    def cache_hit(self, payload):
        prefix = payload.history[0].content if payload.history else payload.prompt.split("|", 1)[0]
        key = hashlib.sha256(prefix.encode()).digest()
        hit = key in self.cache
        self.cache[key] = None
        self.cache.move_to_end(key)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return hit


@asynccontextmanager
async def lifespan(app):
    backend = SimulatedBackend()
    WARM.set(0)
    app.state.backend = backend
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGUSR1, backend.cache.clear)
    loader = asyncio.create_task(backend.load())
    yield
    loop.remove_signal_handler(signal.SIGUSR1)
    loader.cancel()
    with suppress(asyncio.CancelledError):
        await loader


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"alive": True}


@app.get("/startup")
@app.get("/ready")
async def ready(request: Request):
    backend = request.app.state.backend
    ready_now = backend.warm or backend.probe_mode == "http-only"
    return JSONResponse({"ready": ready_now, "warm": backend.warm, **backend.identity()},
                        status_code=200 if ready_now else 503)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})


@app.post("/generate")
async def generate(request: Request, payload: Payload):
    backend = request.app.state.backend
    start = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", "")[:128] or str(uuid.uuid4())
    identity = {"request_id": request_id, **backend.identity()}
    headers = {"X-Request-ID": request_id, "X-Pod": backend.pod,
               "X-Model-ID": backend.version or "", "X-Model-SHA256": backend.sha or "",
               "X-Backend-Mode": "simulated"}
    peer = request.client.host if request.client else None
    log_fields = dict(identity, session_id=payload.session_id, peer=peer)
    emit("received", warm=backend.warm, **log_fields)
    rejection = None
    if not backend.warm:
        rejection = "warming"
    elif backend.admitted >= backend.capacity:
        rejection = "overloaded"
    if rejection:
        OUTCOME.labels(rejection).inc()
        emit("finished", outcome=rejection, **log_fields)
        headers["X-Rejection"] = rejection
        code = 429 if rejection == "overloaded" else 503
        return JSONResponse(dict(identity, outcome=rejection), status_code=code, headers=headers)

    backend.admitted += 1

    async def chunks():
        WAITING.inc()
        acquired = False
        completed = False
        try:
            await backend.slots.acquire()
            acquired = True
            WAITING.dec()
            RUNNING.inc()
            queue = time.perf_counter() - start
            hit = backend.cache_hit(payload)
            await asyncio.sleep(backend.hit_seconds if hit else backend.miss_seconds)
            if payload.cpu_ms:
                await asyncio.to_thread(burn, payload.cpu_ms)
            ttfc = time.perf_counter() - start
            context = payload.history[-1].content if payload.history else payload.prompt
            metadata = dict(identity, queue_seconds=queue, cache_hit=hit, peer=peer)
            for index in range(payload.max_new_tokens):
                if index:
                    await asyncio.sleep(payload.chunk_ms / 1000)
                yield dict(metadata, index=index, text=context if index == 0 else f"part-{index}")
            yield dict(metadata, done=True, parts=payload.max_new_tokens)
            completed = True
            total = time.perf_counter() - start
            TTFC.observe(ttfc)
            TOTAL.observe(total)
            QUEUE.observe(queue)
        finally:
            backend.admitted -= 1
            if acquired:
                RUNNING.dec()
                backend.slots.release()
            else:
                WAITING.dec()
            outcome = "ok" if completed else "aborted"
            OUTCOME.labels(outcome).inc()
            emit("finished", outcome=outcome, **log_fields)

    if payload.stream:
        async def ndjson():
            iterator = chunks()
            try:
                async for chunk in iterator:
                    yield json.dumps(chunk, ensure_ascii=False) + "\n"
            finally:
                await iterator.aclose()

        return StreamingResponse(ndjson(), media_type="application/x-ndjson", headers=headers)

    parts = [chunk async for chunk in chunks()]
    answer = " ".join(part["text"] for part in parts[:-1])
    return JSONResponse(dict(parts[-1], answer=answer, text=answer,
                             total_seconds=time.perf_counter() - start), headers=headers)
