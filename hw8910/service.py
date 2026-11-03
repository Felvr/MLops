"""FastAPI with background load/warmup and immutable loaded-version identity."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import threading
import time
import uuid
from fastapi import FastAPI, HTTPException, Request
from backends import StubBackend, TransformersBackend
from release import verify_files


def create_app(root, warmup_seconds=0, backend_factory=StubBackend, debug=False):
    root = Path(root)
    state = {'ready':False,'backend':None,'error':None,'manifest':None}
    mutex = threading.Lock()  # one compute slot, waiting is measured separately

    def load():
        try:
            manifest = json.loads((root/'accepted.json').read_text())
            verify_files(manifest,root)
            if backend_factory is TransformersBackend:
                from provenance import verify_sources
                verify_sources(manifest,root)
            backend = backend_factory(manifest,root)
            state['backend'] = backend
            state['manifest'] = manifest
            time.sleep(warmup_seconds)
            backend.warmup()
            state['ready'] = True
        except Exception as exc:
            state['error'] = f'{type(exc).__name__}: {exc}'

    @asynccontextmanager
    async def lifespan(app):
        loader = threading.Thread(target=load,daemon=True)
        loader.start()
        yield
        loader.join(timeout=1)

    app = FastAPI(lifespan=lifespan)

    @app.get('/health')
    def health():
        return {'alive':True}

    @app.get('/ready')
    def ready():
        if not state['ready']:
            raise HTTPException(503,'Loading or load failure')
        return {'ready':True,'model_id':state['backend'].loaded_id,
                'runtime_sha256':state['backend'].runtime_sha256}

    if debug:
        @app.get('/debug')
        def diagnostic():
            b = state['backend']
            return {'loaded_id':b.loaded_id if b else None,'calls':b.calls if b else 0,
                    'warmup_finished':state['ready'],'error':state['error']}

    @app.post('/generate')
    async def generate(request: Request):
        started = time.perf_counter()
        if not state['ready']:
            raise HTTPException(503,'Model is not ready')
        # Stream a bounded body instead of trusting Content-Length.
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>65536:
                raise HTTPException(413,'Body too large')
        try:
            payload = json.loads(body)
        except (ValueError,UnicodeDecodeError):
            raise HTTPException(400,'Invalid JSON')
        if not isinstance(payload,dict) or set(payload)-{'prompt','max_new_tokens'}:
            raise HTTPException(422,'Expected prompt and optional max_new_tokens')
        prompt = payload.get('prompt')
        maximum = state['manifest']['runtime']['generation']['max_new_tokens']
        limit = payload.get('max_new_tokens',maximum)
        if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>16000:
            raise HTTPException(422,'Prompt must contain 1..16000 characters')
        # Exactly evaluated generation length: shorter decoding has no inherited gate.
        if type(limit) is not int or limit != maximum:
            raise HTTPException(422,f'max_new_tokens must equal evaluated limit {maximum}')
        b = state['backend']
        def compute():
            with mutex:
                compute_started = time.perf_counter()
                try: b.validate_prompt(prompt)
                except ValueError as exc: raise HTTPException(422,str(exc)) from exc
                result = b.generate(prompt,limit)
                result.update(request_id=str(uuid.uuid4()),model_id=b.loaded_id,runtime_sha256=b.runtime_sha256,
                              queue_seconds=compute_started-started,
                              generation_seconds=time.perf_counter()-compute_started,
                              total_seconds=time.perf_counter()-started)
                return result
        return await asyncio.to_thread(compute)

    return app


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,default=Path('.lab'))
    p.add_argument('--port',type=int,default=8091)
    p.add_argument('--warmup-seconds',type=float,default=5)
    p.add_argument('--backend',choices=['stub','transformers'],default='stub')
    p.add_argument('--debug',action='store_true')
    args=p.parse_args()
    import uvicorn
    # Compatibility: STUB launcher exposes debug for integration; real API requires opt-in.
    uvicorn.run(create_app(args.workspace,args.warmup_seconds,
                          StubBackend if args.backend=='stub' else TransformersBackend,
                          debug=args.debug or args.backend=='stub'),host='127.0.0.1',port=args.port)


if __name__=='__main__': main()
