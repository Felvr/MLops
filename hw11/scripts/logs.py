"""Persist structured logs across Pod replacements. Stop with SIGTERM or Ctrl+C."""
import asyncio
import json
import os
from pathlib import Path
import signal

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("KUBECONFIG", str(ROOT / ".local/kubeconfig"))


async def collect():
    path = ROOT / "results/server.jsonl"
    seen = set(path.read_text().splitlines()) if path.exists() else set()
    while True:
        process = await asyncio.create_subprocess_exec(
            "kubectl", "--context", "kind-mlops-hw11", "-n", "hw11", "logs",
            "-l", "app=hw11-api", "--since=15s", "--max-log-requests=10", "--tail=-1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            output,_ = await process.communicate()
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        with path.open("a") as log:
            for line in output.decode().splitlines():
                if line in seen:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and "event" in event and "pod" in event:
                    log.write(line + "\n")
                    seen.add(line)
        await asyncio.sleep(2)


async def main():
    task = asyncio.create_task(collect())
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM,signal.SIGINT):
        loop.add_signal_handler(signum, task.cancel)
    await asyncio.gather(task,return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
