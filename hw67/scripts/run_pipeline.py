"""One reproducible route; quality rejection is recorded, never bypassed."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(*command):
    print('+ ' + ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-sweep', action='store_true', help='Use a completed sweep, without retraining')
    args = parser.parse_args()
    if not args.after_sweep:
        for target in ('up', 'prepare', 'data-version', 'sweep'):
            run('make', target)
    for target in ('report', 'candidate'):
        run('make', target)
    if not Path('evaluation/baseline.json').exists():
        run('make', 'baseline-init')
    run('make', 'evaluate')
    result = subprocess.run([sys.executable, '-m', 'evaluation.cli', 'gate'])
    if result.returncode:
        gate_path = Path('metrics/quality-gate.json')
        if not gate_path.exists() or json.loads(gate_path.read_text()).get('passed') is not False:
            raise RuntimeError('Gate failed without a recorded quality decision')
        print('Quality release refused; keeping candidate and baseline.', flush=True)
    else:
        run('make', 'promote')
    for target in ('check-negative', 'check-storage'):
        run('make', target)
    run(sys.executable, '-m', 'scripts.finalize_report')
    run('make', 'check-all')
    print('Finished. See docs/conclusion.md; record your own video.', flush=True)


if __name__ == '__main__':
    main()
