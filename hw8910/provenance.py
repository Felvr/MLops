"""Re-score raw HW6-7 outputs and invoke its unchanged quality gate."""
import importlib.util
import json
from pathlib import Path
from release import canonical, digest, file_digest, BINDING

HOME = Path(__file__).resolve().parent
HW67 = HOME.parent/'hw67'
spec = importlib.util.spec_from_file_location('hw67_gate',HW67/'evaluation/solution.py')
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)


def legacy_fingerprint(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode())


def read(path):
    return json.loads(Path(path).read_text())


def verify_sources(manifest, root):
    root = Path(root)
    if file_digest(HW67/'evaluation/solution.py') != manifest['source_hashes']['evidence/evaluation-code/solution.py']:
        raise ValueError('Quality gate implementation changed; reevaluate')
    for name, expected in manifest['source_hashes'].items():
        if file_digest(root/name) != expected:
            raise ValueError('Evaluation source changed: '+name)
    params = read(root/'evidence/params.json')
    runtime = manifest['runtime']
    g = runtime['generation']
    for key in ('system','max_new_tokens','enable_thinking','seed'):
        if g[key] != params[key]:
            raise ValueError('Evaluation/API runtime differs: '+key)
    if g['do_sample'] is not False or runtime['dtype']!='float32' or runtime['device']!='cpu':
        raise ValueError('Saved evaluation used CPU float32 greedy decoding')
    base_spec = read(root/'evidence/baseline-spec.json')
    if runtime['base_revision']!=base_spec['revision'] or runtime['base_name']!=base_spec['name']:
        raise ValueError('Foreign base revision')
    if manifest['golden_sha256'] != legacy_fingerprint(rows(root/'evidence/golden.jsonl')):
        raise ValueError('Golden mismatch')
    # Rebuild exact original protocol, including code text, not its new location.
    protocol = dict(config=params,golden=rows(root/'evidence/golden.jsonl'),
                    behavioral=rows(root/'evidence/behavioral.jsonl'),
                    training_index=rows(root/'evidence/training-index.jsonl'),
                    evaluation_code={p.name:p.read_text() for p in sorted((root/'evidence/evaluation-code').glob('*.py'))})
    if legacy_fingerprint(protocol)!=manifest['protocol_sha256']:
        raise ValueError('Protocol mismatch')
    for role in ('baseline','candidate'):
        report = read(root/f'evidence/{role}-eval.json')
        outputs = read(root/f'evidence/{role}-outputs.json')
        if legacy_fingerprint(outputs)!=report['outputs_sha256']:
            raise ValueError('Raw outputs hash mismatch')
        for dataset,section in [('golden.jsonl',report),('behavioral.jsonl',report['behavioral'])]:
            examples=rows(root/'evidence'/dataset)
            ids={row['id'] for row in examples}
            rescored=solution.score(examples,[out for out in outputs if out['id'] in ids])
            for key in ('overall','slices','per_item'):
                if rescored[key]!=section[key]: raise ValueError('Score mismatch')
        if report['protocol_sha256']!=manifest['protocol_sha256'] or report['golden_sha256']!=manifest['golden_sha256']:
            raise ValueError('Foreign report protocol')
    return params


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def make_report(manifest, root):
    params = verify_sources(manifest,root)
    baseline = read(root/'evidence/baseline-eval.json')
    candidate = read(root/'evidence/candidate-eval.json')
    if manifest['model_spec']!=candidate['model']:
        raise ValueError('Candidate identity mismatch')
    try:
        decision=solution.quality_gate(baseline,candidate,params['critical_slices'])
    except ValueError as exc:
        decision={'passed':False,'reason':str(exc)}
    return {**{key:manifest[key] for key in BINDING},'status':'complete',
            'passed':decision['passed'],'decision':decision,
            'evaluation_run_id':candidate['evaluation_run_id'],
            'metrics':{key:candidate['overall'][key] for key in ('accuracy','format_rate')},
            'source_hashes':manifest['source_hashes'],
            'training_run_id':candidate['model']['run_id']}
