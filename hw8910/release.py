"""Fail-closed metadata contract and single-writer atomic pointer."""
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

BINDING = ('model_id', 'model_sha256', 'protocol_sha256', 'golden_sha256', 'runtime_sha256')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def cache_key(data_path: Path, config: dict) -> str:
    return digest(canonical({'data_sha256': file_digest(data_path), 'config': config}))


def validate_manifest(manifest):
    try:
        if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
            raise ValueError('schema_version must be int 1')
        for key in ('model_id', 'artifact_uri'):
            if not isinstance(manifest[key], str) or not manifest[key].strip():
                raise ValueError('Empty ' + key)
        for key in BINDING[1:]:
            if not isinstance(manifest[key], str) or not re.fullmatch('[0-9a-f]{64}', manifest[key]):
                raise ValueError('Invalid ' + key)
        if not isinstance(manifest['runtime'], dict) or not manifest['runtime']:
            raise ValueError('Missing runtime')
        if digest(canonical(manifest['runtime'])) != manifest['runtime_sha256']:
            raise ValueError('Stale runtime hash')
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete manifest') from exc


def validate_release(manifest: dict, report: dict) -> None:
    validate_manifest(manifest)
    try:
        if report['status'] != 'complete' or report['passed'] is not True:
            raise ValueError('Incomplete evaluation or quality rejection')
        if not isinstance(report['evaluation_run_id'], str) or not report['evaluation_run_id'].strip():
            raise ValueError('Missing evaluation run')
        if any(report[key] != manifest[key] for key in BINDING):
            raise ValueError('Foreign evaluation report')
        for key in ('accuracy', 'format_rate'):
            value = report['metrics'][key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid metric ' + key)
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete report') from exc


def atomic_write(target, payload):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.tmp', delete=False) as f:
            name = f.name
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, target)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def write_json(target, value):
    atomic_write(target, canonical(value) + b'\n')


def tokenizer_hash(path):
    names = {'tokenizer.json','tokenizer_config.json','special_tokens_map.json',
             'added_tokens.json','vocab.json','merges.txt','chat_template.jinja'}
    return digest(canonical({p.name:file_digest(p) for p in sorted(Path(path).iterdir()) if p.name in names}))


def verify_files(manifest, root):
    """Verify bytes before publication AND loading; no weights in task results."""
    validate_manifest(manifest)
    files = manifest.get('file_hashes')
    if not isinstance(files, dict) or not files:
        raise ValueError('Missing artifact inventory')
    base = (Path(root) / manifest.get('artifact_root', '.')).resolve()
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe artifact path')
        path = base / relative
        # HF snapshot files are legitimate symlinks into the local blob store.
        if file_digest(path) != expected:
            raise ValueError('Artifact bytes changed: ' + name)
    if manifest.get('tokenizer_uri') and tokenizer_hash(Path(root)/manifest['tokenizer_uri']) != manifest['runtime']['tokenizer_sha256']:
        raise ValueError('Tokenizer inventory differs from evaluated runtime')
    if digest(canonical(files)) != manifest['model_sha256']:
        raise ValueError('Invalid inventory digest')
    return base


def publish(manifest: dict, report: dict, target: Path) -> None:
    validate_release(manifest, report)
    write_json(target, manifest)


def release_id(manifest, report):
    return digest(canonical({'manifest': manifest, 'decision': report}))


def publish_event(root, manifest, report, fail_after_pointer=False):
    """Durable intent -> pointer -> idempotent event. One writer, recover on retry.

    An interrupted release must be recovered before any different release.
    Journal is a JSONL snapshot atomically replaced, not a blind append.
    """
    root = Path(root)
    validate_release(manifest, report)
    verify_files(manifest, root)
    pending = root / 'pending.json'
    identity = release_id(manifest, report)
    if pending.exists():
        old = json.loads(pending.read_text())
        if old['release_id'] != identity:
            publish_event(root, old['manifest'], old['report'])
    write_json(pending, {'release_id': identity, 'manifest': manifest, 'report': report})
    publish(manifest, report, root / 'accepted.json')
    if fail_after_pointer:
        raise RuntimeError('Injected crash between pointer and event')
    log = root / 'publications.jsonl'
    events = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    if identity not in {event['release_id'] for event in events}:
        events.append({'release_id': identity, 'model_id': manifest['model_id'],
                       'runtime_sha256': manifest['runtime_sha256'],
                       'evaluation_run_id': report['evaluation_run_id']})
        atomic_write(log, b''.join(canonical(event) + b'\n' for event in events))
    pending.unlink(missing_ok=True)
    return identity
