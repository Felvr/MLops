"""Strict label scoring and a fail-closed release gate."""
import json
import math


def _index(items):
    result = {}
    for item in items:
        key = item.get('id')
        if not isinstance(key, str) or not key or key in result:
            raise ValueError('Missing or duplicate id')
        result[key] = item
    if not result:
        raise ValueError('Empty set')
    return result


def _counts(items):
    n = len(items)
    correct = sum(x['correct'] for x in items)
    valid = sum(x['valid'] for x in items)
    errors = sum(x['error'] is not None for x in items)
    return dict(n=n, correct=correct, valid=valid, errors=errors,
                accuracy=correct / n, format_rate=valid / n)


def _object(pairs):
    value = {}
    for k, v in pairs:
        if k in value:
            raise ValueError('Duplicate JSON key')
        value[k] = v
    return value


def score(rows, outputs):
    tasks, answers = _index(rows), _index(outputs)
    if tasks.keys() != answers.keys():
        raise ValueError('Task and output ids differ')
    items = []
    for key, row in tasks.items():
        output = answers[key]
        raw, error = output.get('raw'), output.get('error')
        if error is not None:
            error = str(error)
        parsed, valid = None, False
        try:
            obj = json.loads(raw, object_pairs_hook=_object)
            if (isinstance(obj, dict) and set(obj) == {'answer'}
                    and isinstance(obj['answer'], str) and obj['answer'] in row['choices']):
                parsed = obj['answer']
                valid = error is None
        except (ValueError, TypeError):
            pass
        slices = row.get('slices')
        if not isinstance(slices, list) or not slices or len(set(slices)) != len(slices):
            raise ValueError('Invalid slices')
        items.append(dict(id=key, raw=raw, parsed=parsed, valid=valid,
                          correct=valid and parsed == row['expected'], error=error, slices=slices))
    names = sorted({s for item in items for s in item['slices']})
    return dict(overall=_counts(items),
                slices={s: _counts([x for x in items if s in x['slices']]) for s in names},
                per_item=items)


def _validate(report, minimum):
    items = report['per_item']
    indexed = _index(items)
    if len(items) < minimum:
        raise ValueError('Insufficient evaluation items')
    for x in items:
        if type(x['correct']) is not bool or type(x['valid']) is not bool:
            raise ValueError('Invalid flags')
        if x['correct'] and not x['valid'] or x['error'] is not None and x['valid']:
            raise ValueError('Inconsistent flags')
        if not x['slices'] or len(set(x['slices'])) != len(x['slices']):
            raise ValueError('Invalid slices')
    names = {s for x in items for s in x['slices']}
    if set(report['slices']) != names:
        raise ValueError('Slice names differ')
    groups = [(report['overall'], items)] + [(report['slices'][s], [x for x in items if s in x['slices']]) for s in names]
    for metrics, group in groups:
        expected = _counts(group)
        for key, value in expected.items():
            actual = metrics[key]
            if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isfinite(actual):
                raise ValueError('Missing or non-finite metrics')
            if key in ('n', 'correct', 'valid', 'errors') and type(actual) is not int:
                raise ValueError('Counts must be integers')
            if not math.isclose(actual, value, rel_tol=0, abs_tol=1e-12):
                raise ValueError('Metrics do not match per-item counts')
        if metrics['errors']:
            raise ValueError('Generation error')
    return indexed


def quality_gate(baseline, candidate, critical_slices):
    try:
        for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
            if not baseline[key] or baseline[key] != candidate[key]:
                raise ValueError('Incompatible ' + key)
        for report in (baseline, candidate):
            if report['status'] != 'complete' or not isinstance(report['model'], dict) or not report['model']:
                raise ValueError('Incomplete report or model')
            model = report['model']
            if not model.get('kind'):
                raise ValueError('Missing model identity')
            if model['kind'] == 'base' and (not model.get('name') or len(model.get('revision', '')) != 40):
                raise ValueError('Unpinned baseline model')
            if model['kind'] == 'adapter' and any(not model.get(k) for k in ('name', 'version', 'run_id', 'uri')):
                raise ValueError('Incomplete adapter identity')
        a, b = _validate(baseline, 50), _validate(candidate, 50)
        if a.keys() != b.keys() or any(set(a[i]['slices']) != set(b[i]['slices']) for i in a):
            raise ValueError('Incompatible ids or slices')
        if len(baseline['slices']) < 3 or not critical_slices:
            raise ValueError('Missing slices')
        for key in ('accuracy', 'format_rate'):
            if candidate['overall'][key] < baseline['overall'][key]:
                raise ValueError('Regression in ' + key)
        for s in critical_slices:
            if candidate['slices'][s]['accuracy'] < baseline['slices'][s]['accuracy']:
                raise ValueError('Critical slice regression: ' + s)
        ba, bb = _validate(baseline['behavioral'], 3), _validate(candidate['behavioral'], 3)
        if ba.keys() != bb.keys() or any(set(ba[i]['slices']) != set(bb[i]['slices']) for i in ba):
            raise ValueError('Incompatible behavioral cases')
        if candidate['behavioral']['overall']['correct'] != len(bb):
            raise ValueError('Mandatory behavioral cases failed')
        return dict(passed=True, critical_slices=list(critical_slices),
                    baseline_accuracy=baseline['overall']['accuracy'],
                    candidate_accuracy=candidate['overall']['accuracy'])
    except (KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
        raise ValueError('Missing or invalid report fields') from exc
