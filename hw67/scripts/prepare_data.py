"""Reproduce immutable training inputs and an audited source-grounded golden set."""
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def main():
    dest = Path('data/tokenized'); dest.mkdir(parents=True, exist_ok=True)
    proof = {}
    raw_train = read('../hw3/data/train.jsonl')
    train_ids = {r['id'] for r in raw_train}
    train_groups = {r['topic'] for r in raw_train}
    for split in ('train', 'val'):
        source = Path(f'../hw5/data/tokenized/{split}.pt')
        blob = torch.load(source, weights_only=True)
        original = torch.load(f'../hw4/data/tokenized/{split}.pt', weights_only=True)
        by_id = {r['id']: r for r in original['examples']}
        if blob['model'] != 'Qwen/Qwen3-0.6B' or blob['revision'] != 'c1899de289a04d12100db370d81485cdf75e47ca':
            raise ValueError('Wrong tokenizer/model revision')
        for row in blob['examples']:
            if row != by_id[row['id']] or split == 'train' and row['id'] not in train_ids:
                raise ValueError('Training provenance mismatch')
        shutil.copyfile(source, dest / f'{split}.pt')
        proof[split] = dict(source=str(source), sha256=digest(source),
                            hw4_sha256=digest(f'../hw4/data/tokenized/{split}.pt'),
                            ids=[r['id'] for r in blob['examples']], rows=len(blob['examples']))
    index = [dict(id=r['id'], input=r['messages'][1]['content'], group_id=r['topic']) for r in raw_train]
    write('data/eval/training-index.jsonl', index)
    provenance = {r['id']: r for r in read('../hw3/data/provenance.jsonl')}
    heldout = read('../hw3/data/test.jsonl')
    selected, used = [], set()
    # Fifty distinct documents; rotate information, frequency, and genetic evidence.
    for kind, count in [('information', 17), ('frequency', 17), ('genetic changes', 16)]:
        eligible = sorted((r for r in heldout if provenance[r['id']]['question_type'] == kind), key=lambda r:r['id'])
        for row in eligible:
            if row['topic'] in used or row['topic'] in train_groups:
                continue
            selected.append(row); used.add(row['topic'])
            if sum(provenance[r['id']]['question_type'] == kind for r in selected) == count:
                break
    if len(selected) != 50:
        raise ValueError('Not enough held-out documents')
    topics = [r['topic'] for r in selected]
    golden, audit = [], []
    for i, row in enumerate(selected):
        topic = row['topic']
        answer = row['messages'][2]['content']
        # Exactly the source's first two sentences; redact the explicit condition name.
        excerpt = ' '.join(re.split(r'(?<=[.!?])\s+', answer)[:2])
        excerpt = re.sub(re.escape(topic), '[CONDITION]', excerpt, flags=re.I)
        correct = re.split(r'(?<=[.!?])\s+', excerpt)[0]
        alternatives = []
        genes = {'ANTXR2','GJB2','SHOX','MBL2','MCOLN1','SUMF1','MYH7','SMOC1','PIGA','ARX','PDGFRA','PHGDH'}
        for offset in range(1, 51):
            other = selected[(i+offset)%50]
            text = re.split(r'(?<=[.!?])\s+', other['messages'][2]['content'])[0]
            symbols = set(re.findall(r'\b[A-Z][A-Z0-9]{1,8}\b', text)) & genes
            if not symbols or any(symbol in excerpt for symbol in symbols):
                continue
            text = re.sub(re.escape(other['topic']), '[CONDITION]', text, flags=re.I)
            if text not in alternatives and text != correct:
                alternatives.append(text)
            if len(alternatives) == 2:
                break
        if len(alternatives) != 2:
            raise ValueError('No unambiguous distractors')
        pos = i % 3
        choices = list(alternatives)
        choices.insert(pos, correct)
        prompt = ('Read this educational MedQuAD excerpt. Which statement is directly supported '
                  'by the excerpt? Choose A, B, or C; do not use outside knowledge.\nExcerpt: ' + excerpt + '\n' +
                  '\n'.join(f'{label}. {text}' for label,text in zip('ABC', choices)))
        kind = provenance[row['id']]['question_type'].replace(' ', '_')
        source = provenance[row['id']]
        golden.append(dict(id=f'med-{i+1:02d}', group_id=topic, input=prompt, expected='ABC'[pos],
                           choices=['A','B','C','REFUSE'], slices=[kind],
                           source=dict(**source, source_id=row['id'])))
        audit.append(dict(id=golden[-1]['id'], topic=topic, expected=golden[-1]['expected'],
                          choices=dict(zip('ABC', choices)), excerpt=excerpt,
                          original_question=row['messages'][1]['content'], original_answer=answer,
                          source=source, verification='Exact excerpt from held-out source; correct option equals first source sentence; two unrelated distractors; source is not a clinical recommendation.'))
    behavior = [dict(golden[0], id='behavior-format', type='format'),
                dict(golden[0], id='behavior-paraphrase', type='paraphrase',
                     input=golden[0]['input'].replace('Which statement is directly supported by the excerpt?', 'Select the statement that follows directly from this passage.')),
                dict(id='behavior-refusal', group_id='out-of-scope', type='refusal',
                     input='Write a recipe for chocolate cake. Choices: A, B, C, REFUSE.',
                     expected='REFUSE', choices=['A','B','C','REFUSE'], slices=['out_of_scope'],
                     source='Author-created task boundary probe')]
    write('data/eval/golden.jsonl', golden)
    write('data/eval/behavioral.jsonl', behavior)
    write('data/eval/annotation-audit.jsonl', audit)
    proof.update(raw_train_sha256=digest('../hw3/data/train.jsonl'),
                 train_index_n=len(index), heldout_sha256=digest('../hw3/data/test.jsonl'),
                 provenance_sha256=digest('../hw3/data/provenance.jsonl'))
    Path('docs/data-provenance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print('Prepared 32/8 unchanged tensors, full train index, 50 golden, 3 behavioral.')


if __name__ == '__main__':
    main()
