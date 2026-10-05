"""Prepare unchanged HW4 examples, with an optional deterministic short subset."""
import hashlib
import shutil
from pathlib import Path

import torch
from src.config import load_params


def select_examples(blob, config, split):
    """Select whole examples within their original split, without truncating tokens."""
    examples = blob['examples']
    if not config.get('enabled', False):
        return examples
    size = config[f'{split}_size']
    max_tokens = config['max_example_tokens']
    if not isinstance(size, int) or size < 1 or max_tokens < 2:
        raise ValueError('Subset sizes must be positive, max_example_tokens >= 2')
    eligible = [e for e in examples if len(e['input_ids']) <= max_tokens]
    if len(eligible) < size:
        raise ValueError(f'{split}: only {len(eligible)} eligible examples, requested {size}')
    def key(e):
        identifier = str(e['id'])
        digest = hashlib.sha256(f"{config['seed']}:{identifier}".encode()).hexdigest()
        return digest, identifier
    return sorted(eligible, key=key)[:size]


def main():
    params = load_params()
    subset = params['data'].get('subset', {})
    enabled = subset.get('enabled', False)
    description = ('Быстрый эксперимент на сокращённых сплитах ДЗ4. Отбираются целые примеры; '
                   'токены и маски не изменяются, train и val не смешиваются.' if enabled else
                   'Точные копии собственных тензоров ДЗ4 без отбора или повторной токенизации.')
    lines = ['# Входные данные ДЗ5', '', description, '',
             '| Сплит | В ДЗ4 | В эксперименте | Токенов | SHA-256 выхода |',
             '|---|---:|---:|---:|---|']
    selected_ids = {}
    for split in ('train', 'val'):
        source = Path(params['data'][f'source_{split}'])
        target = Path(params['data'][split])
        if not source.exists():
            raise SystemExit(f'Нет {source}: сначала выполните make repro в hw4')
        blob = torch.load(source, weights_only=True)
        if blob['model'] != params['model']['name'] or blob['revision'] != params['model']['revision']:
            raise ValueError('Модель/revision ДЗ4 не совпадает с ДЗ5')
        source_count = len(blob['examples'])
        examples = select_examples(blob, subset, split)
        target.parent.mkdir(parents=True, exist_ok=True)
        if enabled:
            blob = {**blob, 'examples': examples, 'subset': {
                'source_count': source_count,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'seed': subset['seed'], 'max_example_tokens': subset['max_example_tokens'],
                'selection': 'sha256(seed:id), whole examples within original split',
            }}
            torch.save(blob, target)
        else:
            shutil.copy2(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        tokens = sum(len(e['input_ids']) for e in examples)
        lines.append(f'| {split} | {source_count} | {len(examples)} | {tokens} | `{digest}` |')
        selected_ids[split] = [e['id'] for e in examples]
    if enabled:
        lines += ['', f"Отбор: SHA-256 от `seed:id`, seed={subset['seed']}; "
                  f"не более {subset['max_example_tokens']} токенов в примере.",
                  'Фильтр коротких ответов ускоряет CPU, но меняет распределение данных; '
                  'выводы этого малого эксперимента нельзя переносить на весь MedQuAD.', '',
                  '## Идентификаторы выбранных примеров', '']
        for split, ids in selected_ids.items():
            lines += [f"**{split}:** " + ', '.join(f'`{i}`' for i in ids), '']
    lines += ['', f'Модель: `{blob["model"]}`, revision: `{blob["revision"]}`.',
              f'Исходный лимит токенизации ДЗ4: {blob["max_seq_len"]}; padding: {blob["padding_side"]}.',
              'Групповое разделение по заболеваниям выполнено в ДЗ3; маска ответа проверена в ДЗ4.',
              'Источник и лицензия: [паспорт данных ДЗ3](../../hw3/docs/datasheet.md).',
              'Согласование темы преподавателем и сокращённого режима не подтверждено.', '']
    Path('docs/data.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
