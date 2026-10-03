"""Copy exact HW4 tensors and write their provenance; never retokenize/subsample."""
import hashlib
import shutil
from pathlib import Path

import torch
from src.config import load_params


def main():
    params = load_params()
    lines = ['# Входные данные ДЗ5', '',
             'Точные копии собственных тензоров ДЗ4 (MedQuAD, GHR). Без повторной токенизации и отбора.', '',
             '| Сплит | Примеров | SHA-256 |', '|---|---:|---|']
    for split in ('train', 'val'):
        source = Path(params['data'][f'source_{split}'])
        target = Path(params['data'][split])
        if not source.exists():
            raise SystemExit(f'Нет {source}: сначала выполните make repro в hw4')
        blob = torch.load(source, weights_only=True)
        if blob['model'] != params['model']['name'] or blob['revision'] != params['model']['revision']:
            raise ValueError('Модель/revision ДЗ4 не совпадает с ДЗ5')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        lines.append(f'| {split} | {len(blob["examples"])} | `{digest}` |')
    lines += ['', f'Модель: `{blob["model"]}`, revision: `{blob["revision"]}`.',
              f'Предельная длина: {blob["max_seq_len"]}; padding: {blob["padding_side"]}.',
              'Групповое разделение по заболеваниям выполнено в ДЗ3; маска ответа проверена в ДЗ4.',
              'Источник и лицензия: [паспорт данных ДЗ3](../../hw3/docs/datasheet.md).',
              'Согласование темы преподавателем не подтверждено.', '']
    Path('docs/data.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
