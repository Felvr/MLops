"""Обосновать max_seq_len по исходным длинам и потерям ответов."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from src.config import load_params
from src.tokenize_data import read_jsonl, encode_example, describe


def main():
    params = load_params()
    tok = AutoTokenizer.from_pretrained(params['model']['name'], revision=params['model']['revision'])
    result = {'model': params['model'], 'splits': {}}
    for name in ('train', 'val'):
        records = read_jsonl(Path(params['data'][f'{name}_jsonl']))
        metas = [encode_example(tok, row, params, params['tokenize']['max_seq_len'])['_meta']
                 for row in records]
        rows = []
        for limit in sorted({232, 512, 768, 1024, 1536, 2048, params['tokenize']['max_seq_len']}):
            count = sum(m['full_len'] > limit for m in metas)
            rows.append({'max_seq_len': limit, 'truncated': count,
                         'truncated_ratio': count / len(metas),
                         'dropped_no_supervision': sum(m['prompt_len'] >= limit for m in metas),
                         'supervised_tokens_lost': sum(max(0, m['full_len'] - max(limit, m['prompt_len'])) for m in metas)})
        result['splits'][name] = {'examples': len(records),
                                  'length_tokens': describe([m['full_len'] for m in metas]),
                                  'candidates': rows}
    path = Path('metrics/length_sweep.json')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(path.read_text())


if __name__ == '__main__':
    main()
