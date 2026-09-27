"""Пограничные случаи маски, обрезки и батчей сверх проверки курса."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import warnings

from transformers import AutoTokenizer
from src.collate import DynamicPaddingCollator
from src.config import load_params
from src.prompt import build_chat_text, prompt_token_len
from src.tokenize_data import encode_example, process_split, truncation_stats


class TokenizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = load_params()
        cls.tok = AutoTokenizer.from_pretrained(cls.params['model']['name'], revision=cls.params['model']['revision'])
        cls.record = {'id': 'regression', 'messages': [
            {'role': 'system', 'content': 'Отвечай кратко.'},
            {'role': 'user', 'content': 'Сколько будет два плюс два?'},
            {'role': 'assistant', 'content': 'Четыре.'}]}

    def test_mask_and_eos(self):
        ex = encode_example(self.tok, self.record, self.params, 512)
        n = ex['_meta']['prompt_len']
        self.assertGreater(n, 0)
        self.assertEqual(ex['labels'][:n], [-100] * n)
        ids = ex['labels'][n:]
        self.assertEqual(ids, ex['input_ids'][n:])
        self.assertIn(self.tok.eos_token_id, ids)
        self.assertEqual(self.tok.decode(ids, skip_special_tokens=True).strip(), 'Четыре.')

    def test_inference_with_and_without_answer(self):
        full = build_chat_text(self.tok, self.record['messages'], self.params, False)
        prefix = build_chat_text(self.tok, self.record['messages'], self.params, True)
        actual = build_chat_text(self.tok, self.record['messages'][:-1], self.params, True)
        self.assertEqual(actual, prefix)
        self.assertTrue(full.encode().startswith(prefix.encode()))

    def test_multiturn_only_last_answer(self):
        record = copy.deepcopy(self.record)
        record['messages'] += [{'role': 'user', 'content': 'А три плюс три?'},
                               {'role': 'assistant', 'content': 'Шесть.'}]
        ex = encode_example(self.tok, record, self.params, 512)
        ids = [i for i in ex['labels'] if i != -100]
        self.assertEqual(self.tok.decode(ids, skip_special_tokens=True).strip(), 'Шесть.')

    def test_exact_limit_and_answer_removed(self):
        ex = encode_example(self.tok, self.record, self.params, 512)
        exact = encode_example(self.tok, self.record, self.params, ex['_meta']['full_len'])
        self.assertFalse(exact['_meta']['truncated'])
        removed = encode_example(self.tok, self.record, self.params, ex['_meta']['prompt_len'])
        self.assertTrue(removed['_meta']['truncated'])
        self.assertEqual(removed['_meta']['supervised'], 0)
        self.assertEqual(len(removed['labels']), len(removed['input_ids']))

    def test_drop_count_uses_all_inputs(self):
        long = copy.deepcopy(self.record)
        long['id'] = 'long'
        long['messages'][1]['content'] *= 100
        params = copy.deepcopy(self.params)
        params['tokenize']['max_seq_len'] = 128
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'train.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in (self.record, long)))
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always')
                examples, stats, _ = process_split(self.tok, 'test', path, params)
            self.assertEqual(len(examples), 1)
            self.assertEqual(stats['examples_in'], 2)
            self.assertEqual(stats['dropped_no_supervision'], 1)
            self.assertEqual(stats['truncated_ratio'], 0.5)
            self.assertEqual(len(observed), 1)

    def test_left_padding_values_and_dynamic_width(self):
        features = [{'input_ids': [11, 12], 'attention_mask': [1, 1], 'labels': [-100, 12]},
                    {'input_ids': [21, 22, 23, 24], 'attention_mask': [1] * 4, 'labels': [-100, 22, 23, 24]}]
        batch = DynamicPaddingCollator(self.tok.pad_token_id)(features)
        self.assertEqual(tuple(batch['input_ids'].shape), (2, 4))
        self.assertEqual(batch['attention_mask'][0].tolist(), [0, 0, 1, 1])
        self.assertEqual(batch['labels'][0].tolist(), [-100, -100, -100, 12])
        self.assertEqual(batch['input_ids'][0, :2].tolist(), [self.tok.pad_token_id] * 2)

    def test_bpe_cross_boundary_masked(self):
        prompt, text = 'Приве', 'Привет, мир'
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        n, fallback = prompt_token_len(self.tok, prompt, enc['input_ids'], enc['offset_mapping'])
        self.assertTrue(fallback)
        self.assertGreaterEqual(enc['offset_mapping'][n][0], len(prompt))
        self.assertTrue(all(start < len(prompt) for start, _ in enc['offset_mapping'][:n]))

    def test_warning_threshold_equality(self):
        params = copy.deepcopy(self.params)
        params['tokenize']['truncated_warn_ratio'] = 0.5
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            stats = truncation_stats([{'truncated': True}, {'truncated': False}], 'test', params)
        self.assertEqual(stats['truncated_ratio'], 0.5)
        self.assertFalse(observed)


if __name__ == '__main__':
    unittest.main()
