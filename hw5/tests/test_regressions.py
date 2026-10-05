"""CPU regression tests: masking, evaluation, LoRA placement and RNG."""
import unittest
from types import SimpleNamespace
import torch
from peft import get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM
from src.data import pad_batch, batches, validate_subset
from scripts.prepare_data import select_examples
from src.runtime import set_seed
from src.train import evaluate, lora_config


def example(ids, labels):
    return dict(input_ids=ids, attention_mask=[1]*len(ids), labels=labels)


class RegressionTests(unittest.TestCase):
    def test_left_padding_masks_loss(self):
        result = pad_batch([example([1, 2, 3], [-100, 2, 3]), example([4, 5], [-100, 5])], 0)
        self.assertEqual(result['input_ids'].tolist(), [[1, 2, 3], [0, 4, 5]])
        self.assertEqual(result['labels'].tolist(), [[-100, 2, 3], [-100, -100, 5]])
        self.assertEqual(result['attention_mask'].tolist(), [[1, 1, 1], [0, 1, 1]])

    def test_shuffle_reproducible_and_no_dropped_tail(self):
        data = [example([i, i], [-100, i]) for i in range(1, 12)]
        def order(seed):
            return [i for b in batches(data, 4, 0, True, seed) for i in b['input_ids'][:, 0].tolist()]
        self.assertEqual(order(42), order(42))
        self.assertNotEqual(order(42), order(43))
        self.assertEqual(sorted(order(42)), list(range(1, 12)))

    def test_evaluation_is_token_weighted_and_restores_mode(self):
        class Fake(torch.nn.Module):
            def forward(self, **batch):
                self.assert_eval = not self.training
                return SimpleNamespace(loss=batch['input_ids'][0, 0].float())
        model = Fake()
        data = [example([2, 1], [-100, 1]), example([8, 1, 1, 1], [-100, 1, 1, 1])]
        self.assertEqual(evaluate(model, data, 0, torch.device('cpu'), 1), (2*1+8*3)/4)
        self.assertTrue(model.training)
        self.assertTrue(model.assert_eval)
        model.eval()
        evaluate(model, data, 0, torch.device('cpu'), 1)
        self.assertFalse(model.training)

    def test_empty_supervision_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(torch.nn.Linear(1, 1), [example([1, 2], [-100, -100])], 0, torch.device('cpu'), 1)

    def test_lora_placement_and_seed(self):
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=4, num_attention_heads=2,
                            num_key_value_heads=2, head_dim=8)
        params = {'model': {}, 'lora': {'r': 2, 'alpha': 4, 'dropout': .05, 'target_modules': ['q_proj', 'v_proj']}}
        def build(freeze):
            set_seed(42)
            return get_peft_model(Qwen3ForCausalLM(config), lora_config(params, 4, freeze))
        all_layers, frozen, repeated = build(0), build(2), build(2)
        count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.assertEqual(count(all_layers), 2*count(frozen))
        for name, p in frozen.named_parameters():
            if p.requires_grad:
                self.assertTrue('.layers.2.' in name or '.layers.3.' in name)
                self.assertIn('lora_', name)
                self.assertTrue(torch.equal(p, dict(repeated.named_parameters())[name]))
        with self.assertRaises(ValueError):
            lora_config(params, 4, 4)


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.data = [{'id': f'id-{i}', **example([i] * (i + 2), [-100] + [i] * (i + 1))}
                     for i in range(1, 20)]
        self.blob = {'examples': self.data}
        self.config = dict(enabled=True, train_size=5, val_size=3, max_example_tokens=12, seed=42)

    def test_stable_order_independent_selection_without_truncation(self):
        selected = select_examples(self.blob, self.config, 'train')
        reversed_blob = {'examples': list(reversed(self.data))}
        self.assertEqual(selected, select_examples(reversed_blob, self.config, 'train'))
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(len(e['input_ids']) <= 12 for e in selected))
        for item in selected:
            original = next(e for e in self.data if e['id'] == item['id'])
            self.assertIs(item, original)
        different = select_examples(self.blob, {**self.config, 'seed': 43}, 'train')
        self.assertNotEqual(selected, different)

    def test_selection_stays_inside_its_split_and_full_mode_keeps_everything(self):
        validation = {'examples': [{**e, 'id': 'val-' + e['id']} for e in self.data]}
        selected = select_examples(validation, self.config, 'val')
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(e['id'].startswith('val-') for e in selected))
        self.assertIs(select_examples(self.blob, {'enabled': False}, 'train'), self.data)
        with self.assertRaises(ValueError):
            select_examples(self.blob, {**self.config, 'train_size': 100}, 'train')

    def test_stale_full_or_subset_inputs_rejected(self):
        with self.assertRaises(ValueError):
            validate_subset(self.blob, self.config, 'train')
        blob = {'examples': select_examples(self.blob, self.config, 'train'),
                'subset': {'seed': 42, 'max_example_tokens': 12}}
        validate_subset(blob, self.config, 'train')
        with self.assertRaises(ValueError):
            validate_subset(blob, {'enabled': False}, 'train')
        with self.assertRaises(ValueError):
            validate_subset(blob, {**self.config, 'seed': 43}, 'train')


if __name__ == '__main__':
    unittest.main()
