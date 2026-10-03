"""CPU regression tests: masking, evaluation, LoRA placement and RNG."""
import unittest
from types import SimpleNamespace
import torch
from peft import get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM
from src.data import pad_batch, batches
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


if __name__ == '__main__':
    unittest.main()
